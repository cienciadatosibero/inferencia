"""
Microservicio de inferencia para el Generador de Datasets Satelitales.

Mantiene el modelo YOLO cargado en memoria (una sola vez) y expone:
  POST /load      -> carga un peso y devuelve sus clases (sin activarlo como principal)
  POST /activate  -> deja un peso como modelo activo en memoria
  POST /predict   -> detecta sobre una imagen base64, devuelve cajas + recortes
  GET  /health    -> estado

Ejecuta:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
from pathlib import Path
from threading import RLock

from fastapi import Header
from pydantic import Field

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None
    _IMPORT_ERROR = e

app = FastAPI(title="SatDataset Inference")

# Estado global: un modelo activo en memoria + caché de modelos cargados
_estado = {
    "active_path": None,
    "model": None,
    "names": {},
}
_cache: dict = {}  # ruta -> YOLO
_lock = RLock()
_ROOT = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = os.getenv("MODEL_WEIGHTS_PATH", "modelos/best.pt")
_MODEL_IMGSZ = max(640, min(1280, int(os.getenv("YOLO_IMGSZ", "960"))))
_TOKEN = os.getenv("INFERENCE_API_TOKEN", "").strip()

def _check_token(authorization):
    if _TOKEN and authorization != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="No autorizado")



def _cargar_yolo(path: str):
    if YOLO is None:
        raise RuntimeError(f"ultralytics no está disponible: {_IMPORT_ERROR}")
    ruta = Path(path).expanduser()
    if not ruta.is_absolute():
        ruta = _ROOT / ruta
    ruta = str(ruta.resolve())
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No existe el peso: {Path(ruta).name}")
    with _lock:
        if ruta not in _cache:
            _cache[ruta] = YOLO(ruta)
        return _cache[ruta]


# ----------------------- Modelos de petición -----------------------
class LoadReq(BaseModel):
    weights_path: str


class PredictReq(BaseModel):
    image_base64: str
    conf: float = Field(default=0.20, ge=0.05, le=0.95)
    return_crops: bool = True


# ----------------------- Utilidades -----------------------
def _b64_a_imagen(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > 15 * 1024 * 1024:
            raise ValueError("Imagen demasiado grande (máximo 15 MB)")
        imagen = Image.open(io.BytesIO(raw))
        imagen.load()
        if imagen.width * imagen.height > 16_000_000:
            raise ValueError("Imagen supera el límite de 16 megapíxeles")
        return imagen.convert("RGB")
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}") from e


def _recorte_b64(img: Image.Image, x, y, w, h) -> str:
    izq = max(0, int(x))
    arr = max(0, int(y))
    der = min(img.width, int(x + w))
    aba = min(img.height, int(y + h))
    recorte = img.crop((izq, arr, der, aba))
    buf = io.BytesIO()
    recorte.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------- Endpoints -----------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "active_path": _estado["active_path"],
        "classes": _estado["names"],
        "ultralytics": YOLO is not None,
    }


@app.post("/load")
def load(req: LoadReq, authorization: str | None = Header(default=None)):
    """Carga un peso y devuelve sus clases (para registrarlo en la BD)."""
    _check_token(authorization)
    try:
        modelo = _cargar_yolo(req.weights_path)
        names = modelo.names if hasattr(modelo, "names") else {}
        return {"ok": True, "framework": "yolo", "classes": names}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/activate")
def activate(req: LoadReq, authorization: str | None = Header(default=None)):
    """Deja un peso como modelo activo en memoria."""
    _check_token(authorization)
    try:
        modelo = _cargar_yolo(req.weights_path)
        _estado["model"] = modelo
        _estado["active_path"] = str(Path(req.weights_path).resolve())
        _estado["names"] = modelo.names if hasattr(modelo, "names") else {}
        return {"ok": True, "classes": _estado["names"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
def predict(req: PredictReq, authorization: str | None = Header(default=None)):
    """Detecta sobre la imagen y devuelve cajas en píxeles + recortes."""
    _check_token(authorization)
    if _estado["model"] is None:
        try:
            modelo = _cargar_yolo(_DEFAULT_WEIGHTS)
            with _lock:
                if _estado["model"] is None:
                    _estado.update(model=modelo, active_path=str(_DEFAULT_WEIGHTS), names=modelo.names)
        except Exception as e:
            raise HTTPException(status_code=409, detail=f"No hay modelo disponible: {e}")
    try:
        img = _b64_a_imagen(req.image_base64)
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR para YOLO

        with _lock:
            resultados = _estado["model"].predict(source=arr, conf=req.conf,
                                                    imgsz=_MODEL_IMGSZ, iou=0.45, verbose=False)
        r = resultados[0]
        names = _estado["names"]

        detecciones = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                w = x2 - x1
                h = y2 - y1
                det = {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "x": x1,
                    "y": y1,
                    "w": w,
                    "h": h,
                }
                if req.return_crops:
                    det["crop_base64"] = _recorte_b64(img, x1, y1, w, h)
                detecciones.append(det)

        return {
            "ok": True,
            "image_width": img.width,
            "image_height": img.height,
            "detections": detecciones,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
