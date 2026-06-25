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
from typing import Optional

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


def _cargar_yolo(path: str):
    if YOLO is None:
        raise RuntimeError(f"ultralytics no está disponible: {_IMPORT_ERROR}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el peso: {path}")
    if path not in _cache:
        _cache[path] = YOLO(path)
    return _cache[path]


# ----------------------- Modelos de petición -----------------------
class LoadReq(BaseModel):
    weights_path: str


class PredictReq(BaseModel):
    image_base64: str
    conf: float = 0.25
    return_crops: bool = True


# ----------------------- Utilidades -----------------------
def _b64_a_imagen(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


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
def load(req: LoadReq):
    """Carga un peso y devuelve sus clases (para registrarlo en la BD)."""
    try:
        modelo = _cargar_yolo(req.weights_path)
        names = modelo.names if hasattr(modelo, "names") else {}
        return {"ok": True, "framework": "yolo", "classes": names}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/activate")
def activate(req: LoadReq):
    """Deja un peso como modelo activo en memoria."""
    try:
        modelo = _cargar_yolo(req.weights_path)
        _estado["model"] = modelo
        _estado["active_path"] = req.weights_path
        _estado["names"] = modelo.names if hasattr(modelo, "names") else {}
        return {"ok": True, "classes": _estado["names"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
def predict(req: PredictReq):
    """Detecta sobre la imagen y devuelve cajas en píxeles + recortes."""
    if _estado["model"] is None:
        raise HTTPException(
            status_code=409,
            detail="No hay modelo activo. El administrador debe activar un peso primero.",
        )
    try:
        img = _b64_a_imagen(req.image_base64)
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR para YOLO

        resultados = _estado["model"].predict(source=arr, conf=req.conf, verbose=False)
        r = resultados[0]
        names = _estado["names"]

        detecciones = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
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
