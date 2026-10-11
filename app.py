"""SatDataset: inferencia de paneles solares para Vercel con ONNX Runtime CPU.

El modelo original es un YOLOv8n de una clase: ``solar-pv-panel``.
No usa PyTorch ni Ultralytics durante la ejecución en Vercel.

Compatibilidad API: GET /health, POST /load, POST /activate, POST /predict.
Las funciones serverless son independientes: /predict siempre usa el modelo
ONNX empaquetado, sin depender de que /activate haya ejecutado antes.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import io
import os
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "modelos" / "best.onnx"
MODEL_CLASSES = {0: "solar-pv-panel"}
MODEL_SIZE = 640  # mismo tamaño que el exportador ONNX
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # Vercel limita el payload HTTP; base64 suma ~33%
MAX_IMAGE_PIXELS = 16_000_000
MAX_DETECTIONS = 300
INFERENCE_TOKEN = os.getenv("INFERENCE_API_TOKEN", "").strip()

app = FastAPI(title="SatDataset Inference", version="2.0.0")
_session = None
_session_lock = Lock()


class LoadReq(BaseModel):
    weights_path: str = Field(min_length=1)


class PredictReq(BaseModel):
    image_base64: str = Field(min_length=1)
    conf: float = Field(default=0.20, ge=0.05, le=0.95)
    return_crops: bool = True


def check_token(authorization: str | None):
    if INFERENCE_TOKEN and not hmac.compare_digest(authorization or "", f"Bearer {INFERENCE_TOKEN}"):
        raise HTTPException(status_code=401, detail="Token de inferencia inválido")


def check_model_request(weights_path: str):
    # El backend conoce el modelo como best.pt. Aquí usamos su exportación ONNX.
    # Rechazamos otros pesos en vez de fingir que activamos un modelo diferente.
    if Path(weights_path.replace("\\", "/")).name not in {"best.pt", "best.onnx"}:
        raise HTTPException(
            status_code=409,
            detail="Vercel solo incluye best.onnx. Para modelos adicionales hace falta almacenarlos y publicarlos externamente.",
        )
    if not ONNX_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Falta modelos/best.onnx. Exporta modelos/best.pt a ONNX y súbelo al repositorio antes de desplegar.",
        )


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                if not ONNX_PATH.is_file():
                    raise HTTPException(
                        status_code=503,
                        detail="No se encontró modelos/best.onnx. Ejecuta scripts/exportar_onnx.py localmente.",
                    )
                try:
                    import onnxruntime as ort
                    _session = ort.InferenceSession(
                        str(ONNX_PATH),
                        sess_options=ort.SessionOptions(),
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as exc:
                    raise HTTPException(status_code=503, detail=f"No se pudo iniciar ONNX Runtime: {exc}") from exc
    return _session


def decode_image(payload: str) -> Image.Image:
    encoded = payload.partition(",")[2] if "," in payload else payload
    try:
        if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 16):
            raise ValueError("La imagen excede 3 MB comprimidos")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("La imagen excede 3 MB comprimidos")
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("Máximo 16 megapíxeles por imagen")
            opened.load()
            return opened.convert("RGB")
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {exc}") from exc


def letterbox(image: Image.Image, size: int = MODEL_SIZE):
    """Escala sin deformar y rellena con 114, como Ultralytics.

    Se usa la misma geometría para deshacer el letterbox al entregar las cajas.
    """
    w, h = image.size
    gain = min(size / w, size / h)
    new_w = max(1, min(size, round(w * gain)))
    new_h = max(1, min(size, round(h * gain)))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    tensor = np.ascontiguousarray(np.array(canvas).transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, gain, left, top


def iou_single(box, boxes):
    """IoU con cajas en formato xyxy."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, threshold=0.45, max_detections=MAX_DETECTIONS):
    order = np.argsort(scores)[::-1][:1500]
    keep = []
    while len(order) and len(keep) < max_detections:
        selected = int(order[0])
        keep.append(selected)
        if len(order) == 1:
            break
        remaining = order[1:]
        overlap = iou_single(boxes[selected], boxes[remaining])
        order = remaining[overlap < threshold]
    return keep


def decode_predictions(output, image_size, gain, pad_x, pad_y, conf):
    """Salida YOLOv8: [1, 4+clases, N] o [1, N, 4+clases]."""
    arr = np.asarray(output)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Forma ONNX no esperada: {arr.shape}")
    raw = arr[0]
    num_channels = 4 + len(MODEL_CLASSES)
    if raw.shape[0] == num_channels:
        raw = raw.T
    elif raw.shape[1] != num_channels:
        raise ValueError(f"Esperaba {num_channels} canales YOLOv8, encontré {arr.shape}")

    # YOLOv8 ya produce puntuaciones de clase; no lleva objectness separada.
    classes = raw[:, 4:]
    class_ids = np.argmax(classes, axis=1)
    scores = classes[np.arange(len(raw)), class_ids]
    valid = scores >= conf
    raw, scores, class_ids = raw[valid], scores[valid], class_ids[valid]
    if raw.shape[0] == 0:
        return []

    xywh = raw[:, :4]
    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    # Mismo score para deduplicar; múltiples clases conservan NMS por clase.
    selected = []
    for class_id in np.unique(class_ids):
        ids = np.flatnonzero(class_ids == class_id)
        selected.extend(ids[idx] for idx in nms(boxes[ids], scores[ids]))
    selected = sorted(selected, key=lambda idx: -scores[idx])[:MAX_DETECTIONS]

    w, h = image_size
    detections = []
    for idx in selected:
        x1, y1, x2, y2 = boxes[idx]
        x1 = float(np.clip((x1 - pad_x) / gain, 0, w))
        x2 = float(np.clip((x2 - pad_x) / gain, 0, w))
        y1 = float(np.clip((y1 - pad_y) / gain, 0, h))
        y2 = float(np.clip((y2 - pad_y) / gain, 0, h))
        if x2 <= x1 or y2 <= y1:
            continue
        cls = int(class_ids[idx])
        detections.append({
            "class_id": cls,
            "class_name": MODEL_CLASSES.get(cls, str(cls)),
            "confidence": float(scores[idx]),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
        })
    return detections


def jpeg_crop(image: Image.Image, det):
    x1, y1 = int(det["x"]), int(det["y"])
    x2 = min(image.width, int(np.ceil(det["x"] + det["w"])))
    y2 = min(image.height, int(np.ceil(det["y"] + det["h"])))
    crop = image.crop((x1, y1, x2, y2))
    output = io.BytesIO()
    crop.save(output, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@app.get("/")
def index():
    return {"ok": True, "service": "SatDataset ONNX Inference", "health": "/health"}


@app.get("/health")
def health():
    ready = ONNX_PATH.is_file()
    return {
        "ok": ready,
        "model_loaded": _session is not None,
        "model_file_ready": ready,
        "active_path": None,  # serverless: no garantiza estado entre peticiones
        "classes": MODEL_CLASSES,
        "framework": "onnxruntime-yolov8",
    }


@app.post("/load")
def load(req: LoadReq, authorization: str | None = Header(default=None)):
    check_token(authorization)
    check_model_request(req.weights_path)
    return {"ok": True, "framework": "onnxruntime-yolov8", "classes": MODEL_CLASSES}


@app.post("/activate")
def activate(req: LoadReq, authorization: str | None = Header(default=None)):
    check_token(authorization)
    check_model_request(req.weights_path)
    # No almacenamos la elección en RAM; el único modelo empaquetado es best.onnx.
    return {"ok": True, "framework": "onnxruntime-yolov8", "classes": MODEL_CLASSES}


@app.post("/predict")
def predict(req: PredictReq, authorization: str | None = Header(default=None)):
    check_token(authorization)
    image = decode_image(req.image_base64)
    tensor, gain, left, top = letterbox(image)
    session = _get_session()
    input_name = session.get_inputs()[0].name
    try:
        out = session.run(None, {input_name: tensor})
        detections = decode_predictions(out[0], image.size, gain, left, top, req.conf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar ONNX: {exc}") from exc
    if req.return_crops:
        for det in detections:
            det["crop_base64"] = jpeg_crop(image, det)
    return {
        "ok": True,
        "image_width": image.width,
        "image_height": image.height,
        "detections": detections,
    }
