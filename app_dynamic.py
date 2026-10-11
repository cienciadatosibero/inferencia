"""Inferencia YOLO persistente. Modelos por clave R2, seleccionados en cada /predict.

NO publicar este servicio como función Vercel: PyTorch necesita contenedor persistente.
"""
import base64
import binascii
import hashlib
import hmac
import io
import os
import re
import threading
from pathlib import Path

import boto3
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.getenv('MODEL_CACHE_DIR', '/tmp/satdataset-models'))
CACHE.mkdir(parents=True, exist_ok=True)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 22_000_000
LOCK = threading.RLock()
MODELS = {}  # Se mantienen los dos modelos usados más recientemente.
MODEL_KEY_RE = re.compile(r'^models/[a-f0-9]{32}\.pt$')

app = FastAPI(title='SatDataset Dynamic YOLO', version='3.0')


class LoadReq(BaseModel):
    model_key: str = Field(min_length=1, max_length=255)


class PredictReq(BaseModel):
    model_key: str = Field(min_length=1, max_length=255)
    image_base64: str = Field(min_length=1)
    conf: float = Field(default=0.20, ge=0.05, le=0.95)
    return_crops: bool = False


def check_token(authorization):
    token = os.getenv('INFERENCE_API_TOKEN', '').strip()
    if token and not hmac.compare_digest(authorization or '', 'Bearer ' + token):
        raise HTTPException(status_code=401, detail='Token de inferencia incorrecto')


def storage_client():
    names = ('R2_ACCOUNT_ID', 'R2_BUCKET', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY')
    if any(not os.getenv(name) for name in names):
        raise HTTPException(status_code=503, detail='Falta configurar conexión R2 en el servicio Python')
    return boto3.client(
        's3',
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
        region_name='auto',
    )


def get_model_file(key):
    if key == 'best.pt':
        bundled = ROOT / 'modelos' / 'best.pt'
        if not bundled.is_file():
            raise HTTPException(status_code=404, detail='El modelo original best.pt no está incluido en el contenedor')
        return bundled
    if not MODEL_KEY_RE.fullmatch(key):
        raise HTTPException(status_code=400, detail='Identificador de modelo inválido')
    target = CACHE / (hashlib.sha256(key.encode()).hexdigest()[:40] + '.pt')
    if target.exists() and target.stat().st_size > 1024:
        return target
    temp = target.with_suffix('.downloading')
    try:
        client = storage_client()
        client.download_file(os.environ['R2_BUCKET'], key, str(temp))
        if temp.stat().st_size < 1024:
            raise ValueError('El modelo descargado está vacío')
        os.replace(temp, target)
    except Exception as exc:
        temp.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=502, detail=f'No se pudo descargar modelo de R2: {exc}') from exc
    return target


def load_model(key):
    with LOCK:
        if key in MODELS:
            return MODELS[key]
        path = get_model_file(key)
        try:
            model = YOLO(str(path), task='detect')
            # No usar modelos arbitrarios sin revisión: .pt puede contener código pickle.
            names = {int(k): str(v) for k, v in model.names.items()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f'Peso YOLO incompatible o dañado: {exc}') from exc
        if not names:
            raise HTTPException(status_code=422, detail='El modelo no define clases')
        if len(MODELS) >= 2:
            MODELS.pop(next(iter(MODELS)))
        MODELS[key] = (model, names)
        return MODELS[key]


def decode_image(base64_image):
    data = base64_image.split(',', 1)[-1]
    if len(data) > MAX_IMAGE_BYTES * 4 // 3 + 8192:
        raise HTTPException(status_code=413, detail='Imagen demasiado grande')
    try:
        raw = base64.b64decode(data, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail='Imagen demasiado grande')
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert('RGB')
    except HTTPException:
        raise
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail='Imagen inválida') from exc


def class_map(names):
    return {str(k): v for k, v in names.items()}


@app.get('/health')
def health():
    return {'ok': True, 'mode': 'ultralytics-r2', 'loaded_models': list(MODELS.keys())}


@app.post('/load')
def load(req: LoadReq, authorization: str | None = Header(default=None)):
    check_token(authorization)
    _, names = load_model(req.model_key)
    return {'ok': True, 'classes': class_map(names), 'framework': 'ultralytics-yolo'}


@app.post('/activate')
def activate(req: LoadReq, authorization: str | None = Header(default=None)):
    check_token(authorization)
    _, names = load_model(req.model_key)
    return {'ok': True, 'classes': class_map(names), 'framework': 'ultralytics-yolo', 'model_key': req.model_key}


@app.post('/predict')
def predict(req: PredictReq, authorization: str | None = Header(default=None)):
    check_token(authorization)
    image = decode_image(req.image_base64)
    with LOCK:
        model, names = load_model(req.model_key)
        try:
            result = model.predict(source=image, conf=req.conf, imgsz=640, device='cpu', verbose=False)[0]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'Fallo al ejecutar YOLO: {exc}') from exc

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                idx = int(box.cls[0].item())
                det = {'class_id': idx, 'class_name': names.get(idx, str(idx)),
                       'confidence': float(box.conf[0].item()),
                       'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
                if req.return_crops:
                    out = io.BytesIO()
                    image.crop((x1, y1, x2, y2)).save(out, format='JPEG', quality=83)
                    det['crop_base64'] = base64.b64encode(out.getvalue()).decode('ascii')
                detections.append(det)
    return {'ok': True, 'model_key': req.model_key,
            'image_width': image.width, 'image_height': image.height, 'detections': detections}
