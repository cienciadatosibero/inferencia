"""YOLO de uso único. El .pt se borra del disco inmediatamente después de cargarlo.
El modelo se libera de RAM al terminar el escaneo o al expirar 40 minutos.
Ejecutar en un servicio Docker persistente con un solo worker, NUNCA en Vercel.
ADVERTENCIA: .pt de Ultralytics usa serialización PyTorch. Admitir solamente admins y modelos confiables.
"""
import base64
import binascii
import hashlib
import hmac
import io
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MAX_MODEL_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MODEL_TTL = 40 * 60  # borrado automático si el escaneo se interrumpe
MODEL_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)
TOKEN_RE = re.compile(r'^v1\.(\d{10})\.([\da-f-]{36})\.(\d+)\.([\da-f]{64})$', re.I)
Image.MAX_IMAGE_PIXELS = 22_000_000

app = FastAPI(title='SatDataset Temporary YOLO', version='1.0.0')
origins = [x.strip() for x in os.getenv('CORS_ORIGIN', 'https://satelital.vercel.app').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['POST', 'OPTIONS'], allow_headers=['Authorization', 'Content-Type'], max_age=3600)

_LOCK = threading.RLock()
MODELS = {}  # id -> { model, names, expires_at }. Nunca se almacena el .pt original.
USED_TICKETS = set()

class PredictReq(BaseModel):
    temp_model_id: str
    image_base64: str = Field(min_length=1)
    conf: float = Field(default=0.20, ge=0.05, le=0.95)
    return_crops: bool = False

class ReleaseReq(BaseModel):
    temp_model_id: str


def _service_token(header):
    expected = os.getenv('INFERENCE_API_TOKEN', '')
    if not expected:
        raise HTTPException(503, 'Falta INFERENCE_API_TOKEN en el servicio Python')
    if not hmac.compare_digest(header or '', f'Bearer {expected}'):
        raise HTTPException(401, 'Autorización inválida')


def _upload_token(header):
    key = os.getenv('TEMP_MODEL_SIGNING_KEY', '')
    if len(key) < 24:
        raise HTTPException(503, 'Falta TEMP_MODEL_SIGNING_KEY en el servicio Python')
    token = (header or '').removeprefix('Bearer ').strip()
    match = TOKEN_RE.fullmatch(token)
    if not match:
        raise HTTPException(401, 'Permiso de subida inválido')
    expiry, nonce, uid, signature = match.groups()
    value = f'v1.{expiry}.{nonce}.{uid}'
    expected = hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected) or int(expiry) < int(time.time()) or int(expiry) > int(time.time()) + 620:
        raise HTTPException(401, 'Permiso de subida expirado o alterado')
    with _LOCK:
        if nonce in USED_TICKETS:
            raise HTTPException(409, 'Este permiso de subida ya fue utilizado')
        USED_TICKETS.add(nonce)


def _clear_expired():
    now = time.time()
    with _LOCK:
        for ident, entry in list(MODELS.items()):
            if entry['expires_at'] < now:
                MODELS.pop(ident, None)


def _janitor():
    while True:
        time.sleep(60)
        _clear_expired()

@app.on_event('startup')
def start_cleanup():
    threading.Thread(target=_janitor, daemon=True).start()


def _decode_image(payload):
    raw = payload.split(',', 1)[-1]
    if len(raw) > MAX_IMAGE_BYTES * 4 // 3 + 8192:
        raise HTTPException(413, 'Imagen demasiado grande')
    try:
        b = base64.b64decode(raw, validate=True)
        if len(b) > MAX_IMAGE_BYTES:
            raise HTTPException(413, 'Imagen demasiado grande')
        img = Image.open(io.BytesIO(b))
        img.load()
        return img.convert('RGB')
    except HTTPException:
        raise
    except (ValueError, OSError, UnidentifiedImageError, binascii.Error):
        raise HTTPException(400, 'Imagen inválida')


@app.get('/health')
def health():
    _clear_expired()
    return {'ok': True, 'mode': 'temporary-yolo', 'modelos_temporales': len(MODELS)}


@app.post('/temporary/upload')
async def upload_model(peso: UploadFile = File(...), authorization: str | None = Header(default=None)):
    _upload_token(authorization)
    if not (peso.filename or '').lower().endswith('.pt'):
        raise HTTPException(400, 'Se requiere un .pt YOLO de confianza')
    _clear_expired()
    fd, filename = tempfile.mkstemp(prefix='sat_oneoff_', suffix='.pt')
    size = 0
    try:
        with os.fdopen(fd, 'wb') as out:
            while chunk := await peso.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MODEL_BYTES:
                    raise HTTPException(413, 'El .pt supera los 200 MB')
                out.write(chunk)
        if size < 1024:
            raise HTTPException(400, 'Modelo vacío o demasiado pequeño')
        try:
            model = YOLO(filename, task='detect')
            names = {int(k): str(v) for k, v in model.names.items()}
            if not names:
                raise ValueError('No hay clases en este peso')
        except Exception as exc:
            raise HTTPException(422, f'No se pudo cargar este YOLO .pt: {exc}') from exc
        model_id = str(uuid.uuid4())
        with _LOCK:
            MODELS[model_id] = {'model': model, 'names': names, 'expires_at': time.time() + MODEL_TTL}
        return {'ok': True, 'temp_model_id': model_id, 'classes': {str(k): v for k, v in names.items()}, 'expires_in': MODEL_TTL}
    finally:
        Path(filename).unlink(missing_ok=True)  # borrar .pt aun si falla
        await peso.close()


@app.post('/temporary/predict')
def predict(req: PredictReq, authorization: str | None = Header(default=None)):
    _service_token(authorization)
    if not MODEL_RE.fullmatch(req.temp_model_id):
        raise HTTPException(400, 'ID temporal inválido')
    _clear_expired()
    image = _decode_image(req.image_base64)
    with _LOCK:
        entry = MODELS.get(req.temp_model_id)
        if not entry:
            raise HTTPException(410, 'Modelo temporal no disponible o expirado. Súbelo otra vez.')
        try:
            result = entry['model'].predict(source=image, conf=req.conf, imgsz=640, device='cpu', verbose=False)[0]
        except Exception as exc:
            raise HTTPException(500, f'Falló la inferencia: {exc}') from exc
        dets = []
        if result.boxes is not None:
            for b in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(image.width, int(x2)), min(image.height, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                idx = int(b.cls[0].item())
                d = {'class_id': idx, 'class_name': entry['names'].get(idx, str(idx)),
                     'confidence': float(b.conf[0].item()),
                     'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
                if req.return_crops:
                    out = io.BytesIO()
                    image.crop((x1, y1, x2, y2)).save(out, 'JPEG', quality=82)
                    d['crop_base64'] = base64.b64encode(out.getvalue()).decode('ascii')
                dets.append(d)
    return {'ok': True, 'image_width': image.width, 'image_height': image.height, 'detections': dets}


@app.post('/temporary/release')
def release(req: ReleaseReq, authorization: str | None = Header(default=None)):
    _service_token(authorization)
    if not MODEL_RE.fullmatch(req.temp_model_id):
        raise HTTPException(400, 'ID temporal inválido')
    with _LOCK:
        MODELS.pop(req.temp_model_id, None)  # liberar modelo cargado en RAM
    return {'ok': True, 'deleted': True}
