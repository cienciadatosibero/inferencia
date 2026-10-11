"""Una sola conversión en Windows o Linux, ANTES de publicar Vercel.

pip install -r requirements-export.txt
python scripts/exportar_onnx.py
"""
from pathlib import Path
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
original = root / 'modelos' / 'best.pt'
if not original.is_file():
    raise SystemExit(f'No encuentro el modelo original: {original}')

model = YOLO(str(original))
print('Clases del modelo:', model.names)
output = model.export(format='onnx', imgsz=640, opset=12, dynamic=False, simplify=False, device='cpu')
path = Path(output)
if path.resolve() != (root / 'modelos' / 'best.onnx').resolve():
    import shutil
    shutil.copy2(path, root / 'modelos' / 'best.onnx')
path = root / 'modelos' / 'best.onnx'
assert path.is_file() and path.stat().st_size > 1_000_000, 'La exportación ONNX no se generó correctamente'
print(f'ONNX generado: {path} ({path.stat().st_size / 1024 / 1024:.2f} MiB)')
print('Ahora ejecuta: git add modelos/best.onnx && git commit -m "Modelo ONNX para Vercel" && git push origin main')
