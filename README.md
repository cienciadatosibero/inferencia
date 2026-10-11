# Inferencia solar para Vercel (YOLOv8 ONNX)

## Problema corregido

El repositorio original instalaba `ultralytics`, `torch` y dependencias de CUDA, creando funciones Python de **5.8 GB**, por encima del límite Python de **500 MB**. El código de producción ahora solo necesita `fastapi`, `pillow`, `numpy` y `onnxruntime` (CPU).

## Paso obligatorio: exportar tu peso real a ONNX

**Este ZIP incluye tu `modelos/best.pt` original, pero todavía NO contiene `modelos/best.onnx`.** La conversión es obligatoria, pero incluí **GitHub Actions** para hacerla automáticamente al subir estos cambios a la rama `main`. El workflow usa PyTorch CPU **en GitHub**, nunca en Vercel, y después hace commit de `modelos/best.onnx`.

**Pasos recomendados**: sustituye archivos en tu repo; haz `git add -A`, `git commit` y `git push origin main`; entra a GitHub → Actions → «Exportar modelo YOLO a ONNX» y espera a que termine. Debes ver un nuevo commit que añade `best.onnx`; si Vercel no redespliega automáticamente, selecciona Redeploy. El repositorio debe permitir que GitHub Actions escriba commits sobre `main` (Settings → Actions → General → Workflow permissions). Si la rama está protegida, usa la exportación manual descrita abajo.

Desde la carpeta de tu repositorio `inferencia`, en Windows PowerShell:

```powershell
py -3.12 -m pip install -r requirements-export.txt
py -3.12 scripts/exportar_onnx.py
Test-Path .\modelos\best.onnx
```

Si ya tienes una `.venv` con ultralytics instalada, ejecuta el script con ese Python.

## Actualizar GitHub (GitHub Actions exporta ONNX automáticamente)

Copia los archivos de este ZIP a tu repositorio `inferencia` (mantén su carpeta `.git`). Después:

```powershell
git status
git add -A
git commit -m "Convertir inferencia YOLO a ONNX para Vercel"
git push origin main
```

Una vez que **GitHub Actions añada `modelos/best.onnx`**, entra en **Vercel**: Proyecto **inferencia** → **Deployments** → **Redeploy**, o espera la integración automática de GitHub. **Root Directory** debe apuntar a la raíz del repositorio. Vercel detecta automáticamente el `app.py` de FastAPI sin rewrites. El sistema expone `/health`, `/load`, `/activate`, `/predict` y `/`.

Puedes verificar:

```powershell
Invoke-RestMethod https://TU-PROYECTO-INFERENCIA.vercel.app/health
```

La respuesta debe incluir `ok: true` y `model_file_ready: true`.

## API/Backend Node

En el proyecto `satelital-backend.vercel.app`, configura en su Environment Variables:

```env
INFERENCE_URL=https://TU-PROYECTO-INFERENCIA.vercel.app
```

Deja la URL **sin** `/api`, porque el backend ya invoca `/health`, `/load`, `/activate` y `/predict`.

Si configuras `INFERENCE_API_TOKEN` en inferencia, usa el mismo valor en el backend.

## Compatibilidad y limitaciones

- Modelo original YOLOv8n, una clase: `solar-pv-panel`.
- Inferencia sin estado; `/activate` valida el peso del modelo empaquetado, no guarda la selección entre llamadas.
- `/load` y `/activate` admiten únicamente el nombre de peso incluido, `best.pt` o `best.onnx`. No permiten activar pesos arbitrarios subidos al backend.
- Vercel no comparte el disco local del backend con inferencia. Para modelos nuevos, hace falta almacenamiento de objetos y un mecanismo de despliegue/carga adicional.
- Los cuerpos HTTP de Vercel tienen un límite de 4.5 MB. Este servicio limita imágenes decodificadas a 3 MB y 16 megapíxeles.
- El modelo corre en CPU; el rendimiento real debe probarse tras desplegar.
- No cambies `requirements.txt` para añadir `ultralytics` o `torch`: volvería el error de bundle.

## Pruebas locales

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pip install uvicorn
py -3.12 -m unittest discover -s tests -v
py -3.12 -m uvicorn app:app --host 127.0.0.1 --port 8000
```
