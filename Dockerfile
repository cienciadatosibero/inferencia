FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
COPY requirements-temporary.txt .
RUN pip install --no-cache-dir -r requirements-temporary.txt
COPY app_temporary.py .
RUN useradd --system --create-home appuser
USER appuser
CMD sh -c 'uvicorn app_temporary:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1'
