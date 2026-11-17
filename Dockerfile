# MediVision AI backend — CPU-only, weights baked in.
# Works locally (docker compose) and on Hugging Face Spaces (Docker SDK, free CPU, port 7860).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/backend PORT=7860 \
    STORAGE_DIR=/tmp/medivision/storage DATABASE_URL=sqlite:////tmp/medivision/medivision.db \
    CORS_ORIGINS=http://localhost:3000

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU wheels keep the image small (no CUDA)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY ml ./ml
COPY backend ./backend
COPY artifacts ./artifacts
COPY sample_data ./sample_data

# HF Spaces runs as a non-root user with uid 1000
RUN useradd -m -u 1000 app && mkdir -p /tmp/medivision && chown -R app /app /tmp/medivision
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/api/system/health')"
CMD ["sh", "-c", "python -m app.seed >/dev/null 2>&1 || true; uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --app-dir backend"]
