# Serving image (CPU). For GPU training use the `train` service in docker-compose.yml.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only torch keeps the serving image small; swap the index URL for CUDA wheels when training in Docker.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[serve]"

COPY configs ./configs

ENV CROPIQ_CONFIG=configs/default.yaml
EXPOSE 8000 8501
CMD ["uvicorn", "cropiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
