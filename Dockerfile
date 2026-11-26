FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    TORCH_HOME=/data/torch \
    EASYOCR_MODULE_PATH=/data/easyocr \
    ADAPTIFAI_TMP_DIR=/data/tmp/adaptifai

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libfribidi0 \
        libharfbuzz0b \
        fonts-noto-core \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY backend/app ./app
COPY backend/scripts ./scripts

RUN mkdir -p /data/huggingface /data/torch /data/easyocr /data/tmp/adaptifai

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
