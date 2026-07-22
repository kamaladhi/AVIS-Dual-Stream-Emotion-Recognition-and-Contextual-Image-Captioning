# AVIS — Docker Image
# CUDA 12.1 + Python 3.11

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── System dependencies ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application ───────────────────────────────────────────────────────────
COPY src/ ./src/
COPY agents/ ./agents/
COPY demo/ ./demo/
COPY .env.example .env

# ── Expose ports ───────────────────────────────────────────────────────────────
EXPOSE 8000 7860

# ── Default: start FastAPI ─────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
