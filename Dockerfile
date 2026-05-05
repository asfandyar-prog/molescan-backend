# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Use CUDA-enabled base for GPU inference on the faculty NVIDIA server.
# cuda 12.1 + cudnn8 matches PyTorch 2.3.0 wheels.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

# Suppress interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /build

# Install uv
RUN pip install --no-cache-dir uv

COPY pyproject.toml .

# Install production dependencies only (no dev extras on server)
RUN uv pip install --system --no-cache \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    python-multipart \
    pillow \
    "torch>=2.3.0" \
    torchvision \
    "transformers>=4.41.0" \
    timm \
    numpy \
    python-dotenv


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

LABEL maintainer="Asfand Yar"
LABEL maintainer="Asfand Yar <asfandyar@mailbox.unideb.hu>"
LABEL description="MoleScan backend — CITDS 2026"
LABEL supervisor="Prof. Balázs Harangi, University of Debrecen"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11 2>/dev/null || true
COPY --from=builder /usr/local/bin /usr/local/bin 2>/dev/null || true
COPY --from=builder /usr/lib/python3 /usr/lib/python3 2>/dev/null || true

# Copy application source
COPY app/ ./app/

# Weights directory — populated via volume mount at runtime
# The .pt file is NOT baked into the image (too large, needs to be updated)
RUN mkdir -p weights

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check so Docker knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]