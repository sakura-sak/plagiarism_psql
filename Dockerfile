# syntax=docker/dockerfile:1.4
# ML analysis API — layers ordered for cache: deps → models → app code.
# First build ~25–35 min (PyTorch + models). Rebuild after code-only changes: ~1–2 min.

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/models/huggingface \
    TRANSFORMERS_CACHE=/opt/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers \
    TORCH_HOME=/opt/models/torch \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION=university_docs

# Layer 1 — Python deps (invalidates only when requirements.txt changes)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Layer 2 — HuggingFace models baked into image (fast container start, cached layer)
COPY preload_models.py ./
RUN python preload_models.py

# Layer 3 — application code (changes often; cheap rebuild)
COPY api_server.py worker.py ./

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=5)"

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8765"]
