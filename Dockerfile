# syntax=docker/dockerfile:1.4
# ML analysis API — layers ordered for cache: deps → models → app code.
# First build ~25–35 min (PyTorch + models). Rebuild after code-only changes: ~1–2 min.

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/models/huggingface \
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
# Модели задаются build-аргами и фиксируются в ENV, чтобы рантайм по умолчанию
# использовал именно запечённые модели (а не качал другие при старте).
ARG EMBEDDING_MODEL=BAAI/bge-m3
ARG AI_MODEL=sberbank-ai/rugpt3small_based_on_gpt2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    AI_MODEL=${AI_MODEL}
COPY preload_models.py ./
RUN python preload_models.py

# Layer 3 — application code (changes often; cheap rebuild)
COPY api_server.py worker.py cascade.py university_queues.py ./

EXPOSE 8765

# Проверяем не только доступность эндпоинта, но и флаг ok: иначе контейнер
# с навсегда упавшим ML-worker (health отвечает 200 с ok:false) вечно "healthy".
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import json, sys, urllib.request; d = json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=5)); sys.exit(0 if d.get('ok') else 1)"

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8765"]
