# Guard-main integration contract

This module exposes HTTP API compatible with `guard-main/lib/analysis-client.ts`.

## Endpoint

- `POST /v1/analyze`
- `Content-Type: application/json`
- Optional header: `X-API-Key` (required only if `ANALYSIS_API_KEY` is set)

## Request body

```json
{
  "content": "text to analyze",
  "filename": "document.txt",
  "document_id": 123
}
```

- `content` is required, non-empty string
- `filename` optional, defaults to `document.txt`
- `document_id` optional, used for tracing only

## Success response (200)

```json
{
  "plagiarism_percent": 12.34,
  "ai_percent": 7.89
}
```

## Error responses

- `401` invalid API key
- `422` invalid request schema
- `503` worker not initialized
- `500` internal worker/runtime error

## Environment variables

- `QDRANT_HOST` (default: `localhost`)
- `QDRANT_PORT` (default: `6333`)
- `QDRANT_COLLECTION` (default: `university_docs`)
- `ANALYSIS_API_KEY` (optional; must match `ANALYSIS_SERVICE_API_KEY` on guard-main)
- `DEVICE` (default: `auto`) — `auto` | `cpu` | `cuda` (use `cuda` only with GPU-enabled PyTorch + GPU access)
- `TORCH_CPU_THREADS` (optional) — OpenMP/torch CPU threads on CPU runs (default: up to 8 or available cores)

## Guard-main pairing

- ML base URL on web: **`ANALYSIS_SERVICE_URL`** (e.g. `http://172.16.251.219:8765`, no trailing slash).
- Shared secret header: **`ANALYSIS_SERVICE_API_KEY`** on Next.js MUST equal **`ANALYSIS_API_KEY`** on this service.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8765
```
docker compose up -d --build analysis