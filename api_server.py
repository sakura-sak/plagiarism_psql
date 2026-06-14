import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from worker import AntiPlagiarismWorker

logging.basicConfig(level=logging.INFO)

_worker: Optional[AntiPlagiarismWorker] = None
_worker_error: Optional[str] = None
_worker_loading = False


class AnalyzeRequest(BaseModel):
    content: str = Field(..., min_length=1)
    filename: str = Field(default="document.txt", max_length=512)
    document_id: Optional[int] = None


class AnalyzeResponse(BaseModel):
    plagiarism_percent: float
    ai_percent: float


def _verify_api_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv("ANALYSIS_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _init_worker() -> None:
    global _worker, _worker_error, _worker_loading
    _worker_loading = True
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "university_docs")
        logging.info("Loading ML models (first start may take several minutes on CPU)...")
        _worker = await run_in_threadpool(
            lambda: AntiPlagiarismWorker(
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                collection_name=qdrant_collection,
            )
        )
        logging.info("ML worker is ready")
    except Exception as exc:
        _worker_error = str(exc)
        logging.exception("Failed to initialize ML worker")
    finally:
        _worker_loading = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_init_worker())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Antiplagiarism Worker PGSQL API",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if _cors_origins_raw:
    origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
async def health():
    if _worker is not None:
        return {"ok": True, "ready": True}
    if _worker_error:
        return {"ok": False, "ready": False, "error": _worker_error}
    return {"ok": True, "ready": False, "loading": _worker_loading}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _verify_api_key(x_api_key)
    if _worker is None:
        if _worker_error:
            raise HTTPException(status_code=503, detail=f"Worker failed to start: {_worker_error}")
        raise HTTPException(
            status_code=503,
            detail="Worker is still loading models. Wait and retry (first start can take 5–15 minutes on CPU).",
        )

    result = await run_in_threadpool(
        _worker.process_text, body.content, body.filename, False
    )
    return AnalyzeResponse(
        plagiarism_percent=float(result.get("plagiarism_percent", 0.0)),
        ai_percent=float(result.get("ai_percent", 0.0)),
    )
