"""Public REST API. HTTPS is terminated by the deployment reverse proxy."""
from __future__ import annotations

import asyncio
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from src.answerer import answer_question
from src.config import CONFIG
from src.metrics import compute_query_metrics
from src.mlflow_logger import log_query_event
from src.pipeline import MAX_PDF_BYTES, prepare_paper
from src.question_suggestions import suggest_questions


def _keys() -> list[str]:
    return [key.strip() for key in os.getenv("PAPERWISE_API_KEYS", "").split(",") if key.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("APP_ENV", "development").lower() == "production" and not _keys():
        raise RuntimeError("PAPERWISE_API_KEYS is required in production")
    if os.getenv("APP_ENV", "development").lower() == "production" and "*" in origins:
        raise RuntimeError("Wildcard CORS origins are not allowed in production")
    yield


app = FastAPI(
    title="PaperWise AI API", version="1.0.0",
    description="Ask citation-grounded questions about a PDF. Offline answers work without OpenAI credentials.",
    root_path=os.getenv("PAPERWISE_ROOT_PATH", ""),
    lifespan=lifespan,
)
origins = [origin.strip() for origin in os.getenv("PAPERWISE_CORS_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"],
)

_requests: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = threading.Lock()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: str | None = Security(_api_key_header)) -> str:
    configured = _keys()
    if configured and not (x_api_key and any(secrets.compare_digest(x_api_key, key) for key in configured)):
        raise HTTPException(status_code=401, detail="A valid X-API-Key is required.")
    identity = x_api_key if configured else "local"
    try:
        limit = max(1, int(os.getenv("PAPERWISE_RATE_LIMIT", "30")))
    except ValueError:
        limit = 30
    now = time.monotonic()
    with _rate_lock:
        bucket = _requests[identity]
        while bucket and bucket[0] <= now - 60:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in one minute.")
        bucket.append(now)
    return identity


async def _read_pdf(file: UploadFile) -> bytes:
    data = await file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB limit.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="Upload a valid PDF file.")
    return data


async def _prepare(file: UploadFile):
    data = await _read_pdf(file)
    try:
        return await asyncio.to_thread(prepare_paper, data, file.filename or "paper.pdf")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not extract text from this PDF.") from exc


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", tags=["research"])
async def ask_paper(
    question: str = Form(..., min_length=2, max_length=2000),
    file: UploadFile = File(...),
    top_k: int = Form(CONFIG.top_k, ge=1, le=10),
    _: str = Depends(require_api_key),
) -> dict[str, object]:
    start = time.perf_counter()
    prepared = await _prepare(file)
    results = await asyncio.to_thread(prepared.index.search, question, top_k)
    answer = await asyncio.to_thread(answer_question, question, results)
    metrics = compute_query_metrics(results, answer, time.perf_counter() - start)
    paper_titles = sorted({p.paper_title for p in prepared.pages})
    # Telemetry is optional and must never affect the response.
    await asyncio.to_thread(
        log_query_event, question, paper_titles,
        {"top_k": top_k, "endpoint": "/ask"}, metrics,
        {"app": "fastapi", "project": "research-paper-ragops"},
    )
    return {
        "answer": answer.answer, "used_llm": answer.used_llm,
        "citations": answer.citations,
        "guardrail_warnings": answer.guardrail_warnings,
        "metrics": metrics,
    }


@app.post("/suggest-questions", tags=["research"])
async def suggest(file: UploadFile = File(...), _: str = Depends(require_api_key)) -> dict[str, object]:
    prepared = await _prepare(file)
    return {"questions": await asyncio.to_thread(suggest_questions, prepared.chunks)}
