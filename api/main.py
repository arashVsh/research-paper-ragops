from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from src.answerer import answer_question
from src.chunking import chunk_pages
from src.config import CONFIG
from src.metrics import compute_query_metrics
from src.mlflow_logger import log_query_event
from src.pdf_loader import load_pdf_from_bytes
from src.question_suggestions import suggest_questions
from src.retriever import RetrievalIndex

app = FastAPI(title="Research Paper RAGOps API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
async def ask_paper(
    question: str = Form(...),
    file: UploadFile = File(...),
    top_k: int = Form(CONFIG.top_k),
) -> dict[str, object]:
    pdf_bytes = await file.read()
    pages = load_pdf_from_bytes(pdf_bytes, file.filename or "paper.pdf")
    chunks = chunk_pages(pages, chunk_size=CONFIG.chunk_size, overlap=CONFIG.chunk_overlap)
    index = RetrievalIndex.build(chunks)

    start = time.perf_counter()
    results = index.search(question, top_k=top_k)
    answer = answer_question(question, results)
    latency = time.perf_counter() - start
    metrics = compute_query_metrics(results, answer, latency)
    paper_titles = sorted({p.paper_title for p in pages})

    log_query_event(
        question=question,
        paper_titles=paper_titles,
        params={"top_k": top_k, "endpoint": "/ask"},
        metrics=metrics,
        tags={"app": "fastapi", "project": "research-paper-ragops"},
    )

    return {
        "answer": answer.answer,
        "used_llm": answer.used_llm,
        "citations": answer.citations,
        "guardrail_warnings": answer.guardrail_warnings,
        "metrics": metrics,
    }


@app.post("/suggest-questions")
async def suggest(file: UploadFile = File(...)) -> dict[str, object]:
    pdf_bytes = await file.read()
    pages = load_pdf_from_bytes(pdf_bytes, file.filename or "paper.pdf")
    chunks = chunk_pages(pages, chunk_size=CONFIG.chunk_size, overlap=CONFIG.chunk_overlap)
    return {"questions": suggest_questions(chunks)}
