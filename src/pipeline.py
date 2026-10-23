"""Shared, bounded ingestion pipeline for the UI and API."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.chunking import chunk_pages
from src.config import CONFIG
from src.pdf_loader import load_pdf_from_bytes
from src.retriever import RetrievalIndex
from src.schemas import PaperPage, PaperChunk


MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 250


@dataclass
class PreparedPaper:
    pages: list[PaperPage]
    chunks: list[PaperChunk]
    index: RetrievalIndex


@lru_cache(maxsize=4)
def prepare_paper(pdf_bytes: bytes, filename: str, chunk_size: int = CONFIG.chunk_size,
                  overlap: int = CONFIG.chunk_overlap) -> PreparedPaper:
    if not pdf_bytes or len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF must be between 1 byte and 20 MB.")
    pages = load_pdf_from_bytes(pdf_bytes, filename, max_pages=MAX_PAGES)
    if not pages:
        raise ValueError("No searchable text found. Scanned PDFs require OCR.")
    if len(pages) > MAX_PAGES:
        raise ValueError("PDF exceeds the 250 readable page limit.")
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    return PreparedPaper(pages, chunks, RetrievalIndex.build(chunks))
