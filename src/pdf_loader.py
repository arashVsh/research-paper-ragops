from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from src.schemas import PaperPage


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _paper_id_from_bytes(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:12]


def _safe_title(name: str) -> str:
    title = Path(name).stem.replace("_", " ").replace("-", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title or "Uploaded paper"


def load_pdf_from_bytes(pdf_bytes: bytes, filename: str, max_pages: int | None = None) -> list[PaperPage]:
    """Extract page-aware text from a PDF byte stream."""
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a PDF.")
    paper_id = _paper_id_from_bytes(pdf_bytes)
    paper_title = _safe_title(filename)
    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    if max_pages is not None and len(reader.pages) > max_pages:
        raise ValueError(f"PDF exceeds the {max_pages} page limit.")

    pages: list[PaperPage] = []
    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = _clean_text(raw_text)
        if text:
            pages.append(
                PaperPage(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page_number=i,
                    text=text,
                )
            )
    return pages


def load_pdf_file(path: str | Path) -> list[PaperPage]:
    path = Path(path)
    return load_pdf_from_bytes(path.read_bytes(), path.name)
