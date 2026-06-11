from __future__ import annotations

import hashlib
from collections import defaultdict

from src.schemas import PaperChunk, PaperPage


def _chunk_id(paper_id: str, page_start: int, idx: int, text: str) -> str:
    raw = f"{paper_id}:{page_start}:{idx}:{text[:80]}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:12]


def chunk_pages(
    pages: list[PaperPage],
    chunk_size: int = 1200,
    overlap: int = 220,
) -> list[PaperChunk]:
    """Create page-aware overlapping chunks.

    The implementation keeps chunks simple and transparent. It is intentionally not
    hidden behind a framework so that portfolio reviewers can understand it easily.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[PaperChunk] = []
    pages_by_paper: dict[str, list[PaperPage]] = defaultdict(list)
    for page in pages:
        pages_by_paper[page.paper_id].append(page)

    for paper_id, paper_pages in pages_by_paper.items():
        paper_pages = sorted(paper_pages, key=lambda p: p.page_number)
        paper_title = paper_pages[0].paper_title
        buffer = ""
        page_start = paper_pages[0].page_number
        chunk_index = 0

        for page in paper_pages:
            if not buffer:
                page_start = page.page_number

            page_text = f"\n\n[Page {page.page_number}] {page.text}"
            buffer += page_text

            while len(buffer) >= chunk_size:
                chunk_text = buffer[:chunk_size].strip()
                page_end = page.page_number
                chunks.append(
                    PaperChunk(
                        chunk_id=_chunk_id(paper_id, page_start, chunk_index, chunk_text),
                        paper_id=paper_id,
                        paper_title=paper_title,
                        page_start=page_start,
                        page_end=page_end,
                        text=chunk_text,
                    )
                )
                chunk_index += 1
                buffer = buffer[chunk_size - overlap :]
                page_start = max(page_start, page.page_number)

        if buffer.strip():
            chunks.append(
                PaperChunk(
                    chunk_id=_chunk_id(paper_id, page_start, chunk_index, buffer),
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page_start=page_start,
                    page_end=paper_pages[-1].page_number,
                    text=buffer.strip(),
                )
            )
    return chunks
