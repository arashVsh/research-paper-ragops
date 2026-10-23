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
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[PaperChunk] = []
    pages_by_paper: dict[str, list[PaperPage]] = defaultdict(list)
    for page in pages:
        pages_by_paper[page.paper_id].append(page)

    for paper_id, paper_pages in pages_by_paper.items():
        paper_pages = sorted(paper_pages, key=lambda p: p.page_number)
        paper_title = paper_pages[0].paper_title
        segments: list[tuple[int, int, int]] = []
        parts: list[str] = []
        offset = 0
        for page in paper_pages:
            part = f"\n\n[Page {page.page_number}] {page.text}"
            parts.append(part)
            segments.append((offset, offset + len(part), page.page_number))
            offset += len(part)
        full_text = "".join(parts)
        start = 0
        chunk_index = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            if end < len(full_text):
                boundary = full_text.rfind(" ", start + int(chunk_size * 0.75), end)
                if boundary > start:
                    end = boundary
            chunk_text = full_text[start:end].strip()
            covered = [number for first, last, number in segments if first < end and last > start]
            if not chunk_text:
                break
            chunks.append(
                PaperChunk(
                    chunk_id=_chunk_id(paper_id, covered[0], chunk_index, chunk_text),
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page_start=covered[0],
                    page_end=covered[-1],
                    text=chunk_text,
                )
            )
            chunk_index += 1
            if end == len(full_text):
                break
            start = max(start + 1, end - overlap)
    return chunks
