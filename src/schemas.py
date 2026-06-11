from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PaperPage:
    paper_id: str
    paper_title: str
    page_number: int
    text: str


@dataclass
class PaperChunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    page_start: int
    page_end: int
    text: str


@dataclass
class RetrievedChunk:
    chunk: PaperChunk
    score: float
    rank: int


@dataclass
class AnswerResult:
    answer: str
    used_llm: bool
    citations: list[str]
    guardrail_warnings: list[str]
