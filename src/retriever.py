from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import PaperChunk, RetrievedChunk


def _looks_like_references_or_bibliography(text: str) -> bool:
    """Avoid returning bibliography chunks as answers.

    Research PDFs often have references containing many useful keywords, but those
    chunks are usually not useful for explaining the paper's own method.
    """
    lower = text.lower()
    reference_terms = [
        "references",
        "bibliography",
        "proceedings of",
        "arxiv preprint",
        "conference on",
        "in proceedings",
        "doi:",
        "isbn",
    ]
    return sum(term in lower for term in reference_terms) >= 2


@dataclass
class RetrievalIndex:
    chunks: list[PaperChunk]
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def build(cls, chunks: list[PaperChunk]) -> "RetrievalIndex":
        if not chunks:
            raise ValueError("Cannot build retrieval index with zero chunks")
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        return cls(chunks=chunks, vectorizer=vectorizer, matrix=matrix)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        # Pull a larger candidate pool first, then filter low-value bibliography chunks.
        candidate_count = min(len(self.chunks), max(top_k * 5, top_k))
        candidate_indices = np.argsort(scores)[::-1][:candidate_count]

        filtered: list[tuple[int, float]] = []
        fallback: list[tuple[int, float]] = []

        for idx in candidate_indices:
            idx_int = int(idx)
            score = float(scores[idx_int])
            if score <= 0:
                continue
            fallback.append((idx_int, score))
            if not _looks_like_references_or_bibliography(self.chunks[idx_int].text):
                filtered.append((idx_int, score))

        chosen = filtered[:top_k] if filtered else fallback[:top_k]

        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(chosen, start=1):
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=score, rank=rank))
        return results
