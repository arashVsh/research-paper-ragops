from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import PaperChunk, RetrievedChunk


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
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(chunk=self.chunks[int(idx)], score=score, rank=rank)
            )
        return results
