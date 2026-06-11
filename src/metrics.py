from __future__ import annotations

import statistics

from src.schemas import AnswerResult, RetrievedChunk


def compute_query_metrics(
    results: list[RetrievedChunk],
    answer: AnswerResult,
    latency_seconds: float,
) -> dict[str, float]:
    scores = [r.score for r in results]
    return {
        "latency_seconds": float(latency_seconds),
        "retrieved_chunks": float(len(results)),
        "mean_retrieval_score": float(statistics.mean(scores)) if scores else 0.0,
        "max_retrieval_score": float(max(scores)) if scores else 0.0,
        "answer_chars": float(len(answer.answer)),
        "used_llm": 1.0 if answer.used_llm else 0.0,
        "guardrail_warning_count": float(len(answer.guardrail_warnings)),
        "citation_count": float(len(answer.citations)),
    }
