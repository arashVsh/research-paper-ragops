from __future__ import annotations

import hashlib
from contextlib import suppress

import mlflow

from src.config import CONFIG


def _safe_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)


def log_query_event(
    question: str,
    paper_titles: list[str],
    params: dict[str, object],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
) -> None:
    """Log metadata, not full private paper text."""
    with suppress(Exception):
        setup_mlflow()
        with mlflow.start_run(run_name="paper-question"):
            mlflow.log_param("question_hash", _safe_hash(question))
            mlflow.log_param("paper_count", len(paper_titles))
            mlflow.log_param("paper_titles", "; ".join(paper_titles)[:500])
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
