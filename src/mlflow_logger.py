from __future__ import annotations

import hashlib
from contextlib import suppress
from typing import Any

from src.config import CONFIG


def _safe_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except Exception:
        return None


def setup_mlflow() -> Any | None:
    mlflow = _get_mlflow()
    if mlflow is None:
        return None

    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    return mlflow


def log_query_event(
    question: str,
    paper_titles: list[str],
    params: dict[str, object],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
) -> None:
    with suppress(Exception):
        mlflow = setup_mlflow()

        if mlflow is None:
            return

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