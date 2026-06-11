from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Research Paper RAGOps Assistant"
    chunk_size: int = 1200
    chunk_overlap: int = 220
    top_k: int = 5
    max_uploaded_files: int = 5
    max_answer_context_chars: int = 9000
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "research-paper-ragops")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


CONFIG = AppConfig()
