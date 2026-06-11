from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.schemas import PaperChunk

BASE_RESEARCH_QUESTIONS = [
    "What problem is this paper trying to solve?",
    "What is the main contribution of this paper?",
    "What method does this paper propose, and how does it work in simple terms?",
    "What assumptions does the paper make?",
    "What datasets and evaluation metrics are used?",
    "What are the main results?",
    "What are the limitations or weaknesses?",
    "What would reviewers likely criticize?",
    "How is this different from previous work?",
    "What experiments should I reproduce first?",
    "What future work does this paper suggest?",
    "Summarize the paper as 5 bullet points.",
    "Which section should I read first to understand the method?",
]


def _keywords(chunks: list[PaperChunk], top_n: int = 8) -> list[str]:
    text = " ".join(chunk.text[:3000] for chunk in chunks[:20]).lower()
    tokens = re.findall(r"[a-z][a-z\-]{3,}", text)
    stop = set(ENGLISH_STOP_WORDS) | {
        "paper", "method", "result", "results", "using", "based", "model", "models",
        "approach", "proposed", "show", "shows", "table", "figure", "section"
    }
    words = [t for t in tokens if t not in stop and len(t) > 3]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(top_n)]


def suggest_questions(chunks: list[PaperChunk], max_questions: int = 10) -> list[str]:
    questions = BASE_RESEARCH_QUESTIONS.copy()
    kws = _keywords(chunks)
    if kws:
        questions.insert(2, f"How does this paper use or discuss {kws[0]}?")
    if len(kws) > 1:
        questions.insert(4, f"What is the relationship between {kws[0]} and {kws[1]} in this paper?")
    if len(kws) > 2:
        questions.insert(6, f"Are there any risks, failures, or limitations related to {kws[2]}?")
    return questions[:max_questions]
