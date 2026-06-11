from __future__ import annotations

from src.schemas import PaperChunk
from src.research_features import build_structured_summary, extract_key_terms


def suggest_questions(chunks: list[PaperChunk], max_questions: int = 10) -> list[str]:
    summary = build_structured_summary(chunks)
    terms = extract_key_terms(chunks, top_n=5)

    questions: list[str] = [
        "What is the main contribution of this paper?",
        "What problem does this paper try to solve?",
        "What method does this paper propose, and how does it work in simple terms?",
        "What are the main steps of the proposed method?",
        "What experiments were conducted, and what were the main results?",
        "What datasets, baselines, and metrics were used?",
        "What are the limitations or weaknesses of this paper?",
        "What would reviewers likely criticize?",
        "How is this paper different from prior work?",
        "What should I remember from this paper for a literature review?",
    ]

    if terms:
        questions.insert(3, f"How does this paper use or discuss {terms[0]}?")
    if len(terms) > 1:
        questions.insert(5, f"What is the relationship between {terms[0]} and {terms[1]} in this paper?")

    # If the heuristic summary found no obvious limitations, keep reviewer question visible.
    if summary.limitations.startswith("Not clearly identified"):
        questions.append("What possible limitations are not explicitly discussed?")

    unique_questions = []
    seen = set()
    for question in questions:
        if question not in seen:
            unique_questions.append(question)
            seen.add(question)

    return unique_questions[:max_questions]
