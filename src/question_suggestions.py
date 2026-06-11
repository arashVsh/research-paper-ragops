from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.chunking import TextChunk


@dataclass
class PaperSummary:
    title: str
    short_summary: str
    detected_sections: list[str]
    key_terms: list[str]


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "been", "have", "has", "had", "using", "used", "use", "into", "their",
    "there", "these", "those", "which", "while", "where", "when", "than",
    "then", "they", "them", "its", "can", "may", "our", "we", "in", "on",
    "of", "to", "a", "an", "is", "as", "by", "or", "be", "at", "it"
}


SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "introduction": r"\bintroduction\b",
    "method": r"\b(method|methodology|approach|proposed method|model architecture)\b",
    "experiments": r"\b(experiment|experimental setup|evaluation|results)\b",
    "limitations": r"\b(limitation|limitations|future work|discussion)\b",
    "conclusion": r"\b(conclusion|conclusions)\b",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    text = _normalize_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if 40 <= len(s.strip()) <= 350]


def _extract_key_terms(text: str, top_n: int = 8) -> list[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]

    counts = Counter(words)

    useful_terms = []
    for word, _ in counts.most_common(40):
        if word not in useful_terms:
            useful_terms.append(word)
        if len(useful_terms) >= top_n:
            break

    return useful_terms


def _detect_sections(text: str) -> list[str]:
    lower = text.lower()
    detected = []

    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, lower):
            detected.append(section)

    return detected


def summarize_paper(chunks: list[TextChunk]) -> PaperSummary:
    if not chunks:
        return PaperSummary(
            title="No paper loaded",
            short_summary="No readable paper text was available.",
            detected_sections=[],
            key_terms=[],
        )

    title = chunks[0].paper_title

    # Use early chunks because they usually contain title, abstract, and introduction.
    early_text = " ".join(chunk.text for chunk in chunks[:6])
    full_text_sample = " ".join(chunk.text for chunk in chunks[:20])

    sentences = _split_sentences(early_text)
    detected_sections = _detect_sections(full_text_sample)
    key_terms = _extract_key_terms(full_text_sample)

    # Prefer abstract-like/contribution-like sentences.
    priority_keywords = [
        "propose", "proposed", "present", "introduce", "contribution",
        "method", "approach", "framework", "model", "evaluate", "results",
        "show", "demonstrate"
    ]

    scored = []
    for sentence in sentences:
        lower = sentence.lower()
        score = sum(1 for kw in priority_keywords if kw in lower)
        score += min(len(sentence) / 180, 1.0)
        scored.append((score, sentence))

    scored.sort(reverse=True, key=lambda x: x[0])

    selected = [s for _, s in scored[:3]]
    if not selected:
        selected = sentences[:3]

    if selected:
        short_summary = " ".join(selected)
    else:
        short_summary = (
            "The app extracted the paper text, but it could not confidently create "
            "a readable summary. Try asking about the main contribution, method, or results."
        )

    return PaperSummary(
        title=title,
        short_summary=short_summary,
        detected_sections=detected_sections,
        key_terms=key_terms,
    )


def suggest_questions(chunks: list[TextChunk]) -> list[str]:
    summary = summarize_paper(chunks)
    sections = set(summary.detected_sections)
    terms = summary.key_terms

    questions: list[str] = []

    # Always useful for research papers
    questions.append("What is the main contribution of this paper?")
    questions.append("What problem does this paper try to solve?")

    if "method" in sections:
        questions.append("What method does this paper propose, and how does it work in simple terms?")
        questions.append("What are the main steps of the proposed method?")

    if "experiments" in sections:
        questions.append("What experiments were conducted, and what were the main results?")
        questions.append("What datasets, baselines, and metrics were used?")

    if "limitations" in sections:
        questions.append("What are the limitations or weaknesses of this paper?")
        questions.append("What future work does the paper suggest?")
    else:
        questions.append("What could be possible limitations or reviewer concerns?")

    if "conclusion" in sections:
        questions.append("What is the main conclusion of the paper?")

    if terms:
        questions.append(f"How does the paper use or define {terms[0]}?")
    if len(terms) > 1:
        questions.append(f"What is the relationship between {terms[0]} and {terms[1]} in this paper?")

    questions.append("How is this paper different from prior work?")
    questions.append("What should I remember from this paper for a literature review?")

    # Remove duplicates while preserving order
    unique_questions = []
    seen = set()
    for q in questions:
        if q not in seen:
            unique_questions.append(q)
            seen.add(q)

    return unique_questions[:10]