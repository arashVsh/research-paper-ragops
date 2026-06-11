from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from collections import Counter

from src.schemas import PaperChunk


@dataclass
class StructuredSummary:
    paper_title: str
    problem: str
    method: str
    contribution: str
    experiments: str
    key_results: str
    limitations: str
    key_terms: list[str]


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "been", "have", "has", "had", "using", "used", "use", "into", "their",
    "there", "these", "those", "which", "while", "where", "when", "than",
    "then", "they", "them", "its", "can", "may", "our", "we", "in", "on",
    "of", "to", "a", "an", "is", "as", "by", "or", "be", "at", "it", "paper",
    "method", "approach", "model", "models", "result", "results", "show", "shows",
    "table", "figure", "section", "proposed", "based", "data", "learning"
}

CATEGORY_KEYWORDS = {
    "problem": [
        "problem", "challenge", "issue", "difficulty", "weakness", "limitation", "motivation",
        "aim", "goal", "objective", "robustness", "evaluate", "evaluation"
    ],
    "method": [
        "propose", "proposed", "method", "approach", "framework", "algorithm", "architecture",
        "we introduce", "we present", "we develop", "we use", "protocol"
    ],
    "contribution": [
        "contribution", "novel", "first", "we show", "we demonstrate", "we introduce",
        "we propose", "our work", "our approach"
    ],
    "experiments": [
        "experiment", "experiments", "evaluation", "benchmark", "dataset", "datasets",
        "baseline", "baselines", "metrics", "cifar", "mnist", "imagenet"
    ],
    "key_results": [
        "results", "achieve", "achieves", "outperform", "improve", "improves", "accuracy",
        "robust", "performance", "state-of-the-art", "effective", "stronger"
    ],
    "limitations": [
        "limitation", "limitations", "future work", "however", "although", "fails", "failure",
        "weakness", "drawback", "expensive", "cost", "not robust", "cannot"
    ],
}

FALLBACK_TEXT = "Not clearly identified from the extracted text. Ask a targeted question for a more reliable answer."


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    text = _normalize(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = []
    for sentence in sentences:
        sentence = sentence.strip()
        if 45 <= len(sentence) <= 420 and not _looks_like_reference_sentence(sentence):
            clean.append(sentence)
    return clean


def _looks_like_reference_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    reference_signals = [
        "proceedings of", "arxiv preprint", "conference on", "transactions on", "journal of",
        "in proceedings", "doi", "isbn", "vol.", "pp."
    ]
    return sum(signal in lower for signal in reference_signals) >= 2


def extract_key_terms(chunks: list[PaperChunk], top_n: int = 8) -> list[str]:
    text = " ".join(chunk.text[:2500] for chunk in chunks[:25]).lower()
    words = re.findall(r"\b[a-z][a-z\-]{3,}\b", text)
    words = [w for w in words if w not in STOPWORDS]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(top_n)]


def _best_sentence(sentences: list[str], keywords: list[str]) -> str:
    best_score = 0.0
    best_sentence = ""

    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        score = sum(2.0 for kw in keywords if kw in lower)
        # Earlier abstract/introduction sentences are often more useful.
        score += max(0.0, 1.0 - index / max(len(sentences), 1))
        # Prefer readable medium-length sentences.
        score += min(len(sentence) / 220, 1.0)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence if best_score >= 2.5 else FALLBACK_TEXT


def build_structured_summary(chunks: list[PaperChunk]) -> StructuredSummary:
    if not chunks:
        return StructuredSummary(
            paper_title="No paper loaded",
            problem=FALLBACK_TEXT,
            method=FALLBACK_TEXT,
            contribution=FALLBACK_TEXT,
            experiments=FALLBACK_TEXT,
            key_results=FALLBACK_TEXT,
            limitations=FALLBACK_TEXT,
            key_terms=[],
        )

    paper_title = chunks[0].paper_title
    # Early text usually has abstract/introduction; later sample helps find results/limitations.
    sample_text = " ".join(chunk.text for chunk in chunks[:30])
    sentences = _split_sentences(sample_text)

    return StructuredSummary(
        paper_title=paper_title,
        problem=_best_sentence(sentences, CATEGORY_KEYWORDS["problem"]),
        method=_best_sentence(sentences, CATEGORY_KEYWORDS["method"]),
        contribution=_best_sentence(sentences, CATEGORY_KEYWORDS["contribution"]),
        experiments=_best_sentence(sentences, CATEGORY_KEYWORDS["experiments"]),
        key_results=_best_sentence(sentences, CATEGORY_KEYWORDS["key_results"]),
        limitations=_best_sentence(sentences, CATEGORY_KEYWORDS["limitations"]),
        key_terms=extract_key_terms(chunks),
    )


def build_structured_summaries_by_paper(chunks: list[PaperChunk]) -> list[StructuredSummary]:
    by_title: dict[str, list[PaperChunk]] = {}
    for chunk in chunks:
        by_title.setdefault(chunk.paper_title, []).append(chunk)

    summaries = []
    for title in sorted(by_title):
        summaries.append(build_structured_summary(by_title[title]))
    return summaries


def build_comparison_rows(chunks: list[PaperChunk]) -> list[dict[str, str]]:
    rows = []
    for summary in build_structured_summaries_by_paper(chunks):
        rows.append(
            {
                "Paper": summary.paper_title,
                "Problem": summary.problem,
                "Method": summary.method,
                "Key result": summary.key_results,
                "Limitation": summary.limitations,
            }
        )
    return rows


def reviewer_concern_prompt() -> str:
    return (
        "Act as a critical but fair academic reviewer. What are the most likely reviewer concerns, "
        "missing experiments, weak assumptions, unclear claims, and limitations of this paper? "
        "Give the answer as concise bullet points with citations."
    )


def summary_to_dict(summary: StructuredSummary) -> dict[str, object]:
    return asdict(summary)
