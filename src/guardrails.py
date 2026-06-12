from __future__ import annotations

import re


PROMPT_INJECTION_PATTERNS = [
    # Ignore / override instructions
    r"\bignore\b.*\b(previous|prior|above|earlier|system|developer)\b.*\b(instruction|instructions|prompt|message|messages|rules)\b",
    r"\bdisregard\b.*\b(previous|prior|above|earlier|system|developer)\b.*\b(instruction|instructions|prompt|message|messages|rules)\b",
    r"\boverride\b.*\b(system|developer|previous|prior|above)\b.*\b(instruction|instructions|rules|prompt)\b",
    r"\bforget\b.*\b(previous|prior|above|earlier)\b.*\b(instruction|instructions|rules|prompt)\b",

    # System prompt / hidden prompt extraction
    r"\b(reveal|show|print|display|repeat|tell me)\b.*\b(system prompt|developer prompt|hidden prompt|initial instructions|internal instructions)\b",
    r"\bwhat\b.*\b(system prompt|developer prompt|hidden prompt|initial instructions|internal instructions)\b",
    r"\bcopy\b.*\b(system prompt|developer prompt|hidden prompt|initial instructions)\b",

    # Jailbreak / bypass
    r"\bjailbreak\b",
    r"\bbypass\b.*\b(rule|rules|guardrail|guardrails|safety|filter|policy|policies)\b",
    r"\bdisable\b.*\b(rule|rules|guardrail|guardrails|safety|filter|policy|policies)\b",
    r"\bdo anything now\b",
    r"\bdan mode\b",
    r"\bunrestricted mode\b",
    r"\bdeveloper mode\b",

    # Role manipulation
    r"\byou are now\b",
    r"\bact as\b.*\b(unrestricted|uncensored|jailbroken|developer mode)\b",
    r"\bpretend\b.*\b(no rules|unrestricted|uncensored|jailbroken)\b",

    # Citation / context manipulation
    r"\bfake\b.*\b(citation|citations|source|sources|reference|references)\b",
    r"\binvent\b.*\b(citation|citations|source|sources|reference|references)\b",
    r"\bmake up\b.*\b(citation|citations|source|sources|reference|references)\b",
    r"\bdo not cite\b",
    r"\banswer without citations\b",

    # Exfiltration / secrets
    r"\b(reveal|show|print|display|leak)\b.*\b(api key|secret|token|password|environment variable|env variable)\b",
]


def normalize_text(text: str) -> str:
    """Normalize text so simple obfuscations are easier to detect."""
    text = text.lower()

    # Remove zero-width characters.
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Common leetspeak replacements.
    replacements = {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Collapse punctuation-heavy spacing.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_prompt_injection(text: str) -> list[str]:
    normalized = normalize_text(text)
    warnings: list[str] = []

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            warnings.append("Possible prompt-injection or unsafe instruction detected.")
            break

    return warnings


def is_prompt_injection(text: str) -> bool:
    return bool(detect_prompt_injection(text))


def sanitize_for_prompt(text: str, max_chars: int = 9000) -> str:
    """Limit and sanitize untrusted document text before sending it to an LLM."""
    text = text.replace("```", "'''")
    text = text[:max_chars]
    return text