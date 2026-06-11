from __future__ import annotations

import re

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|prior) instructions",
    r"disregard (all )?(previous|above|prior) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"print (the )?(system|developer) prompt",
    r"you are now",
    r"forget your instructions",
    r"act as (an )?unrestricted",
    r"jailbreak",
    r"do anything now",
    r"bypass (the )?(rules|guardrails|safety)",
]


def detect_prompt_injection(text: str) -> list[str]:
    warnings: list[str] = []
    lower = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lower):
            warnings.append(f"Possible prompt-injection pattern detected: {pattern}")
    return warnings


def sanitize_for_prompt(text: str, max_chars: int = 9000) -> str:
    """Limit untrusted document text before sending it to an LLM."""
    text = text.replace("```", "'''")
    text = text[:max_chars]
    return text
