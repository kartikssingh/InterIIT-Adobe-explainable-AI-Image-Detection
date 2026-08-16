"""Cleaning and length control for generated explanations.

VLMs habitually open with filler ("Sure! In this image…"), repeat themselves and
ignore length instructions. Submissions are judged on a hard word budget, so the
text is normalised here rather than trusting the model to comply.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

#: Openers that carry no information and are stripped from the front.
_FILLER_PREFIXES: Tuple[str, ...] = (
    "sure,",
    "sure!",
    "certainly,",
    "of course,",
    "here is",
    "here's",
    "in this image,",
    "the image shows that",
    "answer:",
    "description:",
    "observation:",
    "assistant:",
    "output:",
)

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def strip_filler(text: str) -> str:
    """Remove conversational preambles and stray markdown."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?|```$", "", cleaned).strip()
    cleaned = cleaned.strip('"').strip()

    changed = True
    while changed:
        changed = False
        lowered = cleaned.lower()
        for prefix in _FILLER_PREFIXES:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip(" :,-—")
                changed = True
                break

    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def dedupe_sentences(sentences: Sequence[str]) -> List[str]:
    """Drop repeated sentences (case/punctuation insensitive), keeping order."""
    seen: set[str] = set()
    unique: List[str] = []
    for sentence in sentences:
        key = re.sub(r"[^a-z0-9 ]+", "", sentence.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(sentence)
    return unique


def limit_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return text
    sentences = dedupe_sentences(split_sentences(text))
    return " ".join(sentences[:max_sentences])


def limit_words(text: str, max_words: int) -> Tuple[str, bool]:
    """Trim to ``max_words``. Returns ``(text, truncated)``.

    Truncation prefers the last complete sentence inside the budget so the text
    never ends mid-clause; only if no sentence fits does it cut hard.
    """
    words = text.split()
    if len(words) <= max_words:
        return text, False

    clipped = " ".join(words[:max_words])
    for terminator in (". ", "! ", "? "):
        index = clipped.rfind(terminator)
        if index > 0 and len(clipped[: index + 1].split()) >= max(3, max_words // 2):
            return clipped[: index + 1].strip(), True

    return clipped.rstrip(",;:- ") + ".", True


def ensure_terminal_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if stripped and stripped[-1] not in ".!?":
        return stripped + "."
    return stripped


def polish(text: str, max_words: int = 50, max_sentences: int = 3) -> Tuple[str, bool]:
    """Full clean-up chain. Returns ``(text, truncated)``."""
    cleaned = collapse_whitespace(strip_filler(text))
    if not cleaned:
        return "", False
    cleaned = limit_sentences(cleaned, max_sentences)
    cleaned, truncated = limit_words(cleaned, max_words)
    return ensure_terminal_punctuation(cleaned), truncated


def word_count(text: str) -> int:
    return len(text.split())
