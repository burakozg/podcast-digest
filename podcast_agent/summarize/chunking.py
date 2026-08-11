"""Transcript chunking for Tier-1 map-reduce (§4 stage 4).

Fixed-size chunking on paragraph boundaries, as specified — deliberately simple.
Falls back to sentence then hard-character splits so a transcript with no
paragraph breaks (every ASR output) still chunks sensibly instead of producing
one oversized piece.
"""

from __future__ import annotations

import re

from ..utils import CHARS_PER_TOKEN, estimate_tokens

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def needs_map_reduce(text: str, max_input_tokens: int) -> bool:
    return estimate_tokens(text) > max_input_tokens


def chunk_transcript(text: str, target_tokens: int) -> list[str]:
    """Split ``text`` into chunks of roughly ``target_tokens`` each.

    Chunks never split a paragraph unless the paragraph alone exceeds the target.
    """
    if not text.strip():
        return []
    target_chars = max(1, target_tokens * CHARS_PER_TOKEN)

    units = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if len(units) == 1:
        units = _split_oversized(units[0], target_chars)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if len(unit) > target_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            chunks.extend(_split_oversized(unit, target_chars))
            continue
        # +2 for the paragraph separator we will rejoin with.
        if current and current_len + len(unit) + 2 > target_chars:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(unit)
        current_len += len(unit) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_oversized(text: str, target_chars: int) -> list[str]:
    """Break one long block on sentence boundaries, then hard-split if needed."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > target_chars:
            if current:
                pieces.append(" ".join(current))
                current, current_len = [], 0
            pieces.extend(
                sentence[i : i + target_chars] for i in range(0, len(sentence), target_chars)
            )
            continue
        if current and current_len + len(sentence) + 1 > target_chars:
            pieces.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces or [text[:target_chars]]


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard cap for the single-call path so one huge input cannot blow the context."""
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "\n\n[transcript truncated]"
