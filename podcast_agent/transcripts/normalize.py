"""Normalise transcript formats to plain text (§4 stage 3).

Supports the formats publishers actually serve for ``<podcast:transcript>``:
plain text, HTML mislabelled as text, WebVTT, SRT and the Podcasting 2.0 JSON
transcript. All inputs are untrusted.
"""

from __future__ import annotations

import json
import re

from ..sanitize import html_to_text, strip_control_chars

_VTT_TIMESTAMP = re.compile(
    r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}"
)
_SRT_INDEX = re.compile(r"^\s*\d+\s*$")
_VTT_NOTE = re.compile(r"^\s*(NOTE|STYLE|REGION)\b")
#: Inline cue markup: <v Speaker>, <c.colour>, <00:00:01.000>, <i>.
_CUE_TAGS = re.compile(r"</?[a-zA-Z0-9.:_ -]*>")
#: A cue identifier line (no timestamp, immediately followed by one). Common in
#: PRX-generated VTT where cue ids are UUIDs or numbers.
_LOOKS_HTML = re.compile(r"<\s*(html|body|div|p|br|span|script|a)\b", re.IGNORECASE)


def normalize_transcript(text: str, content_type: str = "", url: str = "") -> str:
    """Dispatch on declared type, then on content sniffing.

    Publishers mislabel constantly (SANS serves HTML as ``text/plain``), so the
    declared type is only a hint — the body is sniffed as well.
    """
    stripped = strip_control_chars(text).lstrip("﻿").strip()
    if not stripped:
        return ""

    main_type = content_type.split(";", 1)[0].strip().lower()
    lower_url = url.lower().split("?")[0]

    if main_type == "application/json" or lower_url.endswith(".json"):
        parsed = _try_json(stripped)
        if parsed is not None:
            return parsed
    if (
        main_type in ("text/vtt", "text/webvtt")
        or lower_url.endswith(".vtt")
        or stripped.startswith("WEBVTT")
    ):
        return _normalize_cues(stripped)
    if main_type in ("application/srt", "application/x-subrip", "text/srt") or lower_url.endswith(
        ".srt"
    ):
        return _normalize_cues(stripped)
    if main_type == "text/html" or _LOOKS_HTML.search(stripped[:2000]):
        return _collapse(html_to_text(stripped))
    # Unknown/plain: a body full of "-->" is a cue file regardless of labelling.
    if stripped.count("-->") > 3:
        return _normalize_cues(stripped)
    return _collapse(stripped)


def _try_json(text: str) -> str | None:
    """Podcasting 2.0 JSON transcript: {"segments": [{"speaker","body"}, ...]}."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        return None

    lines: list[str] = []
    current_speaker: str | None = None
    buffer: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        body = str(segment.get("body") or "").strip()
        if not body:
            continue
        speaker = segment.get("speaker")
        speaker = str(speaker).strip() if speaker else None
        if speaker != current_speaker:
            if buffer:
                lines.append(_speaker_line(current_speaker, " ".join(buffer)))
            current_speaker = speaker
            buffer = [body]
        else:
            buffer.append(body)
    if buffer:
        lines.append(_speaker_line(current_speaker, " ".join(buffer)))
    return _collapse("\n\n".join(lines))


def _speaker_line(speaker: str | None, body: str) -> str:
    return f"{speaker}: {body}" if speaker else body


def _normalize_cues(text: str) -> str:
    """Strip WebVTT/SRT scaffolding and collapse rollup repetition."""
    out: list[str] = []
    skip_note = False
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            skip_note = False
            continue
        if line.startswith("WEBVTT"):
            continue
        if _VTT_NOTE.match(line):
            skip_note = True
            continue
        if skip_note:
            continue
        if _VTT_TIMESTAMP.match(line) or "-->" in line:
            continue
        if _SRT_INDEX.match(line):
            continue
        # A cue identifier: no spaces and the next line is a timestamp.
        if index + 1 < len(lines) and "-->" in lines[index + 1] and " " not in line:
            continue
        cleaned = _CUE_TAGS.sub("", line).strip()
        # VTT speaker cues survive as "Speaker: text" after tag removal.
        if not cleaned:
            continue
        # Rollup captions repeat the previous line with one word appended;
        # dropping exact repeats handles the common case cheaply.
        if out and out[-1] == cleaned:
            continue
        out.append(cleaned)
    return _collapse(" ".join(out))


def _collapse(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
