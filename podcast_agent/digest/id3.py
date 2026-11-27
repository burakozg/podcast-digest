"""Chapters written into the audio file, as ID3v2.3 ``CHAP`` and ``CTOC`` frames.

Timestamps in the feed's show notes depend on the player reading them. These
travel inside the file it downloads. The format is the ID3v2 chapter addendum;
version 2.3 because it is the one players most widely read.
"""

from __future__ import annotations

import struct
from typing import Any

#: The table of contents counts its entries in a single byte.
MAX_CHAPTERS = 255

#: "No byte offset, use the times."
_NO_OFFSET = 0xFFFFFFFF
_TOP_LEVEL_AND_ORDERED = 0x03
_UTF16 = b"\x01"


def _frame(name: bytes, body: bytes) -> bytes:
    return name + struct.pack(">I", len(body)) + b"\x00\x00" + body


def _title(text: str) -> bytes:
    return _frame(b"TIT2", _UTF16 + text.encode("utf-16"))


def _syncsafe(value: int) -> bytes:
    return bytes((value >> shift) & 0x7F for shift in (21, 14, 7, 0))


def chapter_tag(chapters: list[dict[str, Any]], duration_s: float) -> bytes:
    """A whole ID3 tag to put in front of the audio.

    ``chapters`` are ``{"start_s", "title"}`` in order; each ends where the next
    begins and the last at ``duration_s``. Empty for no chapters.
    """
    chapters = chapters[:MAX_CHAPTERS]
    if not chapters:
        return b""
    ids = [f"ch{index}".encode() + b"\x00" for index in range(len(chapters))]
    starts = [round(float(c["start_s"]) * 1000) for c in chapters]
    ends = [*starts[1:], round(duration_s * 1000)]

    frames = _frame(
        b"CTOC",
        b"toc\x00" + bytes([_TOP_LEVEL_AND_ORDERED, len(ids)]) + b"".join(ids),
    )
    for element, start, end, chapter in zip(ids, starts, ends, chapters, strict=True):
        times = struct.pack(">IIII", start, end, _NO_OFFSET, _NO_OFFSET)
        frames += _frame(b"CHAP", element + times + _title(str(chapter["title"])))
    return b"ID3\x03\x00\x00" + _syncsafe(len(frames)) + frames
