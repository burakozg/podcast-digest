"""Chapter frames inside the narrated MP3."""

from __future__ import annotations

import struct
from typing import Any

from podcast_agent.digest.id3 import MAX_CHAPTERS, chapter_tag


def _frames(data: bytes) -> list[tuple[bytes, bytes]]:
    found = []
    at = 0
    while at < len(data):
        size = struct.unpack(">I", data[at + 4 : at + 8])[0]
        found.append((data[at : at + 4], data[at + 10 : at + 10 + size]))
        at += 10 + size
    return found


def _body(tag: bytes) -> bytes:
    assert tag[:6] == b"ID3\x03\x00\x00"
    size = 0
    for byte in tag[6:10]:
        assert byte < 0x80
        size = (size << 7) | byte
    assert size == len(tag) - 10
    return tag[10:]


def test_contents_and_chapters_are_laid_out_as_players_expect() -> None:
    chapters = [{"start_s": 0.0, "title": "Intro"}, {"start_s": 8.2, "title": "Agents — och ö"}]

    toc, first, second = _frames(_body(chapter_tag(chapters, 20.5)))

    assert toc == (b"CTOC", b"toc\x00\x03\x02ch0\x00ch1\x00")
    assert first[0] == second[0] == b"CHAP"
    assert struct.unpack(">II", first[1][4:12]) == (0, 8200)
    assert struct.unpack(">II", second[1][4:12]) == (8200, 20500)
    assert second[1][12:20] == b"\xff" * 8
    ((name, title),) = _frames(second[1][20:])
    assert name == b"TIT2"
    assert title[0] == 1
    assert title[1:].decode("utf-16") == "Agents — och ö"


def test_no_chapters_means_no_tag() -> None:
    assert chapter_tag([], 10) == b""


def test_chapters_beyond_what_the_contents_can_count_are_dropped() -> None:
    chapters: list[dict[str, Any]] = [{"start_s": i, "title": str(i)} for i in range(300)]

    frames = _frames(_body(chapter_tag(chapters, 400)))

    assert len(frames) == MAX_CHAPTERS + 1
