"""Timing narrated audio."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_agent.digest.mp3 import duration_s, estimated_file_duration_s

#: MPEG-2 layer III, 128 kbit/s, 24 kHz mono: what the speech server returns.
FRAME = bytes([0xFF, 0xF3, 0xC4, 0xC4]) + bytes(380)
TAG = b"ID3\x04\x00\x00\x00\x00\x00\x05" + bytes(5)


def test_joined_pieces_are_counted_frame_by_frame() -> None:
    assert duration_s((TAG + FRAME * 50) * 3) == pytest.approx(150 * 0.024)


def test_bytes_without_frames_have_no_duration() -> None:
    assert duration_s(b"[1]") is None


def test_a_file_is_timed_from_its_first_frame(tmp_path: Path) -> None:
    path = tmp_path / "a.mp3"
    path.write_bytes(TAG + FRAME * 5000)

    assert estimated_file_duration_s(path) == pytest.approx(120, abs=0.01)


def test_an_unrecognised_file_is_not_guessed_at(tmp_path: Path) -> None:
    path = tmp_path / "a.mp3"
    path.write_bytes(bytes(100))

    assert estimated_file_duration_s(path) is None
