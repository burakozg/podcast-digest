"""Just enough MP3 to say how long narrated audio plays for.

Chapter times are the running total of the pieces a digest is synthesised in, so
each has to be measured as it arrives. Layer III only, which is what a speech
server returns when asked for MP3.
"""

from __future__ import annotations

from pathlib import Path

#: kbit/s by header index, for MPEG-1 and for MPEG-2/2.5.
_BITRATES_V1 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_BITRATES_V2 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
#: Hz by version bits: 3 is MPEG-1, 2 is MPEG-2, 0 is MPEG-2.5.
_SAMPLE_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}

#: Enough to get past a leading tag to the first frame.
_HEAD_BYTES = 64 * 1024


def _frame(data: bytes, at: int) -> tuple[int, float, int] | None:
    """``(length in bytes, seconds, bits per second)`` of the frame at ``at``."""
    if at + 4 > len(data) or data[at] != 0xFF or data[at + 1] & 0xE0 != 0xE0:
        return None
    version = (data[at + 1] >> 3) & 3
    layer = (data[at + 1] >> 1) & 3
    bitrate_index = data[at + 2] >> 4
    rate_index = (data[at + 2] >> 2) & 3
    if layer != 1 or version == 1 or not 0 < bitrate_index < 15 or rate_index == 3:
        return None
    bitrate = (_BITRATES_V1 if version == 3 else _BITRATES_V2)[bitrate_index] * 1000
    sample_rate = _SAMPLE_RATES[version][rate_index]
    samples = 1152 if version == 3 else 576
    padding = (data[at + 2] >> 1) & 1
    return samples // 8 * bitrate // sample_rate + padding, samples / sample_rate, bitrate


def _after_tag(data: bytes, at: int) -> int | None:
    """Where an ID3v2 tag starting at ``at`` ends, or None if there is none."""
    if data[at : at + 3] != b"ID3" or at + 10 > len(data):
        return None
    size = 0
    for byte in data[at + 6 : at + 10]:
        size = (size << 7) | (byte & 0x7F)
    return at + 10 + size


def duration_s(data: bytes) -> float | None:
    """Playing time, counted frame by frame. None if there are no frames."""
    at = 0
    total = 0.0
    while at < len(data):
        if (end := _after_tag(data, at)) is not None:
            at = end
        elif (frame := _frame(data, at)) is not None:
            at += frame[0]
            total += frame[1]
        else:
            at += 1
    return total or None


def estimated_file_duration_s(path: Path) -> float | None:
    """Playing time from the size and the first frame's bitrate.

    For files narrated before durations were recorded. Exact enough for a
    constant bitrate, and reads only the start of a file of tens of megabytes.
    """
    with path.open("rb") as handle:
        head = handle.read(_HEAD_BYTES)
    at = 0
    while at < len(head):
        if (end := _after_tag(head, at)) is not None:
            at = end
        elif (frame := _frame(head, at)) is not None:
            return path.stat().st_size * 8 / frame[2]
        else:
            at += 1
    return None
