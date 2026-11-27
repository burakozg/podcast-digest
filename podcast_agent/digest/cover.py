"""Artwork for one week's episode: its number and lead theme over a waveform.

Drawn on request and kept in memory, so there are no image files to manage. The
typeface is the one Pillow carries, so nothing depends on what fonts the
container has.
"""

from __future__ import annotations

import colorsys
import hashlib
import io
import math
import random
import unicodedata
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

#: Square, and inside the 1400-3000 range podcast players ask for.
SIZE = 1500

#: Change when the drawing changes: players cache artwork by its address.
_DESIGN = "1"

_MARGIN = 120
_WIDTH = SIZE - 2 * _MARGIN
_THEME_SIZES = (112, 96, 82)
_THEME_LINES = 3
_WHITE = (255, 255, 255)
_MUTED = (190, 194, 224)

#: Punctuation the typeface may lack, which decomposing does not remove.
_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00b7": "-",
    }
)

RGB = tuple[int, int, int]


def version(period_key: str, theme: str) -> str:
    """Changes whenever the picture would."""
    seed = f"{_DESIGN}\n{period_key}\n{theme}".encode()
    return hashlib.sha256(seed).hexdigest()[:10]


def _printable(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(_PUNCTUATION))
    return " ".join(decomposed.encode("ascii", "ignore").decode().split())


def _colour(hue: float, saturation: float, value: float) -> RGB:
    r, g, b = colorsys.hsv_to_rgb(hue % 1, saturation, value)
    return round(r * 255), round(g * 255), round(b * 255)


def _blend(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _text(
    draw: ImageDraw.ImageDraw, at: tuple[int, int], text: str, size: int, fill: RGB, bold: int = 0
) -> None:
    # The typeface comes in one weight; an outline in the same colour thickens it.
    draw.text(
        at,
        text,
        font=ImageFont.load_default(size),
        fill=fill,
        stroke_width=bold,
        stroke_fill=fill,
    )


def _wrap(text: str, size: int, bold: int) -> list[str]:
    font = ImageFont.load_default(size)
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and font.getlength(candidate) + 2 * bold <= _WIDTH:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def _fit(text: str) -> tuple[int, list[str]]:
    """The largest size at which the theme fits, cut short if none does."""
    for size in _THEME_SIZES:
        lines = _wrap(text, size, 2)
        font = ImageFont.load_default(size)
        if len(lines) <= _THEME_LINES and all(font.getlength(x) <= _WIDTH for x in lines):
            return size, lines
    size = _THEME_SIZES[-1]
    font = ImageFont.load_default(size)
    lines = _wrap(text, size, 2)
    kept = lines[:_THEME_LINES]
    for index, line in enumerate(kept):
        cut = index == len(kept) - 1 and len(lines) > len(kept)
        if cut or font.getlength(line) > _WIDTH:
            while line and font.getlength(f"{line}...") > _WIDTH - 8:
                line = line[:-1]
            kept[index] = f"{line.rstrip()}..."
    return size, kept


@lru_cache(maxsize=64)
def render(period_key: str, theme: str) -> bytes:
    """A JPEG. Each week has its own colour and waveform, the same every time."""
    year, week_text = period_key.split("-W")
    week = int(week_text)
    # Steps of the golden ratio, so neighbouring weeks are far apart in colour.
    hue = (week * 0.618034) % 1

    image = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(image)
    top, bottom = _colour(hue, 0.65, 0.17), _colour(hue + 0.08, 0.70, 0.30)
    for y in range(SIZE):
        draw.line([(0, y), (SIZE, y)], fill=_blend(top, bottom, y / SIZE))

    _text(draw, (_MARGIN, 90), "Podcast Digest", 72, _MUTED, 1)
    _text(draw, (_MARGIN - 10, 190), f"W{week}", 400, _WHITE, 8)
    _text(draw, (_MARGIN, 625), year, 84, _MUTED, 1)

    if words := _printable(theme):
        size, lines = _fit(words)
        for index, line in enumerate(lines):
            _text(draw, (_MARGIN, 765 + index * round(size * 1.22)), line, size, _WHITE, 2)

    bars = 33
    step = _WIDTH / bars
    middle = 1345
    noise = random.Random(period_key)  # noqa: S311 - decoration, not a secret
    left, right = _colour(hue + 0.45, 0.60, 0.95), _colour(hue + 0.25, 0.50, 1.0)
    for index in range(bars):
        t = index / (bars - 1)
        height = 30 + 180 * math.sin(math.pi * (t * 0.9 + 0.05)) * noise.uniform(0.3, 1.0)
        x = _MARGIN + index * step
        draw.rounded_rectangle(
            [x, middle - height / 2, x + step * 0.56, middle + height / 2],
            radius=step * 0.28,
            fill=_blend(left, right, t),
        )

    out = io.BytesIO()
    image.save(out, "JPEG", quality=88, optimize=True)
    return out.getvalue()
