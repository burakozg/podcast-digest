"""Per-episode artwork."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from podcast_agent.digest.cover import SIZE, render, version


@pytest.mark.parametrize(
    "theme",
    [
        "",
        "Agentic Identity",
        "A very long theme about operationalising agentic defence " * 6,
        "Unbroken" * 60,
        "Curly “quotes” — dashes, Åccents and 漢字",
    ],
)
def test_any_theme_draws_a_square_jpeg(theme: str) -> None:
    image = Image.open(io.BytesIO(render("2026-W48", theme)))

    assert image.format == "JPEG"
    assert image.size == (SIZE, SIZE)


def test_the_same_week_always_draws_the_same_picture() -> None:
    first = render("2026-W48", "Agents")
    render.cache_clear()

    assert render("2026-W48", "Agents") == first


def test_the_address_changes_with_the_picture() -> None:
    assert version("2026-W48", "Agents") != version("2026-W48", "Models")
    assert version("2026-W48", "Agents") != version("2026-W47", "Agents")
