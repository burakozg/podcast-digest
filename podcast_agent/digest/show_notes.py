"""What a podcast player shows for one narrated week: a title and its notes.

Built from what the digest already holds, so it costs no model call. Everything
here started as feed or model text, so it is reduced to plain text and then
escaped before it goes into HTML.
"""

from __future__ import annotations

import re
from html import escape
from typing import Any

from ..sanitize import md_to_speech_text, safe_url

#: Escapes still standing after rendering: stored text that arrived already
#: escaped and was escaped again.
_LEFTOVER_ESCAPE = re.compile(r"\\+(?=[^\w\s]|_)")


def _plain(markdown: Any) -> str:
    text = md_to_speech_text(str(markdown or ""))
    return " ".join(_LEFTOVER_ESCAPE.sub("", text).split())


def _themes(synthesis: dict[str, Any] | None) -> list[tuple[str, str]]:
    return [
        (_plain(t.get("title")), _plain(t.get("summary")))
        for t in (synthesis or {}).get("themes") or []
        if _plain(t.get("title"))
    ]


def lead_theme(synthesis: dict[str, Any] | None) -> str:
    """The week's first theme, or nothing when it has none."""
    themes = _themes(synthesis)
    return themes[0][0] if themes else ""


def episode_title(period_key: str, synthesis: dict[str, Any] | None) -> str:
    """Led by the week's first theme, when it has one."""
    year, week = period_key.split("-W")
    if theme := lead_theme(synthesis):
        return f"W{int(week)} · {theme}"
    return f"Week {int(week)}, {year}"


def timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def show_notes(
    period_key: str,
    synthesis: dict[str, Any] | None,
    chapters: list[dict[str, Any]],
    top_picks: list[dict[str, Any]],
    episode_count: int,
) -> str:
    """The episode description, as the small subset of HTML players render.

    The first line is what a player previews. Chapters come next, as one
    ``HH:MM:SS title`` paragraph each, which is the form players read them from.
    """
    year, week = period_key.split("-W")
    themes = _themes(synthesis)
    preview = " · ".join(title for title, _ in themes) or (
        f"{episode_count} episodes from week {int(week)} of {year}."
    )
    parts = [f"<p>{escape(preview)}</p>"]

    parts += [f"<p>{timestamp(c['start_s'])} {escape(str(c['title']))}</p>" for c in chapters]

    if themes:
        parts.append("<p><b>This week</b></p>")
        parts += [f"<p><b>{escape(t)}.</b> {escape(s)}</p>" for t, s in themes]

    for heading, key in (
        ("Where they disagreed", "disagreements"),
        ("New since last week", "whats_new"),
    ):
        lines = [_plain(x) for x in (synthesis or {}).get(key) or []]
        if lines:
            parts.append(f"<p><b>{heading}</b></p>")
            parts.append("<ul>" + "".join(f"<li>{escape(x)}</li>" for x in lines) + "</ul>")

    if top_picks:
        items = []
        for episode in top_picks:
            label = escape(f"{_plain(episode['podcast_name'])}: {_plain(episode['title'])}")
            if link := safe_url(episode.get("link")):
                label = f'<a href="{escape(link)}">{label}</a>'
            why = _plain(episode.get("why_it_matters"))
            items.append(f"<li>{label}{' — ' + escape(why) if why else ''}</li>")
        parts.append("<p><b>Top picks</b></p>")
        parts.append("<ul>" + "".join(items) + "</ul>")

    return "".join(parts)
