"""One episode as a standalone Markdown file, on request from the console.

Distinct from the vault notes :mod:`podcast_agent.digest.generate` writes at
digest time in two ways that matter. It is *ad hoc* — any summarised episode can
be exported, not only those a weekly digest picked up — and its output is meant
to leave the vault, so it carries no Obsidian frontmatter, no wikilinks and no
backlink to a digest the recipient cannot open.

The view and the sanitisation are shared with the digest, so an exported episode
says exactly what the vault note says about it.
"""

from __future__ import annotations

from ..config import Settings
from ..db import Doc
from ..sanitize import slugify
from ..utils import to_local, utcnow
from .generate import BASIS_LABELS, _build_env, summary_view


def episode_markdown(settings: Settings, episode: Doc) -> str:
    """Render ``episode`` as a self-contained Markdown document.

    The caller is responsible for establishing that there is a summary to
    export; without one the document would be a header and nothing else.
    """
    view = summary_view(settings, episode, BASIS_LABELS)
    generated = to_local(utcnow(), settings.tz)
    return str(
        _build_env()
        .get_template("export.md.j2")
        .render(e=view, generated_date=generated.date().isoformat())
    )


def export_filename(episode: Doc) -> str:
    """A filename the recipient can file without renaming.

    Same shape as the vault note names in ``_write_episode_notes``, with the
    podcast in the name rather than in a parent directory: a download lands in
    one flat folder alongside everything else the reader saved that day.
    """
    show = slugify(str(episode.get("podcast_slug") or "podcast"))
    published = (episode.get("published_at") or "")[:10] or "undated"
    title = slugify(str(episode.get("title") or ""))
    return f"{show}-{published}-{title}.md"
