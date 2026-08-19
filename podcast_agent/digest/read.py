"""Reading generated digests back off disk for the console.

The Markdown file is the artefact — it is what lands in Obsidian, and it is what
the console shows. Rendering from the file rather than rebuilding from the
database means the console cannot disagree with what was actually written, and a
digest edited by hand is displayed as edited.

The path comes from a database document, so it is input, not a constant: every
read is confined to the configured digest directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..logging_setup import get_logger
from ..sanitize import md_to_safe_html

log = get_logger(__name__)

#: A digest of a few dozen episodes runs to tens of kilobytes. The cap is for a
#: file that is not what we think it is, not for a digest that ran long.
MAX_DIGEST_BYTES = 4 * 1024 * 1024

_FENCE = "---"


class DigestUnreadable(Exception):
    """The digest file is missing, out of bounds, or not readable as text."""


def resolve_within(digest_dir: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``digest_dir``, refusing to escape it.

    ``file_path`` is stored on the digest document, and a document is data. An
    absolute path, or one containing ``..``, would otherwise let anything the
    service can read be served through the console.
    """
    base = digest_dir.expanduser().resolve()
    candidate = (base / relative).resolve()
    if candidate != base and base not in candidate.parents:
        raise DigestUnreadable(f"path escapes the digest directory: {relative!r}")
    return candidate


def digest_period_key(doc: dict[str, Any]) -> str:
    """``digest:2026-W31`` → ``2026-W31``."""
    return str(doc.get("_id", "")).split(":", 1)[-1]


def digest_runs(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every generation for this week, oldest first.

    Documents written before runs were recorded have none, so the top-level
    fields stand in for the single run they describe.
    """
    runs = list(doc.get("runs") or [])
    if runs:
        return runs
    return [
        {
            "file_path": doc.get("file_path"),
            "period": doc.get("period") or {},
            "episode_ids": doc.get("episode_ids") or [],
            "stats": doc.get("stats") or {},
            "generated_at": doc.get("generated_at"),
        }
    ]


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Separate the YAML frontmatter from the Markdown body.

    Obsidian reads the frontmatter as metadata; a Markdown renderer reads the
    opening ``---`` as a horizontal rule and dumps the rest as a paragraph. The
    console wants it as metadata too, so it comes off here.

    Unparseable frontmatter is not an error worth failing a page over — the body
    is what the reader came for, so it is returned with empty metadata.
    """
    if not text.startswith(_FENCE):
        return {}, text
    parts = text.split("\n" + _FENCE, 1)
    if len(parts) != 2:
        return {}, text
    raw = parts[0][len(_FENCE) :]
    body = parts[1].lstrip("-").lstrip("\n")
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        log.warning("digest.frontmatter_unparseable", error=str(exc))
        return {}, body
    return (loaded if isinstance(loaded, dict) else {}), body


def read_digest(digest_dir: Path, relative: str) -> dict[str, Any]:
    """Return one digest's frontmatter, Markdown body and rendered HTML.

    Raises :class:`DigestUnreadable` when the file is gone — which is not a bug
    but a state the console must report, since digests live in a directory the
    user owns and may move, prune or sync elsewhere.
    """
    path = resolve_within(digest_dir, relative)
    if not path.is_file():
        raise DigestUnreadable(f"no digest file at {relative!r}")
    if path.stat().st_size > MAX_DIGEST_BYTES:
        raise DigestUnreadable(f"digest file is implausibly large: {relative!r}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DigestUnreadable(f"could not read {relative!r}: {exc}") from exc

    frontmatter, body = split_frontmatter(text)
    return {
        "frontmatter": frontmatter,
        "markdown": body,
        "html": md_to_safe_html(body),
        "bytes": path.stat().st_size,
    }
