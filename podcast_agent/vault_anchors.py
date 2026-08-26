"""Markers the digest carries for the vault projection, and how to remove them.

The weekly digest is read by three things: Obsidian, the console, and the
narrator. Only the first wants links to per-episode notes, so the digest on disk
keeps its summaries and carries invisible anchors saying where each entry begins
and which part of it duplicates an episode note. The projection uses them
(:func:`~.vault.slim_digest`); everything else strips them.

Its own module because both sides need it and neither should import the other:
`digest.read` is the console's reader and `vault` is the projection.
"""

from __future__ import annotations

import re

#: Emitted by `digest.md.j2` after each episode heading.
EPISODE_ANCHOR = re.compile(r"[ \t]*<!-- ep:(?P<id>[^>]+?) -->\n?")

#: Wraps the part of an entry that the episode's own note also holds.
FULL_REGION_OPEN = re.compile(r"[ \t]*<!-- full -->\n?")
FULL_REGION_CLOSE = re.compile(r"[ \t]*<!-- /full -->\n?")


def strip_vault_anchors(markdown: str) -> str:
    """Remove every anchor, leaving the text untouched otherwise."""
    for pattern in (EPISODE_ANCHOR, FULL_REGION_OPEN, FULL_REGION_CLOSE):
        markdown = pattern.sub("", markdown)
    return markdown
