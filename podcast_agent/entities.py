"""Entity and trend tracking across the corpus (roadmap D2).

Tier-1 already extracts named things from every episode it summarises — CVEs,
threat actors, tools, frameworks, named operations — and until now nothing read
them back. One episode saying "Volt Typhoon" is a detail in that episode's
summary; six episodes across four shows saying it over five months is the shape
of a story, and no per-episode artefact can show that.

Aggregation only. Nothing here calls a model or writes to an episode: it reads
what Tier-1 already produced and counts it. That is what makes it cheap enough
to recompute on demand rather than maintain as yet another index.

The hard part is not counting, it is deciding what counts as the same thing.
See :func:`canonical`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Doc, Store, typed_sort
from .logging_setup import get_logger
from .sanitize import md_escape_inline, slugify
from .utils import iso, utcnow

log = get_logger(__name__)

#: Documents read per page while aggregating.
BATCH = 500

#: Mentions before an entity is worth a note of its own. One mention is a
#: detail in an episode summary, not a thread through the corpus, and a vault
#: with four thousand single-use notes is a worse graph than none.
DEFAULT_MIN_MENTIONS = 2

#: Entity strings longer than this are almost always a sentence fragment the
#: model returned by mistake, and they poison the index by never matching
#: anything else.
MAX_ENTITY_CHARS = 80

_CVE = re.compile(r"^cve[\s\-_]*(\d{4})[\s\-_]*(\d{4,7})$", re.IGNORECASE)

#: Leading noise words that change nothing about which thing is meant.
_LEADING = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)

#: Trailing corporate suffixes, so "Mandiant" and "Mandiant Inc." agree.
_TRAILING = re.compile(r"[\s,]+(inc|inc\.|llc|ltd|ltd\.|corp|corp\.|gmbh|plc)$", re.IGNORECASE)


def canonical(name: str) -> str:
    """The key two spellings of the same thing must share.

    Deliberately conservative. Over-merging is the worse error: it silently
    fuses two unrelated entities into one timeline that reads as evidence, and
    nothing downstream can tell. Under-merging leaves two rows a reader can see
    and interpret for themselves.

    So this normalises only what is unambiguous — case, whitespace, punctuation
    noise, an article, a corporate suffix — plus CVE identifiers, which have a
    canonical form and are written every possible way.
    """
    text = " ".join(str(name).split()).strip(" .,;:—-")
    if not text:
        return ""
    if match := _CVE.match(text):
        return f"cve-{match.group(1)}-{int(match.group(2)):04d}"
    text = _LEADING.sub("", text)
    text = _TRAILING.sub("", text)
    return text.casefold().strip(" .,;:—-")


def display_name(surfaces: dict[str, int]) -> str:
    """The spelling to show: the most common, ties broken by the longest.

    Length as the tiebreak because the longer form is usually the more
    informative one — "Volt Typhoon" over "Volt", "CVE-2026-1234" over "2026-1234".
    """
    return max(surfaces.items(), key=lambda item: (item[1], len(item[0])))[0]


@dataclass(slots=True)
class Entity:
    key: str
    surfaces: dict[str, int] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    shows: set[str] = field(default_factory=set)

    @property
    def mentions(self) -> int:
        return len(self.episodes)

    @property
    def name(self) -> str:
        return display_name(self.surfaces) if self.surfaces else self.key

    @property
    def first_seen(self) -> str:
        return min((e["published_at"] or "") for e in self.episodes) if self.episodes else ""

    @property
    def last_seen(self) -> str:
        return max((e["published_at"] or "") for e in self.episodes) if self.episodes else ""

    def as_dict(self, *, with_episodes: bool = False) -> dict[str, Any]:
        view: dict[str, Any] = {
            "key": self.key,
            "name": self.name,
            "mentions": self.mentions,
            "shows": sorted(self.shows),
            "show_count": len(self.shows),
            "first_seen": self.first_seen[:10],
            "last_seen": self.last_seen[:10],
        }
        if with_episodes:
            view["episodes"] = sorted(
                self.episodes, key=lambda e: e["published_at"] or "", reverse=True
            )
        return view


def _episode_ref(episode: Doc) -> dict[str, Any]:
    tier1 = episode.get("tier1") or {}
    return {
        "episode_id": episode["_id"],
        "podcast_slug": episode.get("podcast_slug") or "",
        "podcast_name": episode.get("podcast_name") or episode.get("podcast_slug") or "",
        "title": episode.get("title") or "(untitled)",
        "published_at": episode.get("published_at") or "",
        "score": tier1.get("relevance_score"),
        "digest_id": episode.get("digest_id"),
    }


async def aggregate(
    store: Store, *, since: str | None = None, limit_docs: int = 20_000
) -> dict[str, Entity]:
    """Every entity Tier-1 has named, keyed by :func:`canonical`.

    Reads only episodes that carry a Tier-1 block: entities come from that pass,
    and an episode triage rejected never had one.
    """
    selector: dict[str, Any] = {"type": "episode"}
    if since:
        selector["published_at"] = {"$gte": since}

    found: dict[str, Entity] = {}
    skip = 0
    while skip < limit_docs:
        page = await store.find(
            selector,
            sort=typed_sort("published_at", "desc"),
            limit=BATCH,
            skip=skip,
        )
        if not page:
            break
        for episode in page:
            tier1 = episode.get("tier1") or {}
            raw = tier1.get("entities") or []
            if not raw:
                continue
            ref = _episode_ref(episode)
            # Within one episode the same entity may be listed twice in
            # different spellings; it is still one mention.
            seen_here: set[str] = set()
            for surface in raw:
                text = str(surface).strip()
                if not text or len(text) > MAX_ENTITY_CHARS:
                    continue
                key = canonical(text)
                if not key or key in seen_here:
                    continue
                seen_here.add(key)
                entity = found.setdefault(key, Entity(key=key))
                entity.surfaces[text] = entity.surfaces.get(text, 0) + 1
                entity.episodes.append(ref)
                if ref["podcast_slug"]:
                    entity.shows.add(ref["podcast_name"])
        skip += len(page)
        if len(page) < BATCH:
            break

    log.info("entities.aggregated", entities=len(found), since=since)
    return found


def rank(entities: dict[str, Entity], *, min_mentions: int = DEFAULT_MIN_MENTIONS) -> list[Entity]:
    """Most-discussed first, then most widely discussed, then alphabetical.

    Mentions before shows: a thing six episodes covered matters more than one
    two shows mentioned once each. Shows break the tie because agreement across
    independent shows is the stronger signal of the two.
    """
    kept = [e for e in entities.values() if e.mentions >= min_mentions]
    kept.sort(key=lambda e: (-e.mentions, -len(e.shows), e.name.casefold()))
    return kept


def window_start(days: int | None) -> str | None:
    if not days:
        return None
    from datetime import timedelta

    return iso(utcnow() - timedelta(days=days))


# --- Obsidian notes ---------------------------------------------------------


def _note_body(entity: Entity, *, week_of: dict[str, str]) -> str:
    """One entity note. Wikilinks so the vault's graph view has edges to draw."""
    # Escaped for the heading, JSON-quoted for the frontmatter. These strings
    # are model output over an automatic transcript (§10.2): unquoted, a name
    # containing a bracket breaks the YAML, and unescaped it renders as a link
    # in the heading.
    safe = md_escape_inline(entity.name, max_chars=MAX_ENTITY_CHARS)
    lines = [
        "---",
        "type: podcast-entity",
        f"entity: {json.dumps(entity.name)}",
        f"mentions: {entity.mentions}",
        f"shows: {len(entity.shows)}",
        f"first_seen: {entity.first_seen[:10]}",
        f"last_seen: {entity.last_seen[:10]}",
        "tags: [podcast-entity, cybersecurity]",
        "---",
        "",
        f"# {safe}",
        "",
        f"*{entity.mentions} episode{'s' if entity.mentions != 1 else ''} "
        f"across {len(entity.shows)} show{'s' if len(entity.shows) != 1 else ''} · "
        f"{entity.first_seen[:10]} → {entity.last_seen[:10]}*",
        "",
        "## Mentioned in",
        "",
    ]
    for ref in sorted(entity.episodes, key=lambda e: e["published_at"] or "", reverse=True):
        date = (ref["published_at"] or "")[:10]
        score = f" `{ref['score']}/10`" if ref.get("score") is not None else ""
        title = md_escape_inline(ref["title"], max_chars=160)
        show = md_escape_inline(ref["podcast_name"], max_chars=80)
        digest = week_of.get(str(ref.get("digest_id") or ""))
        where = f" — [[podcast-digest-{digest}]]" if digest else ""
        lines.append(f"- {date} · **{show}** — {title}{score}{where}")
    lines.append("")
    return "\n".join(lines)


def write_entity_notes(
    settings: Settings,
    entities: list[Entity],
    *,
    week_of: dict[str, str] | None = None,
) -> list[str]:
    """Write one note per entity under ``entities/`` in the digest directory.

    Rewritten wholesale each time rather than appended to: the note is a view of
    the corpus, and a stale line in it is worse than a rebuilt file, because the
    reader cannot tell which lines are current.
    """
    directory = settings.output.digest_dir / "entities"
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for entity in entities:
        name = slugify(entity.name) or entity.key
        path = directory / f"{name}.md"
        path.write_text(_note_body(entity, week_of=week_of or {}), encoding="utf-8")
        written.append(str(path.relative_to(settings.output.digest_dir)))
    log.info("entities.notes_written", count=len(written), directory=str(directory))
    return written


async def digest_weeks(store: Store) -> dict[str, str]:
    """`digest_id` → week key, so an entity note can link the week it appeared in."""
    weeks: dict[str, str] = {}
    for doc in await store.find({"type": "digest"}, limit=500):
        digest_id = str(doc.get("_id") or "")
        if digest_id.startswith("digest:"):
            weeks[digest_id] = digest_id.split(":", 1)[1]
    return weeks


def timeline(entity: Entity) -> list[dict[str, Any]]:
    """Mentions per month, oldest first — the shape of the story."""
    per_month: dict[str, int] = defaultdict(int)
    for ref in entity.episodes:
        month = (ref["published_at"] or "")[:7]
        if month:
            per_month[month] += 1
    return [{"month": m, "mentions": per_month[m]} for m in sorted(per_month)]


def note_path(settings: Settings, entity: Entity) -> Path:
    return settings.output.digest_dir / "entities" / f"{slugify(entity.name) or entity.key}.md"
