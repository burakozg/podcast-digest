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
from .db import ConflictError, Doc, NotFoundError, Store, typed_sort, update_doc
from .logging_setup import get_logger
from .notes import ENTITIES_DIR, KEY_PREFIX, wrap
from .sanitize import md_escape_inline, slugify
from .state import OURS_ONLY
from .utils import iso, iso_now, utcnow

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
    # OURS_ONLY: an imported episode's entities belong to the application
    # that summarised it, which writes its own topic pages. Counting them
    # here would add lines to shared `99 topics/` notes for content this
    # app does not own, and move the min_mentions threshold that decides
    # whether a topic note exists at all.
    selector: dict[str, Any] = {"type": "episode", **OURS_ONLY}
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


#: Characters that would break out of `[[note|Label]]` syntax — a pipe ends the
#: alias, brackets end the link. Titles are model-adjacent feed data (§10.2), and
#: podcast titles really do contain all three ("VCISO Tradecraft | Carlota Sage").
#: Substituted rather than treated as a reason to skip the link: 18 episodes lost
#: theirs to a bracket before this existed, which is a worse trade than a
#: character that reads slightly differently.
_ALIAS_SAFE = str.maketrans({"|": "-", "[": "(", "]": ")"})


def _alias(title: str) -> str:
    r"""A wikilink label. Markdown escaping is undone first: inside `[[…|…]]`
    Obsidian renders the alias literally, so a `\|` would show as a backslash."""
    return title.replace("\\", "").translate(_ALIAS_SAFE).strip()


def _note_body(
    entity: Entity, *, week_of: dict[str, str], note_of: dict[str, str] | None = None
) -> str:
    """This writer's contribution to one topic note.

    A complete note as it would be created fresh — frontmatter, title, and a
    single region marked as ours. When the note already exists in the vault only
    the marked region and the prefixed frontmatter keys are taken from this; see
    :mod:`.notes`, which is the contract a second application writing to the same
    file has to follow.
    """
    # Escaped for the heading, JSON-quoted for the frontmatter. These strings
    # are model output over an automatic transcript (§10.2): unquoted, a name
    # containing a bracket breaks the YAML, and unescaped it renders as a link
    # in the heading.
    safe = md_escape_inline(entity.name, max_chars=MAX_ENTITY_CHARS)
    front = [
        "type: topic",
        f"title: {json.dumps(entity.name)}",
        "tags: [topic]",
        f"{KEY_PREFIX}mentions: {entity.mentions}",
        f"{KEY_PREFIX}shows: {len(entity.shows)}",
        f"{KEY_PREFIX}first_seen: {entity.first_seen[:10]}",
        f"{KEY_PREFIX}last_seen: {entity.last_seen[:10]}",
    ]
    section = [
        "## From podcasts",
        "",
        f"*{entity.mentions} episode{'s' if entity.mentions != 1 else ''} "
        f"across {len(entity.shows)} show{'s' if len(entity.shows) != 1 else ''} · "
        f"{entity.first_seen[:10]} → {entity.last_seen[:10]}*",
        "",
    ]
    notes = note_of or {}
    for ref in sorted(entity.episodes, key=lambda e: e["published_at"] or "", reverse=True):
        date = (ref["published_at"] or "")[:10]
        score = f" `{ref['score']}/10`" if ref.get("score") is not None else ""
        title = md_escape_inline(ref["title"], max_chars=160)
        show = md_escape_inline(ref["podcast_name"], max_chars=80)
        # The episode's own note is what the line is *about*, so that is what the
        # title links to — the same shape security-digest and the clippings
        # importer use. Plain text when no note exists, never a dangling link.
        note = notes.get(str(ref.get("episode_id") or ""))
        subject = f"[[{note}|{_alias(title)}]]" if note else title
        # The week stays as a second, smaller link: it is where the episode
        # shipped, which is a different question from what the episode said.
        digest = week_of.get(str(ref.get("digest_id") or ""))
        where = f" — [[podcast-digest-{digest}]]" if digest else ""
        section.append(f"- {date} · **{show}** — {subject}{score}{where}")

    return (
        "---\n" + "\n".join(front) + "\n---\n\n" + f"# {safe}\n\n" + wrap("\n".join(section)) + "\n"
    )


#: Where the chosen filename of each topic note is remembered.
#:
#: A note's filename is part of the contract, not a rendering detail: links
#: resolve by filename, and a shared note's other writers — plus the reader's own
#: prose — live in the file at that path. So it is chosen once and pinned here,
#: never recomputed. See :func:`resolve_note_names`.
TOPIC_NAMES_DOC_ID = "control:topic_names"

#: The same idea one level down: an episode note's filename. Publishers edit
#: titles, so a name derived from one moves exactly as a topic name does — see
#: `homelab/TOPIC-NOTE-NAMING.md`. Keyed by `episode_id`, which never changes.
EPISODE_NAMES_DOC_ID = "control:episode_names"


async def pin_note_names(
    store: Store, proposed: dict[str, str], *, doc_id: str = TOPIC_NAMES_DOC_ID
) -> dict[str, str]:
    """Record a filename for each key that does not have one. Returns them all.

    Gap-filling only, never reassignment: a name pinned by an earlier run — or by
    another process racing this one — always wins, so two runs cannot rename a
    note between them. The document is created on first use, and a lost creation
    race is retried rather than overwritten.
    """
    # Captured from the mutator rather than read off the write: `store.put`
    # answers with CouchDB's `{id, rev}` ack, not the stored document, so
    # `update_doc`'s return value carries no fields to read back.
    merged: dict[str, str] = {}
    added = 0

    def _fill(doc: Doc) -> None:
        nonlocal added, merged
        doc.setdefault("type", "control")
        doc.setdefault("key", doc_id.split(":", 1)[-1])
        names = dict(doc.get("names") or {})
        before = len(names)
        for key, name in proposed.items():
            names.setdefault(key, name)
        added = len(names) - before
        doc["names"] = names
        doc["updated_at"] = iso_now()
        merged = names

    try:
        await update_doc(store, doc_id, _fill)
    except NotFoundError:
        seed: Doc = {"_id": doc_id}
        _fill(seed)
        try:
            await store.put(seed)
        except ConflictError:
            # Another process created it first; fold ours into theirs.
            await update_doc(store, doc_id, _fill)

    if added:
        log.info("entities.notes_named", doc=doc_id, newly_named=added)
    return merged


async def resolve_note_names(store: Store, entities: list[Entity]) -> dict[str, str]:
    """``entity key -> note filename``, choosing a name only for the unnamed.

    ``display_name`` is a *moving* value: it returns the most common surface, and
    surfaces accumulate as the corpus grows, so the winner changes. Two spellings
    that ``canonical`` folds but ``slugify`` does not — "Fortinet" and "Fortinet
    Inc." — will eventually swap places, and the note would be written to a new
    path with nothing deleting the old one.

    That is not merely untidy. ``99 topics/`` is section-owned: the file at the
    old path still holds every *other* writer's section and whatever the reader
    wrote, and none of it migrates. Every existing wikilink points at the orphan
    while new ones point at a note containing only our own section, so the graph
    shows two nodes where there is one thing.

    Pinning the first choice costs one document read per run and makes the
    filename stable for as long as the entity exists. The heading and ``title:``
    still follow the current display name, so the note reads correctly as
    spellings settle — it is only the filename, which links depend on, that is
    frozen.
    """
    names = await pin_note_names(
        store, {entity.key: slugify(entity.name) or entity.key for entity in entities}
    )
    return {entity.key: names.get(entity.key) or entity.key for entity in entities}


async def write_entity_notes(
    store: Store,
    settings: Settings,
    entities: list[Entity],
    *,
    week_of: dict[str, str] | None = None,
    note_of: dict[str, str] | None = None,
) -> list[str]:
    """Write one note per entity under ``entities/`` in the digest directory.

    Our *section* of each note is rewritten wholesale rather than appended to:
    it is a view of the corpus, and a stale line in it is worse than a rebuilt
    one because the reader cannot tell which lines are current. The filename is
    the opposite — chosen once and never recomputed (:func:`resolve_note_names`).
    """
    directory = settings.output.digest_dir / ENTITIES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    names = await resolve_note_names(store, entities)
    written: list[str] = []
    for entity in entities:
        path = directory / f"{names[entity.key]}.md"
        path.write_text(
            _note_body(entity, week_of=week_of or {}, note_of=note_of), encoding="utf-8"
        )
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


def note_path(settings: Settings, name: str) -> Path:
    """Path of a topic note, given the *pinned* name from :func:`resolve_note_names`.

    Takes a name rather than an Entity on purpose. It used to derive one from
    `entity.name`, which is exactly the recomputation that moves a note's
    filename out from under every link pointing at it — so the derivation now
    happens in one place, once per entity, and this only joins the path.
    """
    return settings.output.digest_dir / ENTITIES_DIR / f"{name}.md"
