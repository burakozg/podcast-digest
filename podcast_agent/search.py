"""Full-text search over summaries and transcripts (roadmap B1).

A SQLite FTS5 index beside the database rather than a query against it. CouchDB
has no built-in full-text search — Mango's `$regex` is a full scan that no index
can serve, and it cannot reach a gzipped attachment at all, which is where every
transcript lives. The alternatives the roadmap left open were couchdb-lucene and
a SQLite sidecar; the first is a second service and a JVM, the second is in the
standard library.

The index is a **cache, not a record**. Every row is derived from a document
that remains the truth, the file lives in `work_dir` beside the audio scratch
space, and deleting it costs one rebuild. Nothing in the pipeline reads from it
and no decision depends on it, so it going stale degrades a search box and
nothing else.

Kept deliberately dumb: no ranking beyond FTS5's own bm25, no stemming
configuration, no synonyms. The corpus is thousands of episodes, not millions,
and a query that returns the right episode third is not worth a dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from .config import Settings
from .db import TRANSCRIPT_ATTACHMENT, Doc, Store, load_transcript, typed_sort
from .logging_setup import get_logger

log = get_logger(__name__)

INDEX_FILENAME = "search.db"

#: Rebuild page size. Each page holds whole transcripts in memory briefly, so
#: this is a memory bound rather than a throughput knob.
BATCH = 100

#: Transcript characters indexed per episode. A three-hour episode is ~150k
#: characters and the tail is almost never what someone is searching for; the
#: cap keeps the index a fraction of the corpus rather than a copy of it.
MAX_TRANSCRIPT_CHARS = 120_000

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodes USING fts5(
    episode_id UNINDEXED,
    podcast_slug,
    podcast_name,
    title,
    summary,
    takeaways,
    entities,
    transcript,
    published_at UNINDEXED,
    tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS state (
    episode_id TEXT PRIMARY KEY,
    row_ref INTEGER NOT NULL,
    signature TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

#: Fields a query may be restricted to, mapped to their FTS5 column. Exposed as
#: a fixed set so a caller cannot inject a column name into the MATCH string.
FIELDS: dict[str, str] = {
    "title": "title",
    "summary": "summary",
    "takeaways": "takeaways",
    "entities": "entities",
    "transcript": "transcript",
}


class SearchUnavailable(Exception):
    """The index could not be opened or queried."""


def index_path(settings: Settings) -> Path:
    return settings.output.work_dir / INDEX_FILENAME


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def escape_query(text: str) -> str:
    """Turn user input into an FTS5 MATCH expression that cannot be malformed.

    FTS5's query syntax has its own operators, and an unbalanced quote or a bare
    `AND` raises rather than returning nothing — a search box must never 500
    because someone typed an apostrophe. Every term is quoted, which makes the
    whole query a literal phrase-AND: what a search box is expected to do.
    """
    terms = [t for t in text.replace('"', " ").split() if t]
    return " ".join(f'"{t}"' for t in terms)


def signature(episode: Doc) -> str:
    """A cheap fingerprint of everything this episode contributes to the index.

    The point is to decide whether a row needs rewriting *without* fetching the
    transcript, which is the only expensive part of indexing. Built from fields
    that already travel with the document: `transcript_at` moves when one is
    stored, `transcript_expired_at` when retention removes one, and the tier-1
    block changes whenever a re-score rewrites a summary.
    """
    tier1 = episode.get("tier1") or {}
    parts = [
        str(episode.get("title") or ""),
        str(episode.get("status") or ""),
        str(episode.get("podcast_name") or ""),
        str(episode.get("transcript_at") or ""),
        str(episode.get("transcript_expired_at") or ""),
        str(tier1.get("summary_md") or ""),
        str(tier1.get("why_it_matters") or ""),
        repr(tier1.get("key_takeaways") or []),
        repr(tier1.get("entities") or []),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


def _row_text(episode: Doc, transcript: str | None) -> tuple[Any, ...]:
    tier1 = episode.get("tier1") or {}
    takeaways = " \n".join(str(t) for t in (tier1.get("key_takeaways") or []))
    entities = ", ".join(str(e) for e in (tier1.get("entities") or []))
    return (
        episode["_id"],
        episode.get("podcast_slug") or "",
        episode.get("podcast_name") or "",
        episode.get("title") or "",
        str(tier1.get("summary_md") or "") + "\n" + str(tier1.get("why_it_matters") or ""),
        takeaways,
        entities,
        (transcript or "")[:MAX_TRANSCRIPT_CHARS],
        episode.get("published_at") or "",
    )


class SearchIndex:
    """Owns the sidecar file. All SQLite work runs off the event loop."""

    def __init__(self, settings: Settings, store: Store) -> None:
        self._settings = settings
        self._store = store
        self._path = index_path(settings)
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    # --- building -----------------------------------------------------------

    async def rebuild(self) -> dict[str, Any]:
        """Rebuild from scratch. Safe to run at any time; the source is CouchDB.

        The repair operation, not the routine one — :meth:`sync` keeps the index
        current. Use this when the index is missing, or when you would rather
        not reason about whether it drifted.
        """
        async with self._lock:
            rows: list[tuple[str, str, tuple[Any, ...]]] = []
            skip = 0
            while True:
                page = await self._store.find(
                    {"type": "episode"},
                    sort=typed_sort("published_at", "desc"),
                    limit=BATCH,
                    skip=skip,
                )
                if not page:
                    break
                for episode in page:
                    transcript = None
                    if TRANSCRIPT_ATTACHMENT in (episode.get("_attachments") or {}):
                        transcript = await load_transcript(self._store, episode["_id"])
                    rows.append(
                        (episode["_id"], signature(episode), _row_text(episode, transcript))
                    )
                skip += len(page)
                if len(page) < BATCH:
                    break

            written = await asyncio.to_thread(self._write_all, rows)
            log.info("search.rebuilt", episodes=written, path=str(self._path))
            return {"indexed": written, "path": str(self._path)}

    async def sync(self) -> dict[str, Any]:
        """Bring the index up to date without rebuilding it.

        This is what keeps the index honest between rebuilds, and it is the
        cheap operation: it pages the episode documents — which do not carry
        attachment bodies — compares each one's signature against what the index
        holds, and only fetches a transcript for an episode that actually
        changed. A quiet half-hour therefore costs one paged query and no
        attachment reads at all.

        Falls back to a full rebuild when there is no index yet, so a fresh
        install needs no separate first step.
        """
        if not self._path.exists():
            return await self.rebuild()

        async with self._lock:
            known = await asyncio.to_thread(self._known_signatures)
            added = changed = removed = 0
            seen: set[str] = set()
            skip = 0
            while True:
                page = await self._store.find(
                    {"type": "episode"},
                    sort=typed_sort("published_at", "desc"),
                    limit=BATCH,
                    skip=skip,
                )
                if not page:
                    break
                pending: list[tuple[str, str, tuple[Any, ...]]] = []
                for episode in page:
                    episode_id = episode["_id"]
                    seen.add(episode_id)
                    current = signature(episode)
                    if known.get(episode_id) == current:
                        continue
                    transcript = None
                    if episode.get("transcript_at") and not episode.get("transcript_expired_at"):
                        transcript = await load_transcript(self._store, episode_id)
                    pending.append((episode_id, current, _row_text(episode, transcript)))
                    if episode_id in known:
                        changed += 1
                    else:
                        added += 1
                if pending:
                    await asyncio.to_thread(self._upsert, pending)
                skip += len(page)
                if len(page) < BATCH:
                    break

            # Episodes the database no longer has. Nothing deletes them today,
            # but an index that can only grow is a bug waiting for the first
            # thing that does.
            gone = sorted(set(known) - seen)
            if gone:
                removed = await asyncio.to_thread(self._delete, gone)

            total = await asyncio.to_thread(self._count)
            log.info("search.synced", added=added, changed=changed, removed=removed, total=total)
            return {
                "added": added,
                "changed": changed,
                "removed": removed,
                "indexed": total,
            }

    def _known_signatures(self) -> dict[str, str]:
        with closing(_connect(self._path)) as conn:
            rows = conn.execute("SELECT episode_id, signature FROM state").fetchall()
        return {row["episode_id"]: row["signature"] for row in rows}

    def _upsert(self, pending: list[tuple[str, str, tuple[Any, ...]]]) -> None:
        with closing(_connect(self._path)) as conn:
            for episode_id, sig, row in pending:
                existing = conn.execute(
                    "SELECT row_ref FROM state WHERE episode_id = ?", (episode_id,)
                ).fetchone()
                if existing is not None:
                    # FTS5 rows are replaced by rowid, never edited in place.
                    conn.execute("DELETE FROM episodes WHERE rowid = ?", (existing["row_ref"],))
                cursor = conn.execute(
                    "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row
                )
                conn.execute(
                    "INSERT OR REPLACE INTO state (episode_id, row_ref, signature) "
                    "VALUES (?, ?, ?)",
                    (episode_id, cursor.lastrowid, sig),
                )
            conn.commit()

    def _delete(self, episode_ids: list[str]) -> int:
        with closing(_connect(self._path)) as conn:
            for episode_id in episode_ids:
                row = conn.execute(
                    "SELECT row_ref FROM state WHERE episode_id = ?", (episode_id,)
                ).fetchone()
                if row is not None:
                    conn.execute("DELETE FROM episodes WHERE rowid = ?", (row["row_ref"],))
                conn.execute("DELETE FROM state WHERE episode_id = ?", (episode_id,))
            conn.commit()
        return len(episode_ids)

    def _count(self) -> int:
        with closing(_connect(self._path)) as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM state").fetchone()
        return int(row["n"]) if row else 0

    def _write_all(self, rows: Iterable[tuple[str, str, tuple[Any, ...]]]) -> int:
        # Into a fresh file then swapped, so a rebuild that dies part-way leaves
        # the previous index in place rather than an empty one.
        temporary = self._path.with_suffix(".rebuilding")
        temporary.unlink(missing_ok=True)
        count = 0
        with closing(_connect(temporary)) as conn:
            conn.execute("DELETE FROM episodes")
            conn.execute("DELETE FROM state")
            for episode_id, sig, row in rows:
                cursor = conn.execute(
                    "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row
                )
                # Signatures are written here too, so the next sync compares
                # against a populated index instead of re-indexing everything.
                conn.execute(
                    "INSERT OR REPLACE INTO state (episode_id, row_ref, signature) "
                    "VALUES (?, ?, ?)",
                    (episode_id, cursor.lastrowid, sig),
                )
                count += 1
            conn.commit()
        temporary.replace(self._path)
        return count

    # --- querying -----------------------------------------------------------

    async def search(
        self, query: str, *, field: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        expression = escape_query(query)
        if not expression:
            return []
        if field is not None and field not in FIELDS:
            raise SearchUnavailable(f"unknown field {field!r}; expected one of {sorted(FIELDS)}")
        if field:
            expression = f"{FIELDS[field]} : ({expression})"
        return await asyncio.to_thread(self._search_sync, expression, limit)

    def _search_sync(self, expression: str, limit: int) -> list[dict[str, Any]]:
        if not self._path.exists():
            raise SearchUnavailable(
                "the search index has not been built yet — POST /api/v1/search/rebuild"
            )
        try:
            with closing(_connect(self._path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT episode_id, podcast_slug, podcast_name, title, published_at,
                           snippet(episodes, 4, '<<', '>>', ' … ', 24) AS summary_snippet,
                           snippet(episodes, 7, '<<', '>>', ' … ', 24) AS transcript_snippet,
                           bm25(episodes) AS rank
                    FROM episodes
                    WHERE episodes MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (expression, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError as exc:
            # A malformed MATCH is the only realistic one, and escape_query is
            # what should have prevented it. Reported rather than raised as a
            # 500: a search box must never look like a broken server.
            raise SearchUnavailable(f"the query could not be run: {exc}") from exc

    async def stats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"built": False, "episodes": 0, "bytes": 0}
        return {
            "built": True,
            "episodes": self._count(),
            "bytes": self._path.stat().st_size,
        }
