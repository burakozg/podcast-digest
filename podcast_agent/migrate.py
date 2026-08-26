"""One-shot data migrations, run at startup.

Each is idempotent and safe to run on every boot: they find documents that lack
something and add it, so a completed migration finds nothing and costs one
query. That is deliberate — a migration that must be run exactly once is a
migration that will one day be run zero times.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .backfill import BACKFILL_ORIGIN, ROUTINE_ORIGIN
from .db import Store
from .digest.read import split_frontmatter
from .entities import TOPIC_NAMES_DOC_ID, canonical, pin_note_names
from .logging_setup import get_logger
from .notes import ENTITIES_DIR
from .utils import parse_iso

log = get_logger(__name__)

#: Documents rewritten per pass. Small enough to bound memory on a large
#: database, large enough that a few thousand episodes take a handful of passes.
BATCH = 500


async def backfill_origins(store: Store) -> dict[str, int]:
    """Give every episode an explicit ``origin``.

    Episodes ingested before the field existed carry no ``origin`` at all, and
    CouchDB's Mango cannot match a missing field — not with ``$eq``, not with
    ``$ne``. Selecting routine episodes therefore needed an ``$or`` with
    ``$exists: false``, which is correct but unindexable, so every pipeline query
    scanned a range and filtered in memory.

    This runs before the scheduler starts. If it were to run alongside the
    pipeline, an episode not yet migrated would be invisible to the equality
    selector for as long as the migration took — briefly reintroducing exactly
    the bug the explicit field exists to prevent.
    """
    updated = 0
    while True:
        batch = await store.find({"type": "episode", "origin": {"$exists": False}}, limit=BATCH)
        if not batch:
            break
        for doc in batch:
            # An archive episode always had its origin set, so anything reached
            # here is routine by definition. Checked rather than assumed: a
            # mislabelled episode would leak into the weekly digest.
            doc["origin"] = BACKFILL_ORIGIN if doc.get("archive_month") else ROUTINE_ORIGIN
            await store.put(doc)
            updated += 1
        if len(batch) < BATCH:
            break

    remaining = await store.count({"type": "episode", "origin": {"$exists": False}})
    if updated or remaining:
        log.info("migrate.origins", updated=updated, remaining=remaining)
    if remaining:
        # Not fatal, but it means some episodes are invisible to the pipeline
        # until the next boot completes the migration. Worth saying loudly.
        log.error(
            "migrate.origins_incomplete",
            remaining=remaining,
            detail="these episodes will not be selected until the migration completes",
        )
    return {"updated": updated, "remaining": remaining}


def _generated_at(run: dict[str, object]) -> datetime:
    """A run's generation time, for ordering. Undated runs sort first."""
    return parse_iso(str(run.get("generated_at") or "")) or datetime.min.replace(tzinfo=UTC)


async def adopt_orphaned_digest_files(store: Store, digest_dir: Path) -> dict[str, int]:
    """Record digest files that exist on disk but in no document.

    The generator never overwrites: a second run for the same ISO week writes
    `-r2` beside the first. The document, though, is keyed by the week, so the
    second run used to replace the first — leaving the earlier file in the vault,
    referenced by nothing and invisible to the console.

    Period and generation time are read back from each file's own frontmatter.
    Episode ids cannot be recovered, so an adopted run reports none rather than
    guessing; the file itself is intact and is what the console renders.
    """
    adopted = 0
    for doc in await store.find({"type": "digest"}, limit=1_000):
        week = str(doc.get("_id", "")).split(":", 1)[-1]
        known = {run.get("file_path") for run in (doc.get("runs") or [])}
        known.add(doc.get("file_path"))

        year = week.split("-")[0]
        found = sorted((digest_dir / year).glob(f"podcast-digest-{week}*.md"))
        missing = [f for f in found if str(f.relative_to(digest_dir)) not in known]
        if not missing:
            continue

        runs = list(doc.get("runs") or [])
        if not runs and doc.get("file_path"):
            runs = [
                {
                    "file_path": doc["file_path"],
                    "period": doc.get("period") or {},
                    "episode_ids": doc.get("episode_ids") or [],
                    "stats": doc.get("stats") or {},
                    "generated_at": doc.get("generated_at"),
                }
            ]
        for path in missing:
            meta, _ = split_frontmatter(path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "file_path": str(path.relative_to(digest_dir)),
                    "period": {
                        "from": str(meta.get("period_from") or ""),
                        "to": str(meta.get("period_to") or ""),
                    },
                    "episode_ids": [],
                    "stats": {
                        "scanned": meta.get("episodes_scanned"),
                        "summarized": meta.get("episodes_summarized"),
                    },
                    "generated_at": str(meta.get("generated") or ""),
                    "adopted": True,
                }
            )
            adopted += 1
        # By generation time, which is what "run 1" and "run 2" mean.
        #
        # Two traps. Sorting by filename gets it backwards — "-r2.md" precedes
        # ".md" because "-" sorts before "." — and sorting the timestamps as
        # strings is wrong too, because a file's frontmatter records local time
        # with an offset while the database stores UTC, so "+02:00" and "Z"
        # spellings of the same instant do not compare.
        doc["runs"] = sorted(runs, key=_generated_at)
        await store.put(doc)

    if adopted:
        log.info("migrate.digest_files_adopted", adopted=adopted)
    return {"adopted": adopted}


async def anchor_backfill_walks(store: Store, months: int) -> dict[str, int]:
    """Give an in-flight archive walk the anchor its floor is measured from.

    The floor used to be recomputed from `now` on every run, so it moved forward
    each calendar month while the cursor moved backward. When they crossed, the
    month the cursor was sitting on was stepped over and never fetched — five
    shows lost 2025-07 that way, on the night of a rollover.

    The repair is to anchor each unfinished walk so its floor lands exactly on
    the month the cursor has reached: finish the month you are on, then stop.
    That resumes the walk without re-reading anything behind it, and without
    needing to know when the walk originally started.

    Only unfinished walks are touched. A completed one keeps its cursor below
    any floor, so the next run records an anchor for it in passing.
    """
    docs = await store.find({"type": "podcast"}, limit=BATCH)
    anchored = 0
    for doc in docs:
        cursor = doc.get("backfill_cursor")
        if not cursor or doc.get("backfill_anchor") or doc.get("backfill_complete"):
            continue
        # Walk the cursor forward by the window, so floor_from_anchor lands
        # back on it.
        anchor = cursor
        for _ in range(months):
            anchor = _next_month(anchor)
        doc["backfill_anchor"] = anchor
        await store.put(doc)
        anchored += 1
        log.warning(
            "migrate.backfill_anchored",
            podcast=doc.get("slug"),
            cursor=cursor,
            anchor=anchor,
            detail="this walk will resume at its cursor rather than being declared finished",
        )
    if anchored:
        log.info("migrate.backfill_anchors", anchored=anchored)
    return {"anchored": anchored}


def _next_month(key: str) -> str:
    year, month = (int(part) for part in key.split("-"))
    return f"{year + 1:04d}-01" if month == 12 else f"{year:04d}-{month + 1:02d}"


async def seed_topic_note_names(store: Store, digest_dir: Path) -> dict[str, int]:
    """Pin the filename of every topic note that already exists.

    :func:`~.entities.resolve_note_names` pins a name the first time it writes a
    note, which covers everything above the note threshold on the next run. It
    does not cover a note that exists on disk but whose entity has since dropped
    *below* the threshold: nothing writes it, so nothing pins it, and if that
    entity climbs back later it would be named afresh — beside the old file,
    which still holds the other writers' sections.

    So the existing directory is read once and taken as the record. The key is
    recovered from each note's own `title:`, through the same `canonical` the
    aggregation uses, so a note pins to the entity it actually describes.
    """
    directory = digest_dir / ENTITIES_DIR
    if not directory.is_dir():
        return {"examined": 0, "pinned": 0}

    found: dict[str, str] = {}
    examined = 0
    for path in sorted(directory.glob("*.md")):
        examined += 1
        try:
            front, _ = split_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        title = str(front.get("title") or "").strip()
        if not title:
            continue
        # setdefault: if two files somehow claim one entity, the first by name
        # wins and the other is left alone rather than silently retargeted.
        found.setdefault(canonical(title), path.stem)

    if not found:
        return {"examined": examined, "pinned": 0}

    before = len((await store.get(TOPIC_NAMES_DOC_ID) or {}).get("names") or {})
    after = len(await pin_note_names(store, found))
    pinned = after - before
    if pinned:
        log.info("migrate.topic_names_seeded", examined=examined, pinned=pinned)
    return {"examined": examined, "pinned": pinned}


async def run_all(
    store: Store, digest_dir: Path | None = None, backfill_months: int = 12
) -> dict[str, dict[str, int]]:
    """Every migration, in order. Called once during startup."""
    result = {"origins": await backfill_origins(store)}
    result["backfill_anchors"] = await anchor_backfill_walks(store, backfill_months)
    if digest_dir is not None:
        result["digest_files"] = await adopt_orphaned_digest_files(store, digest_dir)
        # Before anything writes a topic note: a name pinned from what is
        # already in the vault is a name nothing has to rename.
        result["topic_names"] = await seed_topic_note_names(store, digest_dir)
    return result
