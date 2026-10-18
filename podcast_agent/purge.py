"""Deliberate removal of low-scoring episodes, with a tombstone so they stay gone.

Ingestion is create-if-absent keyed on the episode id, so deleting a document
outright lets the next feed poll or archive walk bring the episode straight back
and score it again. Each purged episode therefore leaves a small
``tombstone:<episode id>`` document, which both ingesters check.

The search index and the vault reconcile themselves from the episode documents
(``SearchIndex.sync`` drops what is gone, the episode-note writer retires notes
for episodes that no longer exist), so nothing here reaches into either.
"""

from __future__ import annotations

from typing import Any

from .db import NotFoundError, Store, typed_sort
from .logging_setup import get_logger
from .utils import iso_now, tombstone_doc_id

log = get_logger(__name__)

PAGE = 200
#: How many candidate ids a dry run lists back; the counts are always complete.
SAMPLE = 25


async def purge_low_scoring(
    store: Store,
    *,
    podcast_slug: str,
    max_score: int,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Delete a show's episodes scored ``max_score`` or lower, leaving tombstones.

    Only episodes that were actually scored are touched: one dropped at Tier-0
    has no score, so "scored 3 or lower" says nothing about it. Starred episodes
    are kept — a reader's explicit keep outranks the model's number.
    """
    candidates: list[dict[str, Any]] = []
    unscored = starred = scanned = 0
    skip = 0
    while True:
        page = await store.find(
            {"type": "episode", "podcast_slug": podcast_slug},
            sort=typed_sort("published_at", "desc"),
            limit=PAGE,
            skip=skip,
        )
        if not page:
            break
        for episode in page:
            scanned += 1
            score = (episode.get("tier1") or {}).get("relevance_score")
            if score is None:
                unscored += 1
            elif int(score) > max_score:
                continue
            elif episode.get("starred"):
                starred += 1
            else:
                candidates.append(episode)
        skip += len(page)
        if len(page) < PAGE:
            break

    by_score: dict[int, int] = {}
    for episode in candidates:
        score = int(episode["tier1"]["relevance_score"])
        by_score[score] = by_score.get(score, 0) + 1

    deleted = 0
    if not dry_run:
        for episode in candidates:
            # Tombstone first: a crash between the two steps leaves an episode
            # that is still there and merely marked, not one that can return.
            await store.create(
                {
                    "_id": tombstone_doc_id(episode["_id"]),
                    "type": "tombstone",
                    "podcast_slug": podcast_slug,
                    "guid": episode.get("guid"),
                    "title": episode.get("title"),
                    "published_at": episode.get("published_at"),
                    "score": int(episode["tier1"]["relevance_score"]),
                    "purged_at": iso_now(),
                }
            )
            try:
                await store.delete(episode["_id"], episode["_rev"])
                deleted += 1
            except NotFoundError:
                continue
        log.info("purge.done", podcast=podcast_slug, max_score=max_score, deleted=deleted)

    return {
        "podcast": podcast_slug,
        "max_score": max_score,
        "dry_run": dry_run,
        "scanned": scanned,
        "matched": len(candidates),
        "deleted": deleted,
        "by_score": dict(sorted(by_score.items())),
        "kept_starred": starred,
        "kept_unscored": unscored,
        "sample": [
            {"id": e["_id"], "title": e.get("title"), "published_at": e.get("published_at")}
            for e in candidates[:SAMPLE]
        ],
    }
