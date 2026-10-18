"""Purging low-scoring episodes, and keeping them gone."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx
from helpers import make_settings
from test_ingest import FEED, FEED_URL, run_ingest

from podcast_agent.db import MemoryStore
from podcast_agent.purge import purge_low_scoring
from podcast_agent.utils import tombstone_doc_id


async def _episode(
    store: MemoryStore, n: int, score: int | None, *, slug: str = "wired", **extra: object
) -> str:
    doc_id = f"episode:{slug}{n}"
    await store.create(
        {
            "_id": doc_id,
            "type": "episode",
            "podcast_slug": slug,
            "title": f"Ep {n}",
            "published_at": f"2026-01-{n:02d}T00:00:00Z",
            "tier1": None if score is None else {"relevance_score": score},
            **extra,
        }
    )
    return doc_id


class TestPurge:
    async def test_dry_run_counts_and_deletes_nothing(self, store: MemoryStore) -> None:
        await _episode(store, 1, 2)
        await _episode(store, 2, 3)
        await _episode(store, 3, 4)
        result = await purge_low_scoring(store, podcast_slug="wired", max_score=3)
        assert result["matched"] == 2
        assert result["deleted"] == 0
        assert result["by_score"] == {2: 1, 3: 1}
        assert len(store.docs_of_type("episode")) == 3

    async def test_deletes_at_or_below_threshold_and_leaves_tombstones(
        self, store: MemoryStore
    ) -> None:
        low = await _episode(store, 1, 3)
        high = await _episode(store, 2, 4)
        await purge_low_scoring(store, podcast_slug="wired", max_score=3, dry_run=False)
        assert await store.get(low) is None
        assert await store.get(high) is not None
        assert await store.get(tombstone_doc_id(low)) is not None

    async def test_spares_starred_unscored_and_other_shows(self, store: MemoryStore) -> None:
        starred = await _episode(store, 1, 1, starred=True)
        unscored = await _episode(store, 2, None)
        other = await _episode(store, 3, 1, slug="other")
        result = await purge_low_scoring(store, podcast_slug="wired", max_score=3, dry_run=False)
        assert result["deleted"] == 0
        assert result["kept_starred"] == 1
        assert result["kept_unscored"] == 1
        for doc_id in (starred, unscored, other):
            assert await store.get(doc_id) is not None

    @respx.mock
    async def test_a_purged_episode_is_not_ingested_again(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(
            tmp_path,
            podcasts=[
                {"slug": "test-show", "name": "Test Show", "feed_url": FEED_URL, "priority": "med"}
            ],
            pipeline={"initial_lookback_days": 0},
        )
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(settings, store)
        episodes = store.docs_of_type("episode")
        assert len(episodes) == 2
        for e in episodes:
            e["tier1"] = {"relevance_score": 1}
            await store.put(e)

        await purge_low_scoring(store, podcast_slug="test-show", max_score=3, dry_run=False)
        assert store.docs_of_type("episode") == []

        await run_ingest(settings, store)
        assert store.docs_of_type("episode") == []
