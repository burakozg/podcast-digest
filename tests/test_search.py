"""Full-text search over summaries and transcripts (roadmap B1).

A SQLite FTS5 sidecar rather than a query against CouchDB: Mango's `$regex` is
a full scan no index can serve, and it cannot reach a gzipped attachment at all,
which is where every transcript lives.

The index is a cache. The tests that matter most are the ones about it being
allowed to be absent, stale or thrown away without anything else noticing.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.db import MemoryStore, save_transcript
from podcast_agent.main import build_app
from podcast_agent.search import SearchIndex, SearchUnavailable, escape_query
from podcast_agent.state import EpisodeStatus

S = EpisodeStatus
KEY = {"X-API-Key": "test-admin-key"}


def episode(guid: str, *, title: str = "An episode", **tier1: Any):
    block = {"relevance_score": 8, "summary_md": "", "why_it_matters": "", **tier1}
    return make_episode(guid=guid, title=title, status=S.READY_FOR_DIGEST, tier1=block)


@pytest.fixture
def index(tmp_path, store: MemoryStore) -> SearchIndex:
    return SearchIndex(make_settings(tmp_path), store)


class TestQueryEscaping:
    """A search box must never 500 because someone typed an apostrophe."""

    def test_terms_become_quoted_phrases(self) -> None:
        assert escape_query("purview dspm") == '"purview" "dspm"'

    def test_fts_operators_are_neutralised(self) -> None:
        """Bare AND/OR/NEAR are FTS5 syntax; quoted they are just words."""
        assert escape_query("this AND that") == '"this" "AND" "that"'

    def test_an_unbalanced_quote_cannot_survive(self) -> None:
        assert '"' not in escape_query('broken "').replace('"', "", 4)
        assert escape_query('broken "') == '"broken"'

    def test_empty_input_is_empty_output(self) -> None:
        assert escape_query("   ") == ""

    async def test_an_empty_query_returns_nothing_rather_than_everything(
        self, index: SearchIndex
    ) -> None:
        assert await index.search("   ") == []


class TestIndexing:
    async def test_a_summary_is_findable(self, index: SearchIndex, store: MemoryStore) -> None:
        store.seed(episode("a", summary_md="A discussion of Purview DSPM rollouts."))
        await index.rebuild()
        results = await index.search("purview")
        assert [r["episode_id"] for r in results] == [store.docs_of_type("episode")[0]["_id"]]

    async def test_a_transcript_is_findable(self, index: SearchIndex, store: MemoryStore) -> None:
        """The whole reason this is not a Mango query."""
        doc = episode("a", summary_md="Nothing relevant here.")
        store.seed(doc)
        await save_transcript(store, doc["_id"], "They spent an hour on Modbus segmentation.")
        await index.rebuild()
        assert len(await index.search("modbus")) == 1

    async def test_takeaways_and_entities_are_indexed(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        store.seed(episode("a", key_takeaways=["Rotate the PSK"], entities=["CVE-2026-1234"]))
        await index.rebuild()
        assert len(await index.search("CVE-2026-1234")) == 1
        assert len(await index.search("psk")) == 1

    async def test_a_field_can_be_named(self, index: SearchIndex, store: MemoryStore) -> None:
        doc = episode("a", summary_md="nothing")
        store.seed(doc)
        await save_transcript(store, doc["_id"], "Modbus everywhere.")
        await index.rebuild()
        assert len(await index.search("modbus", field="transcript")) == 1
        assert len(await index.search("modbus", field="summary")) == 0

    async def test_an_unknown_field_is_refused(self, index: SearchIndex) -> None:
        """Otherwise a column name reaches the MATCH expression."""
        with pytest.raises(SearchUnavailable, match="unknown field"):
            await index.search("x", field="podcast_slug; DROP TABLE")

    async def test_results_carry_a_snippet(self, index: SearchIndex, store: MemoryStore) -> None:
        store.seed(episode("a", summary_md="A long discussion of Purview DSPM and its limits."))
        await index.rebuild()
        assert "Purview" in (await index.search("purview"))[0]["summary_snippet"]

    async def test_rebuilding_reflects_a_changed_summary(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        """A re-score rewrites summaries; the cache has to be able to catch up."""
        doc = episode("a", summary_md="Original wording about Kerberos.")
        store.seed(doc)
        await index.rebuild()
        assert len(await index.search("kerberos")) == 1

        doc["tier1"]["summary_md"] = "Rewritten, now about Modbus."
        store.seed(doc)
        await index.rebuild()
        assert len(await index.search("kerberos")) == 0
        assert len(await index.search("modbus")) == 1

    async def test_stats_report_the_size(self, index: SearchIndex, store: MemoryStore) -> None:
        store.seed(episode("a"), episode("b"))
        await index.rebuild()
        stats = await index.stats()
        assert stats["built"] is True
        assert stats["episodes"] == 2
        assert stats["bytes"] > 0


class TestItIsOnlyACache:
    async def test_an_absent_index_says_so_rather_than_failing(self, index: SearchIndex) -> None:
        with pytest.raises(SearchUnavailable, match="not been built"):
            await index.search("anything")

    async def test_stats_on_an_absent_index_are_not_an_error(self, index: SearchIndex) -> None:
        assert await index.stats() == {"built": False, "episodes": 0, "bytes": 0}

    async def test_deleting_it_costs_one_rebuild(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        store.seed(episode("a", summary_md="Purview again."))
        await index.rebuild()
        index.path.unlink()
        with pytest.raises(SearchUnavailable):
            await index.search("purview")
        await index.rebuild()
        assert len(await index.search("purview")) == 1

    async def test_it_lives_in_work_dir_not_beside_the_digests(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Derived scratch, never the output the user syncs to a vault."""
        settings = make_settings(tmp_path)
        built = SearchIndex(settings, store)
        assert built.path.parent == settings.output.work_dir
        assert settings.output.digest_dir not in built.path.parents

    async def test_a_failed_rebuild_leaves_the_previous_index(
        self, index: SearchIndex, store: MemoryStore, monkeypatch
    ) -> None:
        store.seed(episode("a", summary_md="Purview again."))
        await index.rebuild()

        def _boom(*_a: object, **_k: object) -> int:
            raise RuntimeError("disk full")

        monkeypatch.setattr(index, "_write_all", _boom)
        with pytest.raises(RuntimeError):
            await index.rebuild()
        # The old index is still queryable, which is the point of the swap.
        assert len(await index.search("purview")) == 1


class TestApi:
    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_search_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/search?q=x").status_code == 401

    def test_rebuild_then_search(self, tmp_path, store: MemoryStore) -> None:
        store.seed(episode("a", summary_md="All about Purview DSPM."))
        with self._client(tmp_path, store) as client:
            assert client.post("/api/v1/search/rebuild", headers=KEY).status_code == 200
            body = client.get("/api/v1/search?q=purview", headers=KEY).json()
        assert body["count"] == 1

    def test_searching_before_the_index_exists_is_409_not_500(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """It is a cache that may not exist yet; the fix is a rebuild."""
        with self._client(tmp_path, store) as client:
            response = client.get("/api/v1/search?q=purview", headers=KEY)
        assert response.status_code == 409
        assert "rebuild" in response.json()["detail"]

    def test_status_reports_whether_it_is_built(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/search/status", headers=KEY).json()["built"] is False
            client.post("/api/v1/search/rebuild", headers=KEY)
            assert client.get("/api/v1/search/status", headers=KEY).json()["built"] is True

    def test_a_hostile_query_is_handled(self, tmp_path, store: MemoryStore) -> None:
        store.seed(episode("a", summary_md="Purview."))
        with self._client(tmp_path, store) as client:
            client.post("/api/v1/search/rebuild", headers=KEY)
            for hostile in ['" OR 1=1 --', "NEAR(", "*", "a AND", '""""']:
                assert client.get(
                    "/api/v1/search", params={"q": hostile}, headers=KEY
                ).status_code in (200, 409)


class TestConsole:
    def _page(self) -> str:
        from pathlib import Path

        return (Path(__file__).parent.parent / "podcast_agent/api/static/episodes.html").read_text()

    def test_the_search_box_exists_with_a_rebuild_control(self) -> None:
        page = self._page()
        for element in ("sQuery", "sField", "sGo", "sRebuild", "sStatus"):
            assert f'id="{element}"' in page

    def test_snippets_are_escaped_before_markers_become_tags(self) -> None:
        """Otherwise transcript text is interpreted as HTML.

        The order is the whole safety argument: escape first, then turn the
        FTS5 `<<`/`>>` markers into tags. Reversed, an episode containing a
        script tag would render it.
        """
        page = self._page()
        marker = page.index("function mark(snippet)")
        body = page[marker : marker + 300]
        assert body.index("esc(snippet)") < body.index("&lt;&lt;")
        assert "<mark>" in body

    def test_it_says_the_index_is_separate_from_the_database(self) -> None:
        """A stale index looks like missing data unless the page says otherwise."""
        page = self._page()
        assert "re-syncs on a schedule" in page
        assert "may not be findable yet" in page


class TestStayingCurrent:
    """The gap that makes a search index quietly wrong.

    Rebuilding by hand indexes everything once. Everything summarised after
    that is invisible to search while the box keeps returning results — which
    is worse than an empty index, because it looks like it works.
    """

    async def test_a_new_episode_is_picked_up(self, index: SearchIndex, store: MemoryStore) -> None:
        store.seed(episode("a", summary_md="Purview."))
        await index.rebuild()
        store.seed(episode("b", summary_md="Modbus segmentation."))
        assert await index.search("modbus") == []

        result = await index.sync()
        assert result["added"] == 1
        assert len(await index.search("modbus")) == 1

    async def test_a_rewritten_summary_replaces_its_row(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        """A re-score rewrites summaries; the old text must not linger."""
        doc = episode("a", summary_md="Originally about Kerberos.")
        store.seed(doc)
        await index.rebuild()

        doc["tier1"]["summary_md"] = "Now about Modbus."
        store.seed(doc)
        result = await index.sync()
        assert result["changed"] == 1
        assert await index.search("kerberos") == []
        assert len(await index.search("modbus")) == 1

    async def test_a_late_transcript_is_indexed(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        """Transcripts routinely arrive long after the episode does."""
        doc = episode("a", summary_md="Nothing useful.")
        store.seed(doc)
        await index.rebuild()

        await save_transcript(store, doc["_id"], "An hour on Modbus.")
        doc["transcript_at"] = "2026-08-01T00:00:00+00:00"
        store.seed(doc)
        await index.sync()
        assert len(await index.search("modbus")) == 1

    async def test_an_expired_transcript_leaves_the_index(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        """Retention removing one must not leave it searchable forever."""
        doc = episode("a", summary_md="Nothing useful.")
        doc["transcript_at"] = "2026-01-01T00:00:00+00:00"
        store.seed(doc)
        await save_transcript(store, doc["_id"], "An hour on Modbus.")
        await index.rebuild()
        assert len(await index.search("modbus")) == 1

        doc["transcript_expired_at"] = "2026-08-01T00:00:00+00:00"
        store.seed(doc)
        await index.sync()
        assert await index.search("modbus") == []

    async def test_an_unchanged_corpus_costs_no_attachment_reads(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        """The whole reason sync is cheap enough to schedule.

        A quiet half-hour must not re-download every transcript in the archive.
        """
        doc = episode("a", summary_md="Purview.")
        doc["transcript_at"] = "2026-01-01T00:00:00+00:00"
        store.seed(doc)
        await save_transcript(store, doc["_id"], "Long transcript.")
        await index.rebuild()

        reads = 0
        original = store.get_attachment

        async def counting(*args: Any, **kwargs: Any) -> Any:
            nonlocal reads
            reads += 1
            return await original(*args, **kwargs)

        store.get_attachment = counting  # type: ignore[method-assign]
        result = await index.sync()
        assert reads == 0
        assert result == {"added": 0, "changed": 0, "removed": 0, "indexed": 1}

    async def test_a_deleted_episode_leaves_the_index(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        """Nothing deletes episodes today; an index that can only grow is a bug
        waiting for the first thing that does."""
        doc = episode("a", summary_md="Purview.")
        store.seed(doc)
        await index.rebuild()
        stored = await store.get(doc["_id"])
        assert stored is not None
        await store.delete(stored["_id"], stored["_rev"])

        result = await index.sync()
        assert result["removed"] == 1
        assert await index.search("purview") == []

    async def test_syncing_without_an_index_builds_one(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        """A fresh install needs no separate first step."""
        store.seed(episode("a", summary_md="Purview."))
        await index.sync()
        assert len(await index.search("purview")) == 1

    async def test_a_rebuild_leaves_the_next_sync_with_nothing_to_do(
        self, index: SearchIndex, store: MemoryStore
    ) -> None:
        """Signatures are written by the rebuild, not only by sync."""
        store.seed(episode("a", summary_md="Purview."), episode("b", summary_md="Modbus."))
        await index.rebuild()
        assert await index.sync() == {"added": 0, "changed": 0, "removed": 0, "indexed": 2}

    def test_the_signature_ignores_fields_that_are_not_indexed(self) -> None:
        """Otherwise every attempt counter bump re-indexes a transcript."""
        from podcast_agent.search import signature

        base = episode("a", summary_md="Purview.")
        noisy = {**base, "attempts": {"tier1": 7}, "_rev": "9-zzz"}
        assert signature(base) == signature(noisy)

    def test_the_signature_moves_when_indexed_content_does(self) -> None:
        from podcast_agent.search import signature

        base = episode("a", summary_md="Purview.")
        changed = {**base, "tier1": {**base["tier1"], "summary_md": "Modbus."}}
        assert signature(base) != signature(changed)


class TestScheduled:
    def test_the_sync_job_is_registered(self, tmp_path, store: MemoryStore) -> None:
        app = build_app(make_settings(tmp_path), store=store, llm=FakeLLM())
        with TestClient(app):
            assert "search_sync" in {j.id for j in app.state.scheduler.get_jobs()}

    def test_it_runs_offset_from_the_pipeline(self, tmp_path) -> None:
        """After a batch of summaries lands, not alongside it."""
        settings = make_settings(tmp_path)
        assert settings.scheduler.search_cron != settings.scheduler.pipeline_cron

    def test_sync_is_reachable_from_the_api(self, tmp_path, store: MemoryStore) -> None:
        store.seed(episode("a", summary_md="Purview."))
        with TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM())) as client:
            body = client.post("/api/v1/search/sync", headers=KEY).json()
        assert body["indexed"] == 1
