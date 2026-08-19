"""API tests (§9): auth, endpoints, and the LAN-only surface."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.api import auth
from podcast_agent.api.auth import MAX_FAILURES, WINDOW_S
from podcast_agent.config import load_settings
from podcast_agent.db import MemoryStore
from podcast_agent.main import build_app
from podcast_agent.state import BACKFILL_ORIGIN, EpisodeStatus
from podcast_agent.utils import (
    digest_doc_id,
    episode_doc_id,
    iso,
    iso_now,
    podcast_doc_id,
)

S = EpisodeStatus
KEY = {"X-API-Key": "test-admin-key"}


@pytest.fixture
def client(tmp_path, store: MemoryStore) -> Iterator[TestClient]:
    settings = make_settings(tmp_path)
    app = build_app(settings, store=store, llm=FakeLLM())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seeded_store(store: MemoryStore) -> MemoryStore:
    published = datetime(2026, 7, 28, tzinfo=UTC)
    store.seed(
        make_episode(guid="a", status=S.NEW, published_at=published),
        make_episode(
            guid="b",
            status=S.READY_FOR_DIGEST,
            published_at=published,
            tier1={"relevance_score": 8, "summary_basis": "transcript"},
        ),
        make_episode(
            guid="c",
            status=S.ERROR,
            published_at=published,
            last_error={"stage": "tier1", "message": "boom", "traceback": "secret trace"},
        ),
        {
            "_id": "llmcall:1",
            "type": "llm_call",
            "tier": "tier0",
            "provider": "ollama",
            "model": "test-small",
            "input_tokens": 100,
            "output_tokens": 20,
            "latency_ms": 500,
            "cost_usd": 0.0,
            "fallback_used": False,
            "ts": iso_now(),
        },
        {
            "_id": "llmcall:2",
            "type": "llm_call",
            "tier": "tier1",
            "provider": "openrouter",
            "model": "remote-big",
            "input_tokens": 8000,
            "output_tokens": 600,
            "latency_ms": 4000,
            "cost_usd": 0.02,
            "fallback_used": True,
            "ts": iso_now(),
        },
    )
    return store


class TestAuth:
    def test_healthz_needs_no_key(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code in (200, 503)

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/status",
            "/api/v1/episodes",
            "/api/v1/telemetry/costs",
            "/api/v1/digests",
            "/docs",
            "/openapi.json",
        ],
    )
    def test_endpoints_require_a_key(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code == 401

    def test_wrong_key_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/v1/status", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_correct_key_is_accepted(self, client: TestClient) -> None:
        assert client.get("/api/v1/status", headers=KEY).status_code == 200

    def test_docs_are_behind_the_key(self, client: TestClient) -> None:
        """§9: /docs is enabled but not public."""
        assert client.get("/docs").status_code == 401
        assert client.get("/docs", headers=KEY).status_code == 200
        assert client.get("/openapi.json", headers=KEY).status_code == 200

    def test_missing_configured_key_fails_closed(self, tmp_path, store: MemoryStore) -> None:
        """An unset admin key must never mean 'open to everyone'."""
        settings = make_settings(tmp_path, admin_api_key=None)
        app = build_app(settings, store=store, llm=FakeLLM())
        with TestClient(app) as unkeyed:
            response = unkeyed.get("/api/v1/status", headers=KEY)
            assert response.status_code == 503
            assert "not configured" in response.json()["detail"]

    def test_cors_is_not_enabled(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/status", headers={**KEY, "Origin": "https://evil.example.com"}
        )
        assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


class TestHealth:
    def test_reports_dependency_state(self, client: TestClient) -> None:
        payload = client.get("/healthz").json()
        body = payload.get("detail", payload)
        assert body["couchdb"] == "ok"
        assert body["scheduler"] in ("running", "stopped")

    def test_unreachable_store_gives_503(self, tmp_path, store: MemoryStore) -> None:
        async def dead_ping() -> bool:
            return False

        store.ping = dead_ping  # type: ignore[method-assign]
        app = build_app(make_settings(tmp_path), store=store, llm=FakeLLM())
        with TestClient(app) as sick:
            response = sick.get("/healthz")
            assert response.status_code == 503
            assert response.json()["detail"]["couchdb"] == "unreachable"


class TestStatus:
    def test_counts_and_config_are_reported(self, tmp_path, seeded_store: MemoryStore) -> None:
        app = build_app(make_settings(tmp_path), store=seeded_store, llm=FakeLLM())
        with TestClient(app) as client:
            body = client.get("/api/v1/status", headers=KEY).json()
        assert body["episode_counts"]["NEW"] == 1
        assert body["episode_counts"]["READY_FOR_DIGEST"] == 1
        assert body["queue_depths"]["triage"] == 1
        assert body["queue_depths"]["awaiting_digest"] == 1
        assert body["config"]["digest_threshold"] == 5
        assert body["config"]["tiers"]["tier0"] == ["ollama_chat/test-small"]
        assert body["jobs_running"] == {
            "ingest": False,
            "pipeline": False,
            "digest": False,
            "rescore": False,
            "backfill": False,
        }

    def test_feed_health_is_surfaced(self, tmp_path, store: MemoryStore) -> None:
        """§10.3: an open circuit breaker must be visible in /status."""
        store.seed(
            {
                "_id": "podcast:test-show",
                "type": "podcast",
                "slug": "test-show",
                # A real document always carries the feed URL; without it startup
                # seeding sees a changed feed and resets the failure counters.
                "feed_url": "https://example.com/feed.xml",
                "consecutive_failures": 7,
                "last_error": "503 from origin",
                "last_polled_at": iso_now(),
            }
        )
        app = build_app(make_settings(tmp_path), store=store, llm=FakeLLM())
        with TestClient(app) as client:
            feeds = client.get("/api/v1/status", headers=KEY).json()["feeds"]
        broken = next(f for f in feeds if f["slug"] == "test-show")
        assert broken["circuit_open"] is True
        assert broken["last_error"] == "503 from origin"

    def test_secrets_never_appear(self, tmp_path, seeded_store: MemoryStore) -> None:
        app = build_app(make_settings(tmp_path), store=seeded_store, llm=FakeLLM())
        with TestClient(app) as client:
            raw = client.get("/api/v1/status", headers=KEY).text
        assert "test-admin-key" not in raw


class TestEpisodes:
    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_list_and_filter(self, tmp_path, seeded_store: MemoryStore) -> None:
        with self._client(tmp_path, seeded_store) as client:
            all_eps = client.get("/api/v1/episodes", headers=KEY).json()
            assert all_eps["count"] == 3
            filtered = client.get("/api/v1/episodes?status=NEW", headers=KEY).json()
            assert filtered["count"] == 1
            by_show = client.get("/api/v1/episodes?podcast=test-show", headers=KEY).json()
            assert by_show["count"] == 3
            empty = client.get("/api/v1/episodes?podcast=nope", headers=KEY).json()
            assert empty["count"] == 0

    def test_unknown_status_filter_is_a_400(self, tmp_path, seeded_store: MemoryStore) -> None:
        with self._client(tmp_path, seeded_store) as client:
            assert client.get("/api/v1/episodes?status=BOGUS", headers=KEY).status_code == 400

    def test_listing_omits_tracebacks(self, tmp_path, seeded_store: MemoryStore) -> None:
        with self._client(tmp_path, seeded_store) as client:
            body = client.get("/api/v1/episodes", headers=KEY).text
        assert "secret trace" not in body

    def test_detail_includes_traceback(self, tmp_path, seeded_store: MemoryStore) -> None:
        episode_id = make_episode(guid="c")["_id"]
        with self._client(tmp_path, seeded_store) as client:
            body = client.get(f"/api/v1/episodes/{episode_id}", headers=KEY).json()
        assert body["last_error"]["traceback"] == "secret trace"

    def test_bare_hash_id_is_accepted(self, tmp_path, seeded_store: MemoryStore) -> None:
        bare = make_episode(guid="c")["_id"].split(":", 1)[1]
        with self._client(tmp_path, seeded_store) as client:
            assert client.get(f"/api/v1/episodes/{bare}", headers=KEY).status_code == 200

    def test_missing_episode_is_a_404(self, tmp_path, seeded_store: MemoryStore) -> None:
        with self._client(tmp_path, seeded_store) as client:
            assert client.get("/api/v1/episodes/deadbeef", headers=KEY).status_code == 404

    def test_retry_clears_the_crash_counter_too(self, tmp_path, store: MemoryStore) -> None:
        """Retry must clear `transcript_crash`, or it is a no-op where it matters.

        That counter retires an episode that was merely in flight when the
        process died. Once the operator has fixed the cause, retry is the button
        they press — and leaving the counter set makes the stage give up again
        before touching the audio, while the API still answers 200.
        """
        episode = make_episode(guid="crashed", status=S.TRANSCRIPT_FAILED)
        episode["attempts"] = {"transcript": 3, "transcript_crash": 3}
        store.seed(episode)
        with self._client(tmp_path, store) as client:
            response = client.post(f"/api/v1/episodes/{episode['_id']}/retry", headers=KEY)
        assert response.status_code == 200
        doc = store._docs[episode["_id"]]
        assert doc["attempts"]["transcript"] == 0
        assert doc["attempts"]["transcript_crash"] == 0, (
            "a crash-retired episode would be retired again without an attempt"
        )

    def test_retry_resets_a_failed_episode(self, tmp_path, seeded_store: MemoryStore) -> None:
        episode_id = make_episode(guid="c")["_id"]
        with self._client(tmp_path, seeded_store) as client:
            body = client.post(f"/api/v1/episodes/{episode_id}/retry", headers=KEY).json()
        assert body["from"] == "ERROR"
        assert body["to"] == "AWAITING_TRANSCRIPT"

    def test_retry_rejects_a_healthy_episode(self, tmp_path, seeded_store: MemoryStore) -> None:
        episode_id = make_episode(guid="b")["_id"]  # READY_FOR_DIGEST
        with self._client(tmp_path, seeded_store) as client:
            response = client.post(f"/api/v1/episodes/{episode_id}/retry", headers=KEY)
        assert response.status_code == 409

    def test_escalate_a_dropped_episode(self, tmp_path, store: MemoryStore) -> None:
        """§9: owner override for something triage discarded."""
        store.seed(make_episode(guid="d", status=S.DROPPED, digest_id="digest:2026-W30"))
        episode_id = make_episode(guid="d")["_id"]
        with self._client(tmp_path, store) as client:
            body = client.post(f"/api/v1/episodes/{episode_id}/escalate", headers=KEY).json()
        assert body["to"] == "AWAITING_TRANSCRIPT"
        doc = store._docs[episode_id]
        assert doc["status"] == S.AWAITING_TRANSCRIPT.value
        # The digest claim is cleared so the re-summarised episode can reappear.
        assert doc["digest_id"] is None
        assert doc["forced_escalation"]["from"] == "DROPPED"

    def test_a_published_episode_can_be_re_opened_by_the_owner(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The archive publishes index-only entries; asking for a real summary
        afterwards is a normal thing to want, not a mistake to prevent.

        Safe because the file already written is never rewritten — the digest
        claim is cleared and a re-summarised episode lands in a new file.
        """
        store.seed(
            make_episode(guid="e", status=S.PUBLISHED, digest_id="archive:test-show:2026-06")
        )
        episode_id = make_episode(guid="e")["_id"]
        with self._client(tmp_path, store) as client:
            response = client.post(f"/api/v1/episodes/{episode_id}/escalate", headers=KEY)
        assert response.status_code == 200
        assert response.json()["to"] == S.AWAITING_TRANSCRIPT.value

        doc = next(iter(store.docs_of_type("episode")))
        assert doc["digest_id"] is None, "must be re-claimable"
        assert doc["forced_escalation"]["from"] == S.PUBLISHED.value


class TestRuns:
    def test_ingest_returns_immediately_by_default(self, client: TestClient) -> None:
        body = client.post("/api/v1/runs/ingest", headers=KEY).json()
        assert body["started"] is True
        assert body["waited"] is False

    def test_digest_runs_inline(self, client: TestClient) -> None:
        body = client.post("/api/v1/runs/digest?dry_run=true", headers=KEY).json()
        assert body["result"]["dry_run"] is True

    def test_digest_accepts_a_since_parameter(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/runs/digest?dry_run=true&since=2026-07-01T00:00:00Z", headers=KEY
        )
        assert response.status_code == 200

    def test_pipeline_can_be_awaited(self, client: TestClient) -> None:
        body = client.post("/api/v1/runs/pipeline?wait=true", headers=KEY).json()
        assert body["waited"] is True
        assert "triaged" in body["result"]

    def test_retention_runs_inline(self, client: TestClient) -> None:
        body = client.post("/api/v1/runs/retention", headers=KEY).json()
        assert "transcripts_deleted" in body["result"]


class TestTelemetry:
    def test_aggregates_by_dimension(self, tmp_path, seeded_store: MemoryStore) -> None:
        app = build_app(make_settings(tmp_path), store=seeded_store, llm=FakeLLM())
        with TestClient(app) as client:
            body = client.get("/api/v1/telemetry/costs", headers=KEY).json()
        assert body["totals"]["calls"] == 2
        assert body["totals"]["cost_usd"] == pytest.approx(0.02)
        assert body["totals"]["fallbacks"] == 1
        # Local-vs-cloud economics is the whole point of this endpoint (§6).
        assert body["by_provider"]["ollama"]["cost_usd"] == 0.0
        assert body["by_provider"]["openrouter"]["cost_usd"] == pytest.approx(0.02)
        assert body["by_tier"]["tier1"]["avg_latency_ms"] == 4000
        assert len(body["by_day"]) == 1

    def test_window_is_honoured(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            {
                "_id": "llmcall:old",
                "type": "llm_call",
                "provider": "ollama",
                "model": "m",
                "tier": "tier0",
                "cost_usd": 5.0,
                "ts": "2020-01-01T00:00:00+00:00",
            }
        )
        app = build_app(make_settings(tmp_path), store=store, llm=FakeLLM())
        with TestClient(app) as client:
            body = client.get("/api/v1/telemetry/costs?days=30", headers=KEY).json()
        assert body["totals"]["calls"] == 0


class TestDigestsListing:
    def test_lists_generated_digests(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            {
                "_id": "digest:2026-W31",
                "type": "digest",
                "period": {"from": "2026-07-24T00:00:00+00:00", "to": "2026-07-31T00:00:00+00:00"},
                "file_path": "2026/podcast-digest-2026-W31.md",
                "episode_ids": ["episode:x"],
                "stats": {"scanned": 1},
                "marking_complete": True,
                "generated_at": iso_now(),
            }
        )
        app = build_app(make_settings(tmp_path), store=store, llm=FakeLLM())
        with TestClient(app) as client:
            body = client.get("/api/v1/digests", headers=KEY).json()
        assert body["count"] == 1
        assert body["digests"][0]["episodes"] == 1
        assert body["digests"][0]["marking_complete"] is True


class TestStartup:
    def test_output_directories_are_created(self, tmp_path, store: MemoryStore) -> None:
        """A permissions problem should surface at boot, not at 06:00 on Friday."""
        settings = make_settings(tmp_path)
        with TestClient(build_app(settings, store=store, llm=FakeLLM())):
            assert settings.output.digest_dir.is_dir()
            assert (settings.output.work_dir / "audio").is_dir()

    def test_store_setup_runs(self, tmp_path, store: MemoryStore) -> None:
        with TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM())):
            assert store.setup_called is True

    def test_scheduler_starts_and_registers_jobs(self, tmp_path, store: MemoryStore) -> None:
        app = build_app(make_settings(tmp_path), store=store, llm=FakeLLM())
        with TestClient(app):
            scheduler = app.state.scheduler
            assert scheduler.running is True
            job_ids = {j.id for j in scheduler.get_jobs()}
            assert job_ids == {
                "ingest",
                "pipeline",
                "digest_weekly",
                "retention_cleanup",
                "backfill",
                # Without a scheduled sync the search index is only ever as
                # current as the last manual rebuild — which looks like it
                # works right up until it silently stops finding this week.
                "search_sync",
                # Reader marks into the vault, weekly. Scheduled rather than
                # offered as a button: a mirror nobody remembers to press is
                # one that is always out of date.
                "signals_export",
            }
            for job in scheduler.get_jobs():
                # §11: a slow run must never overlap the next firing.
                assert job.max_instances == 1
                assert job.coalesce is True


class TestAdminPortal:
    """Operations console (roadmap B2)."""

    def test_page_is_served_without_a_key(self, client: TestClient) -> None:
        """The shell carries no data; it asks for the key in the browser, which
        is the only way a page navigation can authenticate a header-keyed API."""
        response = client.get("/admin")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_page_contains_no_data_or_secrets(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert "test-admin-key" not in body
        # Nothing but the shell: values arrive over the authenticated API.
        assert "podcast_agent" not in body

    def test_page_is_self_contained(self, client: TestClient) -> None:
        """LAN-only service: no CDN fetches, no external anything."""
        body = client.get("/admin").text
        for marker in ("http://", "https://", "//cdn", 'src="//'):
            assert marker not in body, f"external reference {marker!r} in admin page"

    def test_page_is_not_indexable(self, client: TestClient) -> None:
        assert "noindex" in client.get("/admin").text

    def test_control_endpoints_need_the_key(self, client: TestClient) -> None:
        assert client.get("/api/v1/backfill/control").status_code == 401
        assert client.post("/api/v1/backfill/control?paused=true").status_code == 401

    def test_control_round_trip(self, client: TestClient) -> None:
        assert client.get("/api/v1/backfill/control", headers=KEY).json()["paused"] is True
        started = client.post(
            "/api/v1/backfill/control?paused=false&note=from+test", headers=KEY
        ).json()
        assert started["paused"] is False
        assert started["note"] == "from test"
        assert client.get("/api/v1/backfill/control", headers=KEY).json()["paused"] is False

    def test_status_reports_control_state(self, client: TestClient) -> None:
        body = client.get("/api/v1/status", headers=KEY).json()
        assert body["backfill"]["control"]["paused"] is True


class TestEpisodeBrowsing:
    """Answering 'does this episode have a summary, and can I redo it?'"""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    async def _seed(self, store: MemoryStore) -> None:
        from podcast_agent.db import save_transcript

        summarised = make_episode(
            guid="done",
            status=S.READY_FOR_DIGEST,
            tier1={
                "relevance_score": 8,
                "summary_basis": "description_only",
                "summary_md": "The actual summary text.",
                "key_takeaways": ["One thing"],
                "entities": ["Modbus"],
                "why_it_matters": "Relevant to OT.",
            },
        )
        store.seed(summarised, make_episode(guid="bare", status=S.NEW))
        await save_transcript(store, summarised["_id"], "a transcript")

    async def test_list_shows_summary_and_transcript_presence(
        self, tmp_path, store: MemoryStore
    ) -> None:
        await self._seed(store)
        with self._client(tmp_path, store) as client:
            episodes = client.get("/api/v1/episodes", headers=KEY).json()["episodes"]
        by_guid = {e["_id"]: e for e in episodes}
        done = by_guid[make_episode(guid="done")["_id"]]
        bare = by_guid[make_episode(guid="bare")["_id"]]
        assert done["has_summary"] is True
        assert done["has_transcript"] is True
        assert bare["has_summary"] is False
        assert bare["has_transcript"] is False

    async def test_list_still_omits_the_summary_text(self, tmp_path, store: MemoryStore) -> None:
        """The list is a browse view; bodies would bloat it for no benefit."""
        await self._seed(store)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes", headers=KEY).text
        assert "The actual summary text." not in body

    async def test_detail_returns_the_summary_text(self, tmp_path, store: MemoryStore) -> None:
        await self._seed(store)
        episode_id = make_episode(guid="done")["_id"]
        with self._client(tmp_path, store) as client:
            body = client.get(f"/api/v1/episodes/{episode_id}", headers=KEY).json()
        assert body["tier1_full"]["summary_md"] == "The actual summary text."
        assert body["tier1_full"]["key_takeaways"] == ["One thing"]


class TestOnDemandSummarize:
    def _client(self, tmp_path, store: MemoryStore, llm=None) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=llm or FakeLLM()))

    def test_requires_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.post("/api/v1/episodes/abc/summarize").status_code == 401

    def test_unknown_episode_is_404(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            response = client.post("/api/v1/episodes/deadbeef/summarize", headers=KEY)
        assert response.status_code == 404

    def test_summarises_without_asr_from_the_description(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """No transcript and no ASR still yields a verdict, honestly labelled."""
        store.seed(make_episode(guid="none", status=S.DROPPED))
        episode_id = make_episode(guid="none")["_id"]
        with self._client(tmp_path, store) as client:
            body = client.post(
                f"/api/v1/episodes/{episode_id}/summarize?allow_asr=false&wait=true",
                headers=KEY,
            ).json()
        assert body["waited"] is True
        assert body["result"]["summary_basis"] == "description_only"
        assert body["result"]["status"] in ("READY_FOR_DIGEST", "SCORED_LOW")

    def test_re_summarising_an_already_scored_episode_is_allowed(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The main use case: it was done description-only, now do it properly."""
        store.seed(
            make_episode(
                guid="redo",
                status=S.READY_FOR_DIGEST,
                tier1={"relevance_score": 4, "summary_basis": "description_only"},
            )
        )
        episode_id = make_episode(guid="redo")["_id"]
        with self._client(tmp_path, store) as client:
            response = client.post(
                f"/api/v1/episodes/{episode_id}/summarize?allow_asr=false&wait=true",
                headers=KEY,
            )
        assert response.status_code == 200

    def test_an_episode_listed_without_a_summary_can_still_be_summarised(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The grey zone and the archive both publish episodes as index entries.

        Refusing those left an episode the owner wants to read permanently
        unreachable: the button was disabled and no other path existed. There is
        nothing in the written file to contradict — it said the episode was not
        summarised, and that stays true of the file.
        """
        store.seed(
            make_episode(
                guid="listed",
                status=S.PUBLISHED,
                digest_id="archive:test-show:2026-02",
                tier0={"relevance_guess": 7, "confidence": 9, "route": "ESCALATE"},
            )
        )
        episode_id = make_episode(guid="listed")["_id"]
        with self._client(tmp_path, store) as client:
            response = client.post(
                f"/api/v1/episodes/{episode_id}/summarize?allow_asr=false&wait=true",
                headers=KEY,
            )
        assert response.status_code == 200, response.json()
        # Back to PUBLISHED, carrying a summary: it is still listed in the
        # archive file, which is not rewritten. See the test below.
        assert response.json()["result"]["status"] == "PUBLISHED"
        assert response.json()["result"]["relevance_score"] is not None

    def test_an_episode_whose_summary_was_written_is_still_refused(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The protection that matters: a summary a reader has already read in a
        digest file must not be silently replaced in the database alone."""
        store.seed(
            make_episode(
                guid="written",
                status=S.PUBLISHED,
                digest_id="digest:2026-W05",
                tier1={
                    "relevance_score": 8,
                    "summary_md": "already written into the digest",
                    "summary_basis": "transcript",
                },
            )
        )
        episode_id = make_episode(guid="written")["_id"]
        with self._client(tmp_path, store) as client:
            response = client.post(
                f"/api/v1/episodes/{episode_id}/summarize?allow_asr=false&wait=true",
                headers=KEY,
            )
        assert response.status_code == 409
        assert "disagreeing" in response.json()["detail"]

    def test_a_listed_episode_goes_back_to_published(self, tmp_path, store: MemoryStore) -> None:
        """Summarising does not un-publish it.

        It stays listed in a file that is never rewritten, and its claim is
        still held, so no digest will take it. Leaving it at READY_FOR_DIGEST
        would assert it was waiting for one — a queue entry that can never
        drain.
        """
        store.seed(
            make_episode(
                guid="relist",
                status=S.PUBLISHED,
                digest_id="archive:test-show:2026-02",
                tier0={"relevance_guess": 7, "confidence": 9, "route": "ESCALATE"},
            )
        )
        episode_id = make_episode(guid="relist")["_id"]
        with self._client(tmp_path, store) as client:
            body = client.post(
                f"/api/v1/episodes/{episode_id}/summarize?allow_asr=false&wait=true",
                headers=KEY,
            ).json()
        assert body["result"]["status"] == "PUBLISHED"

        doc = next(d for d in store.docs_of_type("episode") if d["_id"] == episode_id)
        assert doc["status"] == "PUBLISHED"
        assert doc["tier1"]["summary_md"], "it does have a summary now"
        assert doc["summary_after_listing"] is True

    def test_a_summary_added_after_listing_can_be_redone(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """That summary is in no file, so redoing it cannot make one disagree.

        Without the marker the episode would be published-with-a-summary again
        and the guard would lock it, so a description-only result could never be
        redone properly.
        """
        store.seed(
            make_episode(
                guid="redoable",
                status=S.PUBLISHED,
                digest_id="archive:test-show:2026-02",
                summary_after_listing=True,
                tier1={
                    "relevance_score": 5,
                    "summary_md": "added on request, not in any file",
                    "summary_basis": "description_only",
                },
            )
        )
        episode_id = make_episode(guid="redoable")["_id"]
        with self._client(tmp_path, store) as client:
            response = client.post(
                f"/api/v1/episodes/{episode_id}/summarize?allow_asr=false&wait=true",
                headers=KEY,
            )
        assert response.status_code == 200, response.json()

    def test_summarising_a_listed_episode_does_not_free_its_claim(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """It was already listed somewhere. Releasing the claim would let a
        later digest list it a second time."""
        store.seed(
            make_episode(
                guid="claimed",
                status=S.PUBLISHED,
                digest_id="archive:test-show:2026-02",
                tier0={"relevance_guess": 7, "confidence": 9, "route": "ESCALATE"},
            )
        )
        episode_id = make_episode(guid="claimed")["_id"]
        with self._client(tmp_path, store) as client:
            client.post(
                f"/api/v1/episodes/{episode_id}/summarize?allow_asr=false&wait=true",
                headers=KEY,
            )
        doc = next(d for d in store.docs_of_type("episode") if d["_id"] == episode_id)
        assert doc["digest_id"] == "archive:test-show:2026-02"

    def test_background_start_returns_immediately(self, tmp_path, store: MemoryStore) -> None:
        """ASR can outlast any browser, so the default does not block."""
        store.seed(make_episode(guid="bg", status=S.NEW))
        episode_id = make_episode(guid="bg")["_id"]
        with self._client(tmp_path, store) as client:
            body = client.post(f"/api/v1/episodes/{episode_id}/summarize", headers=KEY).json()
        assert body["waited"] is False
        assert "poll the episode" in body["detail"]


class TestDisconnectDoesNotCancelWork:
    """Regression: awaiting the coroutine inline meant a client disconnect
    cancelled it. A summarisation killed midway through map-reduce threw away
    seven LLM calls and left the episode exactly where it started.
    """

    async def test_work_survives_its_caller_being_cancelled(self) -> None:
        import asyncio
        from types import SimpleNamespace

        from podcast_agent.api.routes import _spawn

        started, release = asyncio.Event(), asyncio.Event()
        finished: list[str] = []

        async def slow() -> dict[str, Any]:
            started.set()
            await release.wait()
            finished.append("done")
            return {"ok": True}

        tasks: set[asyncio.Task[Any]] = set()
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(background_tasks=tasks))
        )

        task = _spawn(request, "slow", slow)  # type: ignore[arg-type]
        await started.wait()
        assert task in tasks  # owned by the app, not the request

        # The "client hung up" moment: the awaiting side is cancelled. This
        # mirrors what the endpoint does — await a shield, not the task itself.
        async def await_shielded() -> Any:
            return await asyncio.shield(task)

        waiter = asyncio.create_task(await_shielded())
        await asyncio.sleep(0)
        waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter

        assert not task.cancelled(), "work was cancelled along with its caller"
        release.set()
        assert await task == {"ok": True}
        assert finished == ["done"]

    async def test_completed_task_is_released(self) -> None:
        """The set must not grow without bound over a long uptime."""
        import asyncio
        from types import SimpleNamespace

        from podcast_agent.api.routes import _spawn

        tasks: set[asyncio.Task[Any]] = set()
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(background_tasks=tasks))
        )

        async def quick() -> dict[str, Any]:
            return {"ok": True}

        await _spawn(request, "quick", quick)  # type: ignore[arg-type]
        await asyncio.sleep(0)
        assert tasks == set()


class TestWhyNoSummary:
    """A PUBLISHED episode with no summary and no transcript is the designed
    grey-zone path, not a failure — the API must carry enough to say which.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_grey_zone_episode_keeps_its_tier0_verdict(self, tmp_path, store: MemoryStore) -> None:
        """Published as a one-liner: judged, listed, never summarised."""
        store.seed(
            make_episode(
                guid="grey",
                status=S.PUBLISHED,
                digest_id="digest:2026-W31",
                tier0={
                    "relevance_guess": 5,
                    "confidence": 8,
                    "route": "DIGEST_DIRECT",
                    "rule": "grey_zone",
                },
            )
        )
        with self._client(tmp_path, store) as client:
            episode = client.get("/api/v1/episodes", headers=KEY).json()["episodes"][0]

        assert episode["status"] == "PUBLISHED"
        assert episode["has_summary"] is False
        assert episode["has_transcript"] is False
        # The explanation the UI renders comes from these two fields.
        assert episode["tier0"]["route"] == "DIGEST_DIRECT"
        assert episode["tier0"]["relevance_guess"] == 5

    def test_dropped_episode_is_distinguishable(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="dropped",
                status=S.DROPPED,
                tier0={"relevance_guess": 1, "confidence": 9, "route": "DROP"},
            )
        )
        with self._client(tmp_path, store) as client:
            episode = client.get("/api/v1/episodes", headers=KEY).json()["episodes"][0]
        assert episode["tier0"]["route"] == "DROP"
        assert episode["has_summary"] is False

    def test_untriaged_episode_has_no_tier0(self, tmp_path, store: MemoryStore) -> None:
        store.seed(make_episode(guid="new", status=S.NEW))
        with self._client(tmp_path, store) as client:
            episode = client.get("/api/v1/episodes", headers=KEY).json()["episodes"][0]
        assert episode["tier0"] is None


class TestExportingOneEpisode:
    """Handing a single summary to someone who does not have the vault.

    The vault notes the digest writes are Obsidian documents — frontmatter,
    wikilinks, a backlink to the week. This export is the same summary with all
    three removed, because it leaves the vault.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed(
        self, store: MemoryStore, *, _status: EpisodeStatus = S.READY_FOR_DIGEST, **tier1: Any
    ) -> str:
        doc = make_episode(
            guid="exportable",
            title="Ransomware crews are hiring",
            status=_status,
            tier1={
                "relevance_score": 8,
                "summary_basis": "transcript",
                "why_it_matters": "Two crews now run formal recruitment pipelines.",
                "summary_md": "Affiliate churn hit **40%** this quarter.",
                "key_takeaways": ["Affiliate churn is up", "Recruitment is public"],
                "entities": ["LockBit", "Cl0p"],
                **tier1,
            },
        )
        store.seed(doc)
        return str(doc["_id"])

    def _export(self, client: TestClient, episode_id: str) -> Any:
        return client.get(f"/api/v1/episodes/{episode_id}/export", headers=KEY)

    def test_it_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        episode_id = self._seed(store)
        with self._client(tmp_path, store) as client:
            assert client.get(f"/api/v1/episodes/{episode_id}/export").status_code == 401

    def test_the_markdown_carries_the_whole_summary(self, tmp_path, store: MemoryStore) -> None:
        episode_id = self._seed(store)
        with self._client(tmp_path, store) as client:
            body = self._export(client, episode_id).json()

        markdown = body["markdown"]
        assert markdown.startswith("# Ransomware crews are hiring")
        assert "Two crews now run formal recruitment pipelines." in markdown
        assert "Affiliate churn hit **40%** this quarter." in markdown
        assert "- Affiliate churn is up" in markdown
        assert "- Recruitment is public" in markdown
        assert "LockBit · Cl0p" in markdown
        # The show, the date and a way back to the episode itself: without these
        # the file is an anonymous block of text in someone else's inbox.
        assert "Test Show" in markdown
        assert "2026-07-28" in markdown
        assert "https://example.com/ep1" in markdown

    def test_it_says_where_the_summary_came_from(self, tmp_path, store: MemoryStore) -> None:
        """The recipient did not run the pipeline and cannot tell a transcript
        summary from one written off a marketing blurb."""
        episode_id = self._seed(store, summary_basis="description_only")
        with self._client(tmp_path, store) as client:
            markdown = self._export(client, episode_id).json()["markdown"]
        assert "description only (no transcript available)" in markdown
        assert "8/10" in markdown

    def test_no_vault_only_markup_survives(self, tmp_path, store: MemoryStore) -> None:
        episode_id = self._seed(store)
        with self._client(tmp_path, store) as client:
            markdown = self._export(client, episode_id).json()["markdown"]
        assert not markdown.startswith("---")
        assert "[[" not in markdown
        assert "podcast-digest-" not in markdown

    def test_the_filename_is_filable_as_is(self, tmp_path, store: MemoryStore) -> None:
        episode_id = self._seed(store)
        with self._client(tmp_path, store) as client:
            body = self._export(client, episode_id).json()
        assert body["filename"] == "test-show-2026-07-28-ransomware-crews-are-hiring.md"
        assert body["episode_id"] == episode_id

    def test_an_episode_with_no_summary_is_refused(self, tmp_path, store: MemoryStore) -> None:
        """409, not 404: the episode exists, there is simply nothing to send."""
        store.seed(make_episode(guid="bare", status=S.NEW))
        with self._client(tmp_path, store) as client:
            response = self._export(client, episode_doc_id("test-show", "bare"))
        assert response.status_code == 409
        assert "no summary" in response.json()["detail"]

    def test_an_unknown_episode_is_a_404(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert self._export(client, "episode:nope").status_code == 404

    def test_a_grey_zone_episode_summarised_later_still_exports(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """DIGEST_DIRECT has its own one-liner view in the digest, which has no
        summary_md. Reaching it through that branch would export an empty file.
        """
        episode_id = self._seed(store, _status=S.DIGEST_DIRECT)
        with self._client(tmp_path, store) as client:
            markdown = self._export(client, episode_id).json()["markdown"]
        assert "Affiliate churn hit **40%** this quarter." in markdown


class TestPodcastManagement:
    """config.yaml is the declared baseline; the console writes overrides to the
    database, because config.yaml is mounted read-only in the container."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_page_is_served_and_self_contained(self, tmp_path, store: MemoryStore) -> None:
        """No external resource is fetched. A URL inside a placeholder attribute
        is text on screen, not a request, so match on resource references."""
        import re

        with self._client(tmp_path, store) as client:
            body = client.get("/admin/podcasts").text
        assert "Podcasts" in body
        external = re.findall(r'(?:src|href)\s*=\s*["\']https?://|@import|url\(\s*https?://', body)
        assert external == [], f"external resource reference: {external}"

    def test_endpoints_need_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/podcasts").status_code == 401
            assert client.post("/api/v1/podcasts", json={}).status_code == 401

    def test_lists_config_shows_with_provenance(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        assert body["count"] == 2
        show = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert show["source"] == "config"
        assert show["overridden"] == []
        # Nothing is removable; disabling is the only off switch.
        assert "removable" not in show
        assert show["enabled"] is True

    def test_listing_carries_the_feed_description(self, tmp_path, store: MemoryStore) -> None:
        """Captured at poll time onto the podcast doc, surfaced for the console."""
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "description": "Weekly infosec news.",
            }
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        podcast = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert podcast["description"] == "Weekly infosec news."

    def test_a_never_polled_podcast_says_so_rather_than_erroring(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """ "Not polled yet" and "polled, carries none" are different answers."""
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        for entry in body["podcasts"]:
            assert entry["description"] == ""
            assert entry["polled_for_metadata"] is False
            assert entry["transcripts_seen"] is None
            assert entry["transcripts_of"] is None

    def test_transcript_coverage_is_reported_once_polled(
        self, tmp_path, store: MemoryStore
    ) -> None:
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "feed_entries_seen": 25,
                "feed_transcripts_seen": 25,
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        entry = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert entry["transcripts_seen"] == 25
        assert entry["transcripts_of"] == 25
        assert entry["polled_for_metadata"] is True

    def test_a_feed_carrying_no_transcripts_reports_zero_not_unknown(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """A measured zero is an answer; it must not read as "not polled"."""
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "feed_entries_seen": 25,
                "feed_transcripts_seen": 0,
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        entry = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert entry["transcripts_seen"] == 0
        assert entry["transcripts_of"] == 25

    def test_feed_cadence_beats_the_one_derived_from_held_episodes(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """What we hold is bounded by the lookback window; the feed is not."""
        now = datetime.now(UTC)
        for i in range(6):
            store.seed(
                make_episode(
                    guid=f"ep-{i}",
                    title=f"Ep {i}",
                    status=S.PUBLISHED,
                    published_at=now - timedelta(days=i),  # would read as ~daily
                )
            )
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "feed_cadence": "~weekly",
                "feed_cadence_detail": "median 7.0 days",
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        entry = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert entry["cadence"] == "~weekly"
        assert entry["cadence_source"] == "feed"

    def test_held_episodes_are_the_fallback_when_the_feed_said_nothing(
        self, tmp_path, store: MemoryStore
    ) -> None:
        now = datetime.now(UTC)
        for i in range(6):
            store.seed(
                make_episode(
                    guid=f"ep-{i}",
                    title=f"Ep {i}",
                    status=S.PUBLISHED,
                    published_at=now - timedelta(days=7 * i),
                )
            )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        entry = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert entry["cadence"] == "~weekly"
        assert entry["cadence_source"] == "episodes held"

    def test_the_transcripts_note_accounts_for_the_asr_setting(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Advice that ignores the next column reads as a contradiction.

        "no ASR needed" beside a podcast whose ASR toggle says ON looks like a
        misconfiguration. It is not one: acquisition tries the published
        transcript first and only falls back to ASR, so ASR is never reached.
        The line states the outcome rather than giving advice.
        """
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/podcasts").text
        for outcome in (
            "transcription on, never reached",  # publishes transcripts, on
            "no transcription needed",  # publishes transcripts, off
            "transcribed on this machine",  # publishes none, on
            "summarised from descriptions",  # publishes none, off
        ):
            assert outcome in page, f"missing outcome: {outcome}"
        assert "p.asr_enabled" in page, "the note ignores the ASR setting"

    def test_the_legend_says_asr_is_a_fallback(self, tmp_path, store: MemoryStore) -> None:
        """The reason the combination is not a contradiction."""
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/podcasts").text
        assert "fallback" in page
        assert "published transcript first" in page

    def test_a_podcast_with_nothing_to_summarise_from_says_so(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """A weekly podcast holding one episode is otherwise inexplicable.

        Routine polling only looks forward, so history comes from the archive
        walk. It summarises from a transcript, and with none published and local
        transcription off there is nothing to summarise from — the episodes are
        indexed and scored instead.
        """
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "feed_entries_seen": 25,
                "feed_transcripts_seen": 0,
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"asr_enabled": False})
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        entry = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert entry["archive_indexes_only"] is True

    def test_transcribing_locally_makes_summaries_reachable(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The one switch that changes it, now that there is no global override."""
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "feed_entries_seen": 25,
                "feed_transcripts_seen": 0,
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"asr_enabled": True})
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        entry = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert entry["archive_indexes_only"] is False

    def test_a_podcast_publishing_transcripts_does_not_say_so(
        self, tmp_path, store: MemoryStore
    ) -> None:
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "feed_entries_seen": 25,
                "feed_transcripts_seen": 25,
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        entry = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert entry["archive_indexes_only"] is False

    def test_an_unpolled_podcast_makes_no_claim_about_its_archive(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Not yet measured is not the same as measured zero."""
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        assert all(p["archive_indexes_only"] is False for p in body["podcasts"])

    def test_an_unreachable_archive_mode_says_so(self, tmp_path, store: MemoryStore) -> None:
        """ "Summarise" beside "indexed only" is a contradiction, not a setting.

        The archive walk only summarises an episode it has a transcript for, and
        while backfill.require_transcript is set it will not make one. A podcast
        publishing none is therefore indexed and scored whatever its archive mode
        says — which is exactly what happened after setting it to summarise and
        turning local transcription on.
        """
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "feed_entries_seen": 25,
                "feed_transcripts_seen": 0,
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            # Publishes nothing and is not set to transcribe: nothing to
            # summarise from, whatever the archive mode says.
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"asr_enabled": False})
            entry = next(
                p
                for p in client.get("/api/v1/podcasts", headers=KEY).json()["podcasts"]
                if p["slug"] == "test-show"
            )
        assert entry["backfill_mode"] == "full"
        assert entry["archive_indexes_only"] is True

    def test_a_podcast_publishing_transcripts_can_reach_summarise(
        self, tmp_path, store: MemoryStore
    ) -> None:
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "feed_entries_seen": 25,
                "feed_transcripts_seen": 25,
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            entry = next(
                p
                for p in client.get("/api/v1/podcasts", headers=KEY).json()["podcasts"]
                if p["slug"] == "test-show"
            )
        assert entry["archive_indexes_only"] is False

    def test_a_skipped_podcast_makes_no_claim_about_summarising(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Nothing is reachable when the archive ignores it entirely."""
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "overrides": {"backfill_mode": "skip"},
                "feed_entries_seen": 25,
                "feed_transcripts_seen": 0,
                "feed_metadata_at": iso_now(),
            }
        )
        with self._client(tmp_path, store) as client:
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"backfill_mode": "skip"})
            entry = next(
                p
                for p in client.get("/api/v1/podcasts", headers=KEY).json()["podcasts"]
                if p["slug"] == "test-show"
            )
        assert entry["archive_indexes_only"] is False

    def test_the_page_puts_the_warning_beside_the_setting(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """It was two columns away, next to the episode count."""
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/podcasts").text
        assert "not reachable — indexes only" in page
        assert "Turn on Transcribe locally" in page

    def test_the_page_has_a_transcripts_column(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/podcasts").text
        assert "<th>Transcripts</th>" in page
        assert "<strong>Transcripts</strong>" in page, "column is not in the legend"

    def test_listing_reports_publication_cadence(self, tmp_path, store: MemoryStore) -> None:
        now = datetime.now(UTC)
        for i in range(6):
            store.seed(
                {
                    "_id": episode_doc_id("test-show", f"ep-{i}"),
                    "type": "episode",
                    "podcast_slug": "test-show",
                    "status": "PUBLISHED",
                    "published_at": iso(now - timedelta(days=7 * i)),
                }
            )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        podcast = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert podcast["cadence"] == "~weekly"
        assert "median" in podcast["cadence_detail"]
        assert podcast["episodes"] == 6

    def test_cadence_is_null_without_enough_history(self, tmp_path, store: MemoryStore) -> None:
        """Better nothing than a rhythm inferred from two episodes."""
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/podcasts", headers=KEY).json()
        podcast = next(p for p in body["podcasts"] if p["slug"] == "test-show")
        assert podcast["cadence"] is None
        assert podcast["cadence_detail"] is None

    def test_override_changes_the_effective_value(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert (
                client.patch(
                    "/api/v1/podcasts/test-show", headers=KEY, json={"asr_enabled": False}
                ).status_code
                == 200
            )
            show = next(
                p
                for p in client.get("/api/v1/podcasts", headers=KEY).json()["podcasts"]
                if p["slug"] == "test-show"
            )
        assert show["asr_enabled"] is False
        # Marked as overridden so it is clear it no longer tracks config.yaml.
        assert "asr_enabled" in show["overridden"]

    def test_override_can_be_reverted_to_config(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"priority": "low"})
            client.delete("/api/v1/podcasts/test-show/overrides/priority", headers=KEY)
            show = next(
                p
                for p in client.get("/api/v1/podcasts", headers=KEY).json()["podcasts"]
                if p["slug"] == "test-show"
            )
        assert show["priority"] == "med"  # back to the config value
        assert show["overridden"] == []

    def test_unknown_field_is_rejected(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            response = client.patch(
                "/api/v1/podcasts/test-show", headers=KEY, json={"slug": "renamed"}
            )
        assert response.status_code == 422

    def test_no_podcast_can_be_deleted(self, tmp_path, store: MemoryStore) -> None:
        """Shows are disabled, never deleted — a deleted show would leave its
        episodes in the database with no way to explain where they came from."""
        with self._client(tmp_path, store) as client:
            assert client.delete("/api/v1/podcasts/test-show", headers=KEY).status_code == 405
            client.post(
                "/api/v1/podcasts",
                headers=KEY,
                json={
                    "slug": "new-show",
                    "name": "New Show",
                    "feed_url": "https://cdn-host.net/feed.xml",
                },
            )
            assert client.delete("/api/v1/podcasts/new-show", headers=KEY).status_code == 405

    def test_add_a_console_show(self, tmp_path, store: MemoryStore) -> None:
        new = {
            "slug": "new-show",
            "name": "New Show",
            "feed_url": "https://cdn-host.net/feed.xml",
        }
        with self._client(tmp_path, store) as client:
            assert client.post("/api/v1/podcasts", headers=KEY, json=new).status_code == 201
            listed = client.get("/api/v1/podcasts", headers=KEY).json()
            added = next(p for p in listed["podcasts"] if p["slug"] == "new-show")
        assert added["source"] == "console"
        assert added["asr_enabled"] is False  # off unless asked for
        assert added["enabled"] is True

    def test_disabling_stops_intake_and_keeps_history(self, tmp_path, store: MemoryStore) -> None:
        """The only off switch. Summaries and episodes survive untouched."""
        store.seed(
            make_episode(guid="kept", status=S.PUBLISHED, tier1={"relevance_score": 8}),
            make_episode(guid="queued", status=S.NEW),
        )
        with self._client(tmp_path, store) as client:
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"enabled": False})
            show = next(
                p
                for p in client.get("/api/v1/podcasts", headers=KEY).json()["podcasts"]
                if p["slug"] == "test-show"
            )
            episodes = client.get("/api/v1/episodes", headers=KEY).json()

        assert show["enabled"] is False
        assert show["episodes"] == 2  # history intact
        # Already-ingested episodes still finish; the count makes that visible.
        assert show["queued"] == 1
        assert len(episodes["episodes"]) == 2

    def test_duplicate_slug_is_refused(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            response = client.post(
                "/api/v1/podcasts",
                headers=KEY,
                json={
                    "slug": "test-show",
                    "name": "Clash",
                    "feed_url": "https://cdn-host.net/f.xml",
                },
            )
        assert response.status_code == 409

    def test_non_http_feed_url_is_refused(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            response = client.post(
                "/api/v1/podcasts",
                headers=KEY,
                json={"slug": "bad", "name": "Bad", "feed_url": "file:///etc/passwd"},
            )
        assert response.status_code == 422

    def test_a_feed_on_a_bare_ip_is_allowed(self, tmp_path, store: MemoryStore) -> None:
        """Deliberate: the allowlist governs where a feed's *audio* may come from,
        not where the feed itself lives, so a self-hosted feed on a LAN address
        works. Ingestion applies exactly the same rule."""
        with self._client(tmp_path, store) as client:
            response = client.post(
                "/api/v1/podcasts",
                headers=KEY,
                json={
                    "slug": "selfhosted",
                    "name": "Self hosted",
                    "feed_url": "http://192.168.1.9/feed.xml",
                },
            )
        assert response.status_code == 201


class TestEpisodePaging:
    """The console pages at 50; a pager needs to know the total, not just the
    size of the page it was handed."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed(self, store: MemoryStore, count: int) -> None:
        base = datetime(2026, 7, 1, tzinfo=UTC)
        store.seed(
            *[
                make_episode(guid=f"e{i}", published_at=base + timedelta(days=i))
                for i in range(count)
            ]
        )

    def test_total_is_the_whole_result_set(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, 120)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?limit=50", headers=KEY).json()
        assert body["count"] == 50  # this page
        assert body["total"] == 120  # everything matching

    def test_paging_walks_the_whole_set_without_repeats(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, 120)
        seen: list[str] = []
        with self._client(tmp_path, store) as client:
            for skip in (0, 50, 100):
                page = client.get(f"/api/v1/episodes?limit=50&skip={skip}", headers=KEY).json()
                seen += [e["_id"] for e in page["episodes"]]
        assert len(seen) == 120
        assert len(set(seen)) == 120  # no episode appears on two pages

    def test_total_respects_the_filter(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, 10)
        store.seed(make_episode(guid="other", slug="priority-show"))
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?podcast=priority-show&limit=50", headers=KEY).json()
        assert body["total"] == 1


class TestConsolePages:
    """Three pages, each self-contained, each reachable from the others."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    ALL_PAGES: ClassVar[list[str]] = [
        "/admin",
        "/admin/episodes",
        "/admin/podcasts",
        "/admin/backfill",
        "/admin/settings",
    ]

    @pytest.mark.parametrize(
        "path",
        ["/admin", "/admin/episodes", "/admin/podcasts", "/admin/backfill", "/admin/settings"],
    )
    def test_served_and_self_contained(self, tmp_path, store: MemoryStore, path: str) -> None:
        import re

        with self._client(tmp_path, store) as client:
            response = client.get(path)
        assert response.status_code == 200
        external = re.findall(
            r'(?:src|href)\s*=\s*["\']https?://|@import|url\(\s*https?://', response.text
        )
        assert external == [], f"{path} references {external}"

    def test_every_page_carries_the_same_nav(self, tmp_path, store: MemoryStore) -> None:
        """The nav is injected from one definition, so pages cannot disagree
        about what exists — the failure mode when it was copied per file."""
        from podcast_agent.api.pages import NAV_ITEMS

        with self._client(tmp_path, store) as client:
            for path in self.ALL_PAGES:
                body = client.get(path).text
                for href, label in NAV_ITEMS:
                    assert f'href="{href}"' in body, f"{path} missing nav link {href}"
                    assert label in body, f"{path} missing nav label {label}"

    def test_nav_marks_the_current_page(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/podcasts").text
        assert "href=\"/admin/podcasts\" class='here'" in body

    def test_no_page_still_carries_the_placeholder(self, tmp_path, store: MemoryStore) -> None:
        """A page that forgot the marker would silently render without a nav."""
        with self._client(tmp_path, store) as client:
            for path in self.ALL_PAGES:
                assert "<!--NAV-->" not in client.get(path).text, path

    def test_episodes_moved_off_the_operations_page(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            ops = client.get("/admin").text
            episodes = client.get("/admin/episodes").text
        assert "epRows" not in ops
        assert "epRows" in episodes

    def test_episodes_page_explains_every_status(self, tmp_path, store: MemoryStore) -> None:
        """The legend must not go stale as statuses are added."""
        from podcast_agent.state import EpisodeStatus

        with self._client(tmp_path, store) as client:
            body = client.get("/admin/episodes").text
        for status_value in EpisodeStatus:
            assert status_value.value in body, f"legend missing {status_value.value}"

    def test_podcasts_page_explains_every_backfill_mode(self, tmp_path, store: MemoryStore) -> None:
        """The archive-mode legend must not go stale as modes are added."""
        from typing import get_args

        from podcast_agent.config import PodcastConfig

        modes = get_args(PodcastConfig.model_fields["backfill_mode"].annotation)
        assert modes, "backfill_mode is no longer a Literal — update this test"

        with self._client(tmp_path, store) as client:
            body = client.get("/admin/podcasts").text
        assert "Archive mode" in body
        for mode in modes:
            assert f"<code>{mode}</code>" in body, f"legend missing {mode}"

    def test_podcasts_page_explains_every_column(self, tmp_path, store: MemoryStore) -> None:
        """A nine-column table with no legend is a puzzle, not a console."""
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/podcasts").text
        assert "What the columns mean" in body
        for column in (
            "Priority",
            "Always escalate",
            "Transcribe locally",
            "Archive mode",
            "Episodes",
            "Feed health",
            "Intake",
        ):
            assert f"<strong>{column}</strong>" in body, f"legend missing {column}"

    def test_podcasts_page_says_backfill_mode_does_not_affect_weekly_digests(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The column sits beside ASR and priority, which *do* affect the weekly
        run, so the one thing it must say is that it does not."""
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/podcasts").text
        assert "No effect on weekly digests" in body

    def test_operations_page_explains_each_job(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/admin").text
        assert "Scheduled jobs" in body
        for hint in ("every 6 h", "every 30 min", "Friday 06:00", "nightly at 04:00"):
            assert hint in body, f"job description missing {hint!r}"

    def test_backfill_controls_left_the_operations_page(self, tmp_path, store: MemoryStore) -> None:
        """Historical intake is opt-in and long-running; it does not belong next
        to buttons that take seconds."""
        with self._client(tmp_path, store) as client:
            ops = client.get("/admin").text
            backfill = client.get("/admin/backfill").text
        assert "backfill/control" not in ops
        assert "backfill/control" in backfill

    def test_feed_table_is_not_duplicated_on_the_operations_page(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Per-feed detail lives on the Podcasts page; the operations page keeps
        only the "is anything broken?" signal."""
        with self._client(tmp_path, store) as client:
            ops = client.get("/admin").text
        assert "feedRows" not in ops
        assert "feedHealth" in ops


class TestSettingsConsole:
    """config.yaml is the baseline; the console stores overrides that apply at
    the next restart, because swapping a provider mid-run is a bad trade."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_requires_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/settings").status_code == 401

    def test_reports_the_running_configuration(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["tiers"]["tier0"]["primary"]["model"] == "test-small"
        assert body["tiers"]["tier0"]["active_chain"] == ["ollama_chat/test-small"]
        assert body["pipeline"]["digest_threshold"] == 5
        assert len(body["interest_profile"]) == 2
        assert body["pending_restart"] is False

    def test_a_reply_limit_survives_the_round_trip(self, tmp_path, store: MemoryStore) -> None:
        """Unset, providers pick their own and pick badly.

        OpenRouter cut tier-1 off mid-JSON on every call: billed, unusable, and
        the tier fell through to the local model anyway. The limit was only
        settable by editing config.yaml, which this deployment overrides.
        """
        with self._client(tmp_path, store) as client:
            saved = client.put(
                "/api/v1/settings",
                headers=KEY,
                json={
                    "tiers": {
                        "tier0": {
                            "primary": {
                                "provider": "ollama",
                                "model": "small",
                                "max_tokens": 900,
                            },
                            "timeout_s": 60,
                        },
                        "tier1": {
                            "primary": {"provider": "ollama", "model": "big"},
                            "timeout_s": 300,
                        },
                    }
                },
            )
            assert saved.status_code == 200
            body = client.get("/api/v1/settings", headers=KEY).json()
        stored = body["overrides"]["llm"]["tiers"]["tier0"]["primary"]
        assert stored["max_tokens"] == 900
        # And the tier that did not set one keeps the provider default.
        assert body["overrides"]["llm"]["tiers"]["tier1"]["primary"]["max_tokens"] is None

    def test_a_model_host_the_file_never_named_is_refused(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """`api_base` decides where every prompt and transcript is sent.

        A console key alone must not be able to add a destination: whoever can
        write that value can read everything the pipeline reads.
        """
        with self._client(tmp_path, store) as client:
            refused = client.put(
                "/api/v1/settings",
                headers=KEY,
                json={
                    "tiers": {
                        "tier0": {
                            "primary": {
                                "provider": "ollama",
                                "model": "small",
                                "api_base": "http://attacker.example:11434",
                            },
                            "timeout_s": 60,
                        },
                        "tier1": {
                            "primary": {"provider": "ollama", "model": "big"},
                            "timeout_s": 300,
                        },
                    }
                },
            )
        assert refused.status_code == 400
        assert "attacker.example" in refused.json()["detail"]
        assert "config.yaml" in refused.json()["detail"]

    def test_nothing_is_stored_when_the_host_is_refused(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            client.put(
                "/api/v1/settings",
                headers=KEY,
                json={
                    "tiers": {
                        "tier0": {
                            "primary": {
                                "provider": "ollama",
                                "model": "small",
                                "api_base": "http://attacker.example:11434",
                            },
                            "timeout_s": 60,
                        },
                        "tier1": {
                            "primary": {"provider": "ollama", "model": "big"},
                            "timeout_s": 300,
                        },
                    }
                },
            )
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["overrides"] == {}

    def test_the_local_default_is_still_settable(self, tmp_path, store: MemoryStore) -> None:
        """The ordinary case — pointing a tier at the Ollama on this machine."""
        with self._client(tmp_path, store) as client:
            saved = client.put(
                "/api/v1/settings",
                headers=KEY,
                json={
                    "tiers": {
                        "tier0": {
                            "primary": {
                                "provider": "ollama",
                                "model": "small",
                                "api_base": "http://127.0.0.1:11434",
                            },
                            "timeout_s": 60,
                        },
                        "tier1": {
                            "primary": {"provider": "ollama", "model": "big"},
                            "timeout_s": 300,
                        },
                    }
                },
            )
        assert saved.status_code == 200

    def test_a_host_the_file_does_name_is_accepted(self, tmp_path, store: MemoryStore) -> None:
        """Deploy-time authority: config.yaml names it, so the console may use it."""
        settings = make_settings(
            tmp_path,
            llm={
                "tiers": {
                    "tier0": {
                        "primary": {
                            "provider": "ollama",
                            "model": "test-small",
                            "api_base": "http://gpu-box.lan:11434",
                        },
                        "timeout_s": 30,
                    },
                    "tier1": {
                        "primary": {"provider": "ollama", "model": "test-large"},
                        "timeout_s": 60,
                    },
                }
            },
        )
        with TestClient(build_app(settings, store=store, llm=FakeLLM())) as client:
            saved = client.put(
                "/api/v1/settings",
                headers=KEY,
                json={
                    "tiers": {
                        "tier0": {
                            "primary": {
                                "provider": "ollama",
                                "model": "small",
                                "api_base": "http://gpu-box.lan:11434",
                            },
                            "timeout_s": 60,
                        },
                        "tier1": {
                            "primary": {"provider": "ollama", "model": "big"},
                            "timeout_s": 300,
                        },
                    }
                },
            )
        assert saved.status_code == 200

    #: A self-contained config file, because the boot-time override path reads
    #: one: `load_settings` re-reads from disk rather than layering onto the
    #: settings the app was handed, so a test that skips this asserts nothing.
    _CONFIG = """
output:
  digest_dir: DIGEST_DIR
  work_dir: WORK_DIR
podcasts:
  - slug: test-show
    name: Test Show
    feed_url: https://example.com/feed.xml
interest_profile:
  - key: ot_ics
    label: OT/ICS security
    description: industrial control systems
    weight: 10
llm:
  tiers:
    tier0:
      primary: {provider: ollama, model: test-small}
      timeout_s: 30
    tier1:
      primary: {provider: ollama, model: test-large}
      timeout_s: 60
"""

    def _from_file(self, tmp_path, store: MemoryStore, monkeypatch) -> TestClient:
        path = tmp_path / "config.yaml"
        path.write_text(
            self._CONFIG.replace("DIGEST_DIR", str(tmp_path / "digests")).replace(
                "WORK_DIR", str(tmp_path / "work")
            )
        )
        monkeypatch.setenv("PODAGENT_CONFIG_FILE", str(path))
        monkeypatch.setenv("PODAGENT_ADMIN_API_KEY", "test-admin-key")
        settings = load_settings(config_file=path)
        return TestClient(build_app(settings, store=store, llm=FakeLLM()))

    def _stored(self, api_base: str | None) -> dict[str, Any]:
        primary: dict[str, Any] = {"provider": "ollama", "model": "swapped"}
        if api_base:
            primary["api_base"] = api_base
        return {
            "_id": "control:settings",
            "type": "control",
            "key": "settings",
            "overrides": {
                "llm": {
                    "tiers": {
                        "tier0": {"primary": primary, "timeout_s": 60},
                        "tier1": {
                            "primary": {"provider": "ollama", "model": "test-large"},
                            "timeout_s": 300,
                        },
                    }
                }
            },
            "updated_at": "2026-08-01T00:00:00+00:00",
        }

    def test_a_sound_stored_override_is_adopted_at_boot(
        self, tmp_path, store: MemoryStore, monkeypatch
    ) -> None:
        """The positive control. Without it the rejection test below would pass
        just as well against a boot path that adopts nothing at all."""
        store.seed(self._stored(None))
        with self._from_file(tmp_path, store, monkeypatch) as client:
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["tiers"]["tier0"]["primary"]["model"] == "swapped"

    def test_a_stored_override_with_a_bad_host_is_not_adopted_at_boot(
        self, tmp_path, store: MemoryStore, monkeypatch
    ) -> None:
        """The document is writable by anything with database access, so the
        check runs again where the endpoints become real. A rejected override
        leaves the service on the file rather than on the override."""
        store.seed(self._stored("http://attacker.example:11434"))
        with self._from_file(tmp_path, store, monkeypatch) as client:
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["tiers"]["tier0"]["primary"]["model"] == "test-small"
        assert body["tiers"]["tier0"]["primary"]["api_base"] is None

    def test_saving_marks_a_restart_pending(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            saved = client.put(
                "/api/v1/settings", headers=KEY, json={"pipeline": {"digest_threshold": 6}}
            ).json()
            assert saved["pending_restart"] is True
            # The running process is unchanged until it restarts.
            after = client.get("/api/v1/settings", headers=KEY).json()
        assert after["pipeline"]["digest_threshold"] == 5
        assert after["overrides"]["pipeline"]["digest_threshold"] == 6

    def test_pending_banner_can_tell_which_process_is_out_of_date(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """A restart aimed at the wrong container leaves the banner up.

        Without the boot time on the page, that is indistinguishable from a
        restart that never happened, so the banner reads as stuck rather than
        as accurate — which is exactly how it was once reported.
        """
        with self._client(tmp_path, store) as client:
            before = client.get("/api/v1/settings", headers=KEY).json()
            assert before["started_at"] is not None
            client.put("/api/v1/settings", headers=KEY, json={"pipeline": {"digest_threshold": 6}})
            after = client.get("/api/v1/settings", headers=KEY).json()

        assert after["pending_restart"] is True
        # Boot predates the save, so this process cannot be running the change.
        assert after["started_at"] == before["started_at"]
        assert after["started_at"] < after["updated_at"]
        # applied_at is what pending_restart is actually comparing against.
        assert after["applied_at"] != after["updated_at"]

    def test_restarting_clears_the_pending_banner(self, tmp_path, store: MemoryStore) -> None:
        """The second process boots with the overrides, so it marks them applied."""
        with self._client(tmp_path, store) as client:
            client.put("/api/v1/settings", headers=KEY, json={"pipeline": {"digest_threshold": 6}})
            first_boot = client.get("/api/v1/settings", headers=KEY).json()["started_at"]

        with self._client(tmp_path, store) as client:
            after = client.get("/api/v1/settings", headers=KEY).json()

        assert after["started_at"] != first_boot
        assert after["applied_at"] == after["updated_at"]
        assert after["pending_restart"] is False
        # Whether the override reaches `pipeline` is not asserted here: the
        # second boot rebuilds through load_settings(), which reads the real
        # config file and environment rather than this test's constructed
        # settings. Override merging is covered in the settings_store tests.

    def test_incoherent_change_is_rejected_before_it_is_stored(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Otherwise the next restart is when you find out it will not boot."""
        with self._client(tmp_path, store) as client:
            response = client.put(
                "/api/v1/settings",
                headers=KEY,
                json={"pipeline": {"digest_threshold": 9, "top_pick_threshold": 5}},
            )
            assert response.status_code == 400
            assert "top_pick_threshold" in response.json()["detail"]
            # Nothing was persisted.
            assert client.get("/api/v1/settings", headers=KEY).json()["overrides"] == {}

    def test_cloud_endpoint_without_a_key_is_rejected(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            response = client.put(
                "/api/v1/settings",
                headers=KEY,
                json={
                    "tiers": {
                        "tier0": {
                            "primary": {"provider": "openrouter", "model": "x/y"},
                            "timeout_s": 60,
                        },
                        "tier1": {
                            "primary": {"provider": "ollama", "model": "big"},
                            "timeout_s": 300,
                        },
                    }
                },
            )
        assert response.status_code == 400
        assert "OPENROUTER" in response.json()["detail"]

    def test_overrides_are_discardable(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            client.put("/api/v1/settings", headers=KEY, json={"pipeline": {"t_rel_low": 3}})
            client.delete("/api/v1/settings", headers=KEY)
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["overrides"] == {}

    def test_protected_sections_cannot_be_overridden(self) -> None:
        """A typo in a browser must not be able to make the service unreachable
        or unable to find its own database."""
        from podcast_agent.settings_store import OVERRIDABLE_SECTIONS, validate_overrides

        for section in ("couchdb", "api", "security", "output", "scheduler"):
            assert section not in OVERRIDABLE_SECTIONS
            with pytest.raises(ValueError, match="not overridable"):
                validate_overrides({section: {"anything": 1}})


class TestDigestBrowsing:
    """The /admin/digests reading view and the endpoints behind it."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed(
        self,
        store: MemoryStore,
        tmp_path,
        week: str,
        body: str = "# Digest\n",
        generated_at: str | None = None,
    ) -> None:
        relative = f"{week[:4]}/podcast-digest-{week}.md"
        path = Path(make_settings(tmp_path).output.digest_dir) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nweek: {week}\n---\n\n{body}", encoding="utf-8")
        store.seed(
            {
                "_id": digest_doc_id(week),
                "type": "digest",
                "period": {
                    "from": f"{week[:4]}-07-24T00:00:00+00:00",
                    "to": f"{week[:4]}-07-31T00:00:00+00:00",
                },
                "file_path": relative,
                "episode_ids": ["episode:a", "episode:b"],
                "stats": {"total_cost_usd": 0.0},
                "marking_complete": True,
                "generated_at": generated_at or iso_now(),
            }
        )

    def test_endpoints_need_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/digests").status_code == 401
            assert client.get("/api/v1/digests/2026-W31").status_code == 401

    def test_page_is_served_and_self_contained(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/digests").text
        assert "Digests" in body
        external = re.findall(r'(?:src|href)\s*=\s*["\']https?://|@import|url\(\s*https?://', body)
        assert external == [], f"external resource reference: {external}"

    def test_listing_is_newest_week_first(self, tmp_path, store: MemoryStore) -> None:
        for week in ("2026-W29", "2026-W31", "2026-W30"):
            self._seed(store, tmp_path, week)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/digests", headers=KEY).json()
        assert [d["period_key"] for d in body["digests"]] == ["2026-W31", "2026-W30", "2026-W29"]
        assert body["count"] == 3

    def test_regenerating_an_old_week_does_not_move_it_to_the_top(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Ordering is by period, not by generation time — this reads as a calendar."""
        self._seed(
            store,
            tmp_path,
            "2026-W31",
            generated_at=iso(datetime.now(UTC) - timedelta(days=14)),
        )
        # W29 regenerated just now, so it carries the newest generated_at.
        self._seed(store, tmp_path, "2026-W29", generated_at=iso_now())
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/digests", headers=KEY).json()
        assert body["digests"][0]["period_key"] == "2026-W31"

    def test_one_digest_returns_metadata_markdown_and_html(
        self, tmp_path, store: MemoryStore
    ) -> None:
        self._seed(store, tmp_path, "2026-W31", body="# Week 31\n\n**bold**\n")
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/digests/2026-W31", headers=KEY).json()
        assert body["period_key"] == "2026-W31"
        assert body["episodes"] == 2
        assert body["frontmatter"]["week"] == "2026-W31"
        assert body["markdown"].startswith("# Week 31")
        assert "<strong>bold</strong>" in body["html"]
        # Frontmatter is metadata, not body text.
        assert "week: 2026-W31" not in body["markdown"]

    def test_html_from_a_digest_is_sanitised(self, tmp_path, store: MemoryStore) -> None:
        """Digest text descends from LLM output, which descends from feeds."""
        self._seed(store, tmp_path, "2026-W31", body="# Ok\n\n<script>alert(1)</script>\n")
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/digests/2026-W31", headers=KEY).json()
        assert "<script" not in body["html"].lower()

    def test_unknown_week_is_a_404(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/digests/2026-W99", headers=KEY).status_code == 404

    def test_a_deleted_digest_file_is_reported_as_gone(self, tmp_path, store: MemoryStore) -> None:
        """The digest directory belongs to the user, who may prune or move it."""
        self._seed(store, tmp_path, "2026-W31")
        (
            Path(make_settings(tmp_path).output.digest_dir) / "2026/podcast-digest-2026-W31.md"
        ).unlink()
        with self._client(tmp_path, store) as client:
            response = client.get("/api/v1/digests/2026-W31", headers=KEY)
        assert response.status_code == 410
        assert "no digest file" in response.json()["detail"]

    def test_a_file_path_escaping_the_digest_directory_is_refused(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """`file_path` is document data, so it is treated as input."""
        (tmp_path / "secret.md").write_text("password: hunter2", encoding="utf-8")
        store.seed(
            {
                "_id": digest_doc_id("2026-W31"),
                "type": "digest",
                "period": {},
                "file_path": "../../secret.md",
                "episode_ids": [],
                "marking_complete": True,
            }
        )
        with self._client(tmp_path, store) as client:
            response = client.get("/api/v1/digests/2026-W31", headers=KEY)
        assert response.status_code == 410
        assert "hunter2" not in response.text

    def test_no_digests_yet_is_an_empty_list_not_an_error(
        self, tmp_path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/digests", headers=KEY).json()
        assert body == {"count": 0, "digests": []}


class TestNarratingADigest:
    """The console's route into text-to-speech.

    Naming a week is the only way to narrate one that is not the newest — the
    scheduled job cannot, by construction — so this endpoint is what stops the
    archive from being unreachable *and* what stops it from being walked.
    """

    def _client(self, tmp_path, store: MemoryStore, *, tts: bool = True) -> TestClient:
        settings = make_settings(
            tmp_path,
            tts={"enabled": True, "base_url": "http://mac.lan:8880"} if tts else {},
        )
        return TestClient(build_app(settings, store=store, llm=FakeLLM()))

    def _seed(self, store: MemoryStore, week: str = "2026-W31") -> None:
        store.seed(
            {
                "_id": digest_doc_id(week),
                "type": "digest",
                "period": {"from": f"{week[:4]}-07-24T00:00:00+00:00", "to": f"{week[:4]}-07-31"},
                "file_path": f"{week[:4]}/podcast-digest-{week}.md",
                "episode_ids": [],
                "marking_complete": True,
                "generated_at": iso_now(),
            }
        )

    def test_it_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        with self._client(tmp_path, store) as client:
            assert client.post("/api/v1/digests/2026-W31/narrate").status_code == 401

    def test_an_unknown_week_is_a_404(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            response = client.post("/api/v1/digests/1999-W01/narrate", headers=KEY)
        assert response.status_code == 404

    def test_it_is_refused_while_speech_is_off(self, tmp_path, store: MemoryStore) -> None:
        """Rather than starting a job that can only fail at the first request."""
        self._seed(store)
        with self._client(tmp_path, store, tts=False) as client:
            response = client.post("/api/v1/digests/2026-W31/narrate", headers=KEY)
        assert response.status_code == 409
        assert "disabled" in response.json()["detail"]

    def test_a_week_with_nothing_to_read_is_a_409(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        with self._client(tmp_path, store) as client:
            response = client.post("/api/v1/digests/2026-W31/narrate?wait=true", headers=KEY)
        assert response.status_code == 409
        assert "no summarised episodes" in response.json()["detail"]

    def test_the_listing_reports_whether_a_run_has_audio(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The console hides its button on a run that already has a file."""
        self._seed(store)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/digests", headers=KEY).json()
        assert body["digests"][0]["runs"][0]["narration"] is None


class TestActivityConsole:
    """/admin/logs and the endpoints behind it."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_endpoints_need_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/logs").status_code == 401
            assert client.get("/api/v1/runs").status_code == 401
            assert client.get("/api/v1/runs/last").status_code == 401

    def test_page_is_served_and_self_contained(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/logs").text
        assert "Log stream" in body
        external = re.findall(r'(?:src|href)\s*=\s*["\']https?://|@import|url\(\s*https?://', body)
        assert external == [], f"external resource reference: {external}"

    def test_logs_report_what_the_buffer_holds(self, tmp_path, store: MemoryStore) -> None:
        from podcast_agent.logbuffer import buffer

        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/logs?limit=10", headers=KEY).json()
        assert body["capacity"] == buffer.capacity
        assert isinstance(body["events"], list)
        assert "levels" in body

    def test_a_log_line_is_visible_through_the_api(self, tmp_path, store: MemoryStore) -> None:
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging, get_logger

        configure_logging(LoggingConfig(level="INFO", format="json"))
        with self._client(tmp_path, store) as client:
            buffer.clear()
            get_logger("test.api").warning("something_odd", podcast="risky")
            body = client.get("/api/v1/logs?level=warning&limit=20", headers=KEY).json()
        assert any(e.get("event") == "something_odd" for e in body["events"])

    def test_the_text_filter_is_applied_server_side(self, tmp_path, store: MemoryStore) -> None:
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging, get_logger

        configure_logging(LoggingConfig(level="INFO", format="json"))
        with self._client(tmp_path, store) as client:
            buffer.clear()
            get_logger("test.api").info("alpha_event")
            get_logger("test.api").info("beta_event")
            body = client.get("/api/v1/logs?contains=alpha", headers=KEY).json()
        events = [e.get("event") for e in body["events"]]
        assert "alpha_event" in events
        assert "beta_event" not in events

    def test_runs_are_listed_newest_first(self, tmp_path, store: MemoryStore) -> None:
        for i, at in enumerate(("2026-07-01T00:00:00+00:00", "2026-07-30T00:00:00+00:00")):
            store.seed(
                {"_id": f"run:r{i}", "type": "run", "job": "ingest", "at": at, "summary": {}}
            )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/runs", headers=KEY).json()
        assert [r["at"] for r in body["runs"]] == [
            "2026-07-30T00:00:00+00:00",
            "2026-07-01T00:00:00+00:00",
        ]

    def test_runs_can_be_filtered_by_job(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            {"_id": "run:a", "type": "run", "job": "ingest", "at": iso_now(), "summary": {}},
            {"_id": "run:b", "type": "run", "job": "digest", "at": iso_now(), "summary": {}},
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/runs?job=digest", headers=KEY).json()
        assert [r["job"] for r in body["runs"]] == ["digest"]

    def test_last_run_survives_a_restart(self, tmp_path, store: MemoryStore) -> None:
        """The whole point: process memory is empty exactly when you ask.

        The run document is seeded without the process ever having run the job,
        which is precisely the state after a restart.
        """
        store.seed(
            {
                "_id": "run:x",
                "type": "run",
                "job": "ingest",
                "at": "2026-07-30T09:00:00+00:00",
                "summary": {"wall_ms": 4200, "feeds_polled": 14},
            }
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/runs/last", headers=KEY).json()
        ingest = body["jobs"]["ingest"]
        assert ingest["at"] == "2026-07-30T09:00:00+00:00"
        assert ingest["summary"]["feeds_polled"] == 14
        assert ingest["this_process"] is False

    def test_a_job_that_never_ran_reports_null_rather_than_missing(
        self, tmp_path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/runs/last", headers=KEY).json()
        assert set(body["jobs"]) >= {"ingest", "pipeline", "digest", "backfill", "retention"}
        assert all(j["at"] is None for j in body["jobs"].values())

    def test_running_a_job_records_it_durably(self, tmp_path, store: MemoryStore) -> None:
        """A run must leave a trace the next process can read."""
        with self._client(tmp_path, store) as client:
            assert client.post("/api/v1/runs/ingest?wait=true", headers=KEY).status_code == 200
            body = client.get("/api/v1/runs/last", headers=KEY).json()
        assert body["jobs"]["ingest"]["at"] is not None
        assert body["jobs"]["ingest"]["this_process"] is True
        assert [d["job"] for d in store.docs_of_type("run")] == ["ingest"]


class TestEpisodesConsole:
    """Names over identifiers, and where a published episode ended up."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_the_listing_carries_the_podcast_name(self, tmp_path, store: MemoryStore) -> None:
        """The page shows the name; the slug stays available as a tooltip."""
        store.seed(
            make_episode(
                guid="a",
                title="An episode",
                status=S.PUBLISHED,
                published_at=datetime.now(UTC),
            )
        )
        with self._client(tmp_path, store) as client:
            episodes = client.get("/api/v1/episodes", headers=KEY).json()["episodes"]
        assert episodes[0]["podcast_name"]
        assert episodes[0]["podcast_slug"]

    def test_a_published_episode_reports_its_digest(self, tmp_path, store: MemoryStore) -> None:
        """ "Which digest did this end up in?" is answerable from the drawer."""
        store.seed(
            make_episode(
                guid="a",
                title="An episode",
                status=S.PUBLISHED,
                published_at=datetime.now(UTC),
                digest_id="digest:2026-W31",
            )
        )
        with self._client(tmp_path, store) as client:
            listed = client.get("/api/v1/episodes", headers=KEY).json()["episodes"][0]
            detail = client.get(
                f"/api/v1/episodes/{listed['_id'].split(':', 1)[1]}", headers=KEY
            ).json()
        assert detail["digest_id"] == "digest:2026-W31"

    def test_an_unpublished_episode_has_no_digest(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="b",
                title="Waiting",
                status=S.READY_FOR_DIGEST,
                published_at=datetime.now(UTC),
            )
        )
        with self._client(tmp_path, store) as client:
            episodes = client.get("/api/v1/episodes", headers=KEY).json()["episodes"]
        assert episodes[0]["digest_id"] is None

    def test_an_archive_episode_is_not_shown_as_a_weekly_digest(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The claim field holds two different kinds of id.

        Archive material carries `archive:<slug>:2026-06`. Reading the last
        colon-separated part gives "2026-06", which is a month, not an ISO week
        — linking it as one lands on whichever digest happens to be newest and
        presents that as where the episode was published.
        """
        store.seed(
            make_episode(
                guid="c",
                title="From the archive",
                status=S.PUBLISHED,
                published_at=datetime(2026, 6, 15, tzinfo=UTC),
                origin="backfill",
                digest_id="archive:test-show:2026-06",
                archive_month="2026-06",
            )
        )
        with self._client(tmp_path, store) as client:
            listed = client.get("/api/v1/episodes", headers=KEY).json()["episodes"][0]
        assert listed["digest_id"] == "archive:test-show:2026-06"
        assert listed["archive_month"] == "2026-06"

    def test_the_drawer_tells_the_two_kinds_of_claim_apart(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """A prefix check, not a split — see the test above for why."""
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/episodes").text
        assert 'startsWith("digest:")' in body
        assert 'startsWith("archive:")' in body
        assert "not a weekly digest" in body

    def test_the_page_renders_names_not_slugs(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/episodes").text
        assert "e.podcast_name || e.podcast_slug" in body, "table still shows the slug"

    def test_the_drawer_links_to_the_digest(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/episodes").text
        assert "Published in" in body
        assert "/admin/digests?week=" in body

    def test_the_digests_page_honours_a_week_parameter(self, tmp_path, store: MemoryStore) -> None:
        """Otherwise the link from an episode lands on whatever is newest."""
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/digests").text
        assert 'get("week")' in body


class TestHistoricalIntakeConsole:
    """Names over identifiers, and a window that can be widened from the page."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_progress_carries_display_names_not_just_slugs(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """A podcast is chosen by its name; a slug is an implementation detail."""
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/status", headers=KEY).json()
        podcasts = body["backfill"]["podcasts"]
        assert podcasts, "no podcasts reported"
        for entry in podcasts:
            assert entry["name"], f"{entry['slug']} has no display name"
            assert set(entry) >= {"slug", "name", "mode", "cursor", "complete"}
        assert {p["name"] for p in podcasts} == {"Test Show", "Priority Show"}

    def test_each_podcast_carries_its_own_window(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            bf = client.get("/api/v1/status", headers=KEY).json()["backfill"]
        assert bf["window_months_default"] == 12
        assert bf["window_choices"] == [12, 24, 36]
        for entry in bf["podcasts"]:
            assert entry["months"] == 12
            assert entry["months_overridden"] is False
            assert entry["oldest_month_targeted"]

    def test_changing_one_podcast_does_not_change_the_others(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """The point of making it per podcast."""
        with self._client(tmp_path, store) as client:
            assert (
                client.patch(
                    "/api/v1/podcasts/test-show", headers=KEY, json={"backfill_months": 36}
                ).status_code
                == 200
            )
            bf = client.get("/api/v1/status", headers=KEY).json()["backfill"]

        by_slug = {p["slug"]: p for p in bf["podcasts"]}
        assert by_slug["test-show"]["months"] == 36
        assert by_slug["test-show"]["months_overridden"] is True
        assert by_slug["priority-show"]["months"] == 12
        assert by_slug["priority-show"]["months_overridden"] is False
        # The one with the wider window reaches further back.
        assert (
            by_slug["test-show"]["oldest_month_targeted"]
            < by_slug["priority-show"]["oldest_month_targeted"]
        )

    def test_the_podcasts_endpoint_reports_the_effective_window(
        self, tmp_path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"backfill_months": 24})
            podcasts = client.get("/api/v1/podcasts", headers=KEY).json()["podcasts"]
        by_slug = {p["slug"]: p for p in podcasts}
        assert by_slug["test-show"]["backfill_months"] == 24
        assert by_slug["test-show"]["backfill_months_overridden"] is True
        assert "backfill_months" in by_slug["test-show"]["overridden"]
        # Untouched podcasts report the inherited default, not null.
        assert by_slug["priority-show"]["backfill_months"] == 12
        assert by_slug["priority-show"]["backfill_months_overridden"] is False

    @pytest.mark.parametrize("months", [6, 18, 48, 0])
    def test_an_unsupported_window_is_rejected(
        self, tmp_path, store: MemoryStore, months: int
    ) -> None:
        """Each step is roughly another year of archive; a typo must not pick it."""
        with self._client(tmp_path, store) as client:
            response = client.patch(
                "/api/v1/podcasts/test-show", headers=KEY, json={"backfill_months": months}
            )
        assert response.status_code == 422

    def test_reverting_returns_the_podcast_to_the_default(
        self, tmp_path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"backfill_months": 36})
            client.delete("/api/v1/podcasts/test-show/overrides/backfill_months", headers=KEY)
            podcasts = client.get("/api/v1/podcasts", headers=KEY).json()["podcasts"]
        entry = next(p for p in podcasts if p["slug"] == "test-show")
        assert entry["backfill_months"] == 12
        assert entry["backfill_months_overridden"] is False

    def test_the_window_control_is_on_the_podcasts_page(self, tmp_path, store: MemoryStore) -> None:
        """It is a per-podcast setting, so it belongs with the others."""
        with self._client(tmp_path, store) as client:
            body = client.get("/admin/podcasts").text
        assert "backfill_months" in body
        assert "History" in body


class TestRewindEndpoint:
    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_it_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.post("/api/v1/backfill/rewind?confirm=true").status_code == 401

    def test_it_refuses_without_confirmation(self, tmp_path, store: MemoryStore) -> None:
        """It queues hours of local work; a stray click should not start it."""
        with self._client(tmp_path, store) as client:
            response = client.post("/api/v1/backfill/rewind", headers=KEY)
        assert response.status_code == 400
        assert "confirm=true" in response.json()["detail"]

    def test_it_rewinds_and_reports_what_it_touched(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "backfill_cursor": "2025-08",
                "backfill_complete": True,
            }
        )
        with self._client(tmp_path, store) as client:
            body = client.post("/api/v1/backfill/rewind?confirm=true", headers=KEY).json()
            bf = client.get("/api/v1/status", headers=KEY).json()["backfill"]
        assert body["rewound"] == ["test-show"]
        entry = next(p for p in bf["podcasts"] if p["slug"] == "test-show")
        assert entry["cursor"] is None
        assert entry["complete"] is False

    def test_the_button_is_on_the_page(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/backfill").text
        assert 'id="bfRewind"' in page
        assert "Re-walk archive" in page


class TestSpeechSettings:
    """The console half of reading the digest aloud."""

    def _client(self, tmp_path, store: MemoryStore, **tts) -> TestClient:
        settings = make_settings(tmp_path, tts=tts) if tts else make_settings(tmp_path)
        return TestClient(build_app(settings, store=store, llm=FakeLLM()))

    def test_the_current_voice_is_reported(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["tts"]["enabled"] is False
        assert body["tts"]["voice"]
        assert "tts" in body["editable_sections"]

    def test_the_address_is_reported_but_not_editable(self, tmp_path, store: MemoryStore) -> None:
        """Which machine synthesises is deployment topology, like asr.remote_url."""
        with self._client(tmp_path, store, enabled=True, base_url="http://mac.lan:8880") as client:
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["tts_fixed"]["base_url"] == "http://mac.lan:8880"
        assert "base_url" not in body["tts"]

        with self._client(tmp_path, store) as client:
            refused = client.put(
                "/api/v1/settings", headers=KEY, json={"tts": {"base_url": "http://elsewhere"}}
            )
        assert refused.status_code == 400
        assert "not overridable" in refused.json()["detail"]

    def test_the_voice_can_be_changed(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            saved = client.put(
                "/api/v1/settings", headers=KEY, json={"tts": {"voice": "bm_george"}}
            )
            assert saved.status_code == 200, saved.text
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["overrides"]["tts"]["voice"] == "bm_george"

    def test_enabling_it_with_nowhere_to_send_it_is_refused(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Otherwise the failure arrives at the first hourly fire, in a log."""
        with self._client(tmp_path, store) as client:
            response = client.put("/api/v1/settings", headers=KEY, json={"tts": {"enabled": True}})
        assert response.status_code == 400
        assert "base_url" in response.text

    def test_the_page_shows_the_speech_controls(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/settings").text
        assert "Reading the digest aloud" in page
        assert 'id="tts"' in page
        # The two things that surprise people: where the file goes, and that
        # history is not swept up automatically.
        assert "next to the Markdown in your vault" in page
        assert "never narrated automatically" in page


class TestASRSettings:
    """Local transcription was configurable everywhere except the console."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_the_current_asr_configuration_is_reported(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["asr"]["backend"] == "local"
        assert body["asr"]["model"]
        assert body["asr"]["device"]
        assert "asr" in body["editable_sections"]

    def test_machine_protecting_caps_are_reported_but_not_editable(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Size and concurrency limits guard the host, not a preference."""
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["asr_fixed"]["max_audio_mb"]
        assert "max_audio_mb" not in body["asr"]
        assert "asr_concurrency" not in body["asr"]

    def test_whether_faster_whisper_is_installed_is_reported(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Configurable on an install that cannot actually transcribe."""
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert isinstance(body["asr_installed"], bool)

    def test_the_model_can_be_changed(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            response = client.put(
                "/api/v1/settings", headers=KEY, json={"asr": {"model": "medium.en"}}
            )
            assert response.status_code == 200, response.text
            body = client.get("/api/v1/settings", headers=KEY).json()
        assert body["overrides"]["asr"]["model"] == "medium.en"
        assert body["pending_restart"] is True

    def test_a_configuration_that_could_not_boot_is_refused(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """`remote` without a URL fails ASRConfig's own validator."""
        with self._client(tmp_path, store) as client:
            response = client.put(
                "/api/v1/settings", headers=KEY, json={"asr": {"backend": "remote"}}
            )
        assert response.status_code == 400
        assert "remote_url" in response.text

    def test_a_field_outside_the_allowlist_is_refused(self, tmp_path, store: MemoryStore) -> None:
        """A perfectly valid value, for a field the console may not set.

        500 MB is within ASRConfig's own bounds, so this reaches the allowlist
        rather than being stopped by field validation — which is the check being
        tested.
        """
        with self._client(tmp_path, store) as client:
            response = client.put(
                "/api/v1/settings", headers=KEY, json={"asr": {"max_audio_mb": 500}}
            )
        assert response.status_code == 400
        assert "not overridable" in response.json()["detail"]

    def test_the_page_shows_the_asr_controls(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/settings").text
        assert "Local transcription" in page
        assert 'id="asr"' in page
        assert "faster-whisper" in page
        # The three questions the section exists to answer.
        assert "on this machine" in page
        assert "Nothing is sent anywhere" in page


class TestASRAvailabilityWarning:
    """A missing optional extra must announce itself, not stall a queue.

    `faster-whisper` is installed by an extra, and a plain `uv sync` removes
    extras. When it disappeared here, nothing failed loudly: transcript
    acquisition deferred and episodes queued up behind a capability that had
    quietly gone away.
    """

    def _check(self, tmp_path, *, installed: bool, asr_enabled: bool, backend: str = "local"):
        from podcast_agent.main import _warn_if_asr_unavailable
        from podcast_agent.podcasts import PodcastRegistry

        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "T",
                    "feed_url": "https://example.com/feed.xml",
                    "asr_enabled": asr_enabled,
                }
            ],
            asr={"backend": backend},
        )
        registry = PodcastRegistry(settings)

        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        with mock.patch("importlib.util.find_spec", return_value=object() if installed else None):
            _warn_if_asr_unavailable(settings, registry)
        return [e.get("event") for e in buffer.tail(limit=10)]

    def test_it_warns_when_a_podcast_wants_asr_that_cannot_run(self, tmp_path) -> None:
        assert "asr.unavailable" in self._check(tmp_path, installed=False, asr_enabled=True)

    def test_it_is_quiet_when_the_extra_is_installed(self, tmp_path) -> None:
        events = self._check(tmp_path, installed=True, asr_enabled=True)
        assert "asr.unavailable" not in events
        assert "asr.ready" in events

    def test_it_is_quiet_when_no_podcast_wants_asr(self, tmp_path) -> None:
        """Most installs never transcribe; that is not a problem to report."""
        events = self._check(tmp_path, installed=False, asr_enabled=False)
        assert "asr.unavailable" not in events
        assert "asr.ready" not in events

    def test_a_remote_backend_does_not_need_the_local_extra(self, tmp_path) -> None:
        settings_asr = {"backend": "remote"}
        from podcast_agent.main import _warn_if_asr_unavailable
        from podcast_agent.podcasts import PodcastRegistry

        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "T",
                    "feed_url": "https://example.com/feed.xml",
                    "asr_enabled": True,
                }
            ],
            asr={**settings_asr, "remote_url": "https://asr.example/transcribe"},
        )
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        with mock.patch("importlib.util.find_spec", return_value=None):
            _warn_if_asr_unavailable(settings, PodcastRegistry(settings))
        assert "asr.unavailable" not in [e.get("event") for e in buffer.tail(limit=10)]


class TestReadingASpecificRun:
    """?run=N must actually select that run's file.

    The first attempt at this looked right and silently did nothing: the edit
    that introduced run selection never matched its anchor, so every request
    kept returning the most recent file. Only fetching run 1 and run 2 and
    comparing them catches that.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed_two_runs(self, store: MemoryStore, tmp_path) -> None:
        digest_dir = Path(make_settings(tmp_path).output.digest_dir)
        (digest_dir / "2026").mkdir(parents=True, exist_ok=True)
        for name, body in (
            ("podcast-digest-2026-W31.md", "# First run\n"),
            ("podcast-digest-2026-W31-r2.md", "# Second run\n"),
        ):
            (digest_dir / "2026" / name).write_text(
                f"---\nweek: 2026-W31\n---\n\n{body}", encoding="utf-8"
            )
        store.seed(
            {
                "_id": digest_doc_id("2026-W31"),
                "type": "digest",
                "file_path": "2026/podcast-digest-2026-W31-r2.md",
                "period": {"from": "2026-07-30T00:00:00+00:00", "to": "2026-07-31T00:00:00+00:00"},
                "episode_ids": ["episode:b"],
                "marking_complete": True,
                "generated_at": "2026-07-31T00:00:00+00:00",
                "runs": [
                    {
                        "file_path": "2026/podcast-digest-2026-W31.md",
                        "period": {
                            "from": "2026-07-23T00:00:00+00:00",
                            "to": "2026-07-30T00:00:00+00:00",
                        },
                        "episode_ids": ["episode:a", "episode:a2"],
                        "generated_at": "2026-07-30T00:00:00+00:00",
                    },
                    {
                        "file_path": "2026/podcast-digest-2026-W31-r2.md",
                        "period": {
                            "from": "2026-07-30T00:00:00+00:00",
                            "to": "2026-07-31T00:00:00+00:00",
                        },
                        "episode_ids": ["episode:b"],
                        "generated_at": "2026-07-31T00:00:00+00:00",
                    },
                ],
            }
        )

    def test_each_run_returns_its_own_file(self, tmp_path, store: MemoryStore) -> None:
        self._seed_two_runs(store, tmp_path)
        with self._client(tmp_path, store) as client:
            first = client.get("/api/v1/digests/2026-W31?run=1", headers=KEY).json()
            second = client.get("/api/v1/digests/2026-W31?run=2", headers=KEY).json()
        assert "First run" in first["markdown"]
        assert "Second run" in second["markdown"]
        assert first["file_path"] != second["file_path"]

    def test_each_run_reports_its_own_period_and_count(self, tmp_path, store: MemoryStore) -> None:
        """They are two digests, not two versions of one."""
        self._seed_two_runs(store, tmp_path)
        with self._client(tmp_path, store) as client:
            first = client.get("/api/v1/digests/2026-W31?run=1", headers=KEY).json()
        assert first["from"].startswith("2026-07-23")
        assert first["episodes"] == 2
        assert first["run"] == 1

    def test_no_run_given_returns_the_most_recent(self, tmp_path, store: MemoryStore) -> None:
        self._seed_two_runs(store, tmp_path)
        with self._client(tmp_path, store) as client:
            latest = client.get("/api/v1/digests/2026-W31", headers=KEY).json()
        assert "Second run" in latest["markdown"]
        assert latest["run"] == 2

    def test_a_run_that_does_not_exist_is_a_404(self, tmp_path, store: MemoryStore) -> None:
        self._seed_two_runs(store, tmp_path)
        with self._client(tmp_path, store) as client:
            response = client.get("/api/v1/digests/2026-W31?run=9", headers=KEY)
        assert response.status_code == 404
        assert "2 run(s)" in response.json()["detail"]

    def test_the_listing_reports_every_run(self, tmp_path, store: MemoryStore) -> None:
        self._seed_two_runs(store, tmp_path)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/digests", headers=KEY).json()
        runs = body["digests"][0]["runs"]
        assert [r["run"] for r in runs] == [1, 2]
        assert runs[0]["episodes"] == 2 and runs[1]["episodes"] == 1


class TestWhatIsBlockingTheArchive:
    """A bare status reads as a fault when most of these are ordinary steps.

    TRANSCRIPT_FAILED especially: it is the expected outcome for a podcast that
    publishes no transcript with local transcription off, and the episode still
    gets a summary from its description. Shown as "TRANSCRIPT_FAILED" beside
    "3 episodes are still in the pipeline" it reads as breakage holding up the
    archive.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_each_blocking_episode_says_what_happens_next(
        self, tmp_path, store: MemoryStore
    ) -> None:
        store.seed(
            make_episode(
                guid="no-transcript",
                title="No transcript published",
                status=S.TRANSCRIPT_FAILED,
                published_at=datetime.now(UTC),
            )
        )
        with self._client(tmp_path, store) as client:
            waiting = client.get("/api/v1/status", headers=KEY).json()["backfill"][
                "waiting_on_recent"
            ]
        entry = waiting["episodes"][0]
        assert entry["status"] == "TRANSCRIPT_FAILED"
        assert "summarised from its description" in entry["next_step"]

    @pytest.mark.parametrize(
        "status",
        [S.NEW, S.TRIAGED, S.AWAITING_TRANSCRIPT, S.TRANSCRIBED, S.TRANSCRIPT_FAILED, S.SUMMARIZED],
    )
    def test_every_blocking_state_has_a_plain_explanation(
        self, tmp_path, store: MemoryStore, status
    ) -> None:
        """Whatever holds the archive back, the page can say why."""
        store.seed(make_episode(guid="x", title="X", status=status, published_at=datetime.now(UTC)))
        with self._client(tmp_path, store) as client:
            waiting = client.get("/api/v1/status", headers=KEY).json()["backfill"][
                "waiting_on_recent"
            ]
        entry = waiting["episodes"][0]
        assert entry["next_step"] != "waiting", f"{status.value} has no explanation"

    def test_a_transcript_failure_still_blocks_the_archive(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """It is unfinished work, not finished-badly work: a summary is still owed.

        Deliberate — but it means the archive waits on it, so it must drain,
        which the summarise stage does by writing a description-only summary.
        """
        store.seed(
            make_episode(
                guid="x",
                title="X",
                status=S.TRANSCRIPT_FAILED,
                published_at=datetime.now(UTC),
            )
        )
        with self._client(tmp_path, store) as client:
            waiting = client.get("/api/v1/status", headers=KEY).json()["backfill"][
                "waiting_on_recent"
            ]
        assert waiting["blocked"] is True

    def test_a_settled_episode_does_not_block(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="done",
                title="Done",
                status=S.SCORED_LOW,
                published_at=datetime.now(UTC),
            )
        )
        with self._client(tmp_path, store) as client:
            waiting = client.get("/api/v1/status", headers=KEY).json()["backfill"][
                "waiting_on_recent"
            ]
        assert waiting["blocked"] is False


class TestDatabaseUnavailable:
    """A database blip should report itself, not look like a console bug."""

    def test_a_store_failure_becomes_a_503_not_a_500(self, tmp_path) -> None:
        """Unhandled it was "Exception in ASGI application" plus a traceback."""
        from podcast_agent.db.base import StoreError

        class Broken(MemoryStore):
            # Fails only once serving requests. Breaking `find` outright would
            # raise during the lifespan instead — startup runs migrations and
            # refreshes the registry — and never reach the handler under test.
            failing = False

            async def find(self, *a, **k):  # type: ignore[no-untyped-def]
                if self.failing:
                    raise StoreError("CouchDB POST /_find failed: connection reset")
                return await super().find(*a, **k)

        store = Broken()
        client = TestClient(
            build_app(make_settings(tmp_path), store=store, llm=FakeLLM()),
            raise_server_exceptions=False,
        )
        with client:
            store.failing = True
            response = client.get("/api/v1/episodes", headers=KEY)
        assert response.status_code == 503
        assert "database unavailable" in response.json()["detail"]

    def test_healthz_still_answers_when_the_database_is_down(self, tmp_path) -> None:
        """The probe exists to report this, so it must not fail with it."""

        class Down(MemoryStore):
            async def ping(self) -> bool:
                return False

        client = TestClient(
            build_app(make_settings(tmp_path), store=Down(), llm=FakeLLM()),
            raise_server_exceptions=False,
        )
        with client:
            response = client.get("/healthz")
        assert response.status_code == 503
        assert response.json()["detail"]["couchdb"] == "unreachable"


class TestQueueSaysWhatMovesIt:
    """Two jobs move the queue and each ignores the other's half.

    A queue of 82 that "Process queue" leaves untouched is not a stalled
    pipeline — it is 82 archive episodes waiting on the archive walk, which the
    routine pipeline deliberately never touches. One combined number said
    nothing about that.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_the_two_halves_are_reported_separately(self, tmp_path, store: MemoryStore) -> None:
        for i in range(3):
            store.seed(
                make_episode(
                    guid=f"a{i}",
                    status=S.NEW,
                    published_at=datetime.now(UTC),
                    origin=BACKFILL_ORIGIN,
                )
            )
        store.seed(make_episode(guid="r", status=S.NEW, published_at=datetime.now(UTC)))

        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/status", headers=KEY).json()

        assert body["queue_depths"]["triage"] == 4
        assert body["queue_depths_routine"]["triage"] == 1

    def test_an_all_archive_queue_shows_no_routine_work(self, tmp_path, store: MemoryStore) -> None:
        """Exactly the case that looked stuck: nothing for Process queue to do."""
        store.seed(
            make_episode(
                guid="a", status=S.TRIAGED, published_at=datetime.now(UTC), origin=BACKFILL_ORIGIN
            )
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/status", headers=KEY).json()

        assert body["queue_depths"]["dispatch"] == 1
        assert body["queue_depths_routine"]["dispatch"] == 0

    def test_the_page_says_which_job_moves_the_archive_half(
        self, tmp_path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            page = client.get("/admin").text
        assert "archive episodes, moved by" in page
        assert "not by Process queue" in page
        # And the button itself no longer claims to move everything.
        assert "Archive episodes are not touched" in page


class TestStoredWarnings:
    """The /admin/logs "Kept warnings" tab and the endpoint behind it."""

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed(self, store: MemoryStore, **over) -> None:
        store.seed(
            {
                "_id": f"log:{over.get('event', 'e')}-{over.get('at', '1')}",
                "type": "log",
                "level": "warning",
                "event": "something_odd",
                "logger": "podcast_agent.x",
                "at": iso_now(),
                "occurrences": 1,
                **over,
            }
        )

    def test_it_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/logs/stored").status_code == 401

    def test_stored_warnings_survive_a_restart(self, tmp_path, store: MemoryStore) -> None:
        """The whole point: the in-memory tail is empty after a restart."""
        self._seed(store, event="scheduler.job_failed", level="error")
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/logs/stored", headers=KEY).json()
        assert [e["event"] for e in body["events"]] == ["scheduler.job_failed"]
        assert body["retention_days"] == 30

    def test_newest_first(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, event="older", at="2026-07-01T00:00:00+00:00")
        self._seed(store, event="newer", at="2026-07-30T00:00:00+00:00")
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/logs/stored", headers=KEY).json()
        assert [e["event"] for e in body["events"]] == ["newer", "older"]

    def test_it_can_be_filtered_by_level(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, event="a_warning", level="warning")
        self._seed(store, event="an_error", level="error")
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/logs/stored?level=error", headers=KEY).json()
        assert [e["event"] for e in body["events"]] == ["an_error"]

    def test_it_can_be_filtered_by_event_name(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, event="couchdb.request_retry")
        self._seed(store, event="transcript.attempt_failed")
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/logs/stored?contains=couchdb", headers=KEY).json()
        assert [e["event"] for e in body["events"]] == ["couchdb.request_retry"]

    def test_internal_document_fields_are_not_exposed(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/logs/stored", headers=KEY).json()
        entry = body["events"][0]
        assert "_id" not in entry and "_rev" not in entry and "type" not in entry

    def test_nothing_stored_is_not_an_error(self, tmp_path, store: MemoryStore) -> None:
        """The empty case is the good outcome, not a failure."""
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/logs/stored", headers=KEY).json()
        assert body["count"] == 0
        assert body["events"] == []

    def test_the_page_has_the_tab(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/logs").text
        assert 'data-tab="kept"' in page
        assert "Kept warnings" in page
        # Says plainly what it does and does not hold.
        assert "not duplicated here" in page


class TestScoreFilter:
    """Filtering by the number the table shows.

    "The score" is Tier-1 where it exists and the triage guess otherwise, which
    is not something a Mango selector can express — an `$or` across both fields
    would match an episode whose triage guessed 9 and whose summary then scored
    3. So it is computed once, server-side, and shared by the table, the filter
    and the API.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed(self, store: MemoryStore, guid: str, *, final=None, guess=None) -> None:
        store.seed(
            make_episode(
                guid=guid,
                title=guid,
                status=S.READY_FOR_DIGEST,
                published_at=datetime.now(UTC),
                tier0={"relevance_guess": guess, "confidence": 8} if guess is not None else None,
                tier1={"relevance_score": final} if final is not None else None,
            )
        )

    def test_the_view_reports_the_score_the_table_shows(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, "summarised", final=8, guess=5)
        with self._client(tmp_path, store) as client:
            entry = client.get("/api/v1/episodes", headers=KEY).json()["episodes"][0]
        assert entry["score"] == 8
        assert entry["score_provisional"] is False

    def test_a_grey_zone_episode_scores_by_its_triage_guess(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Never summarised, so the guess is the only number it has."""
        self._seed(store, "greyzone", guess=4)
        with self._client(tmp_path, store) as client:
            entry = client.get("/api/v1/episodes", headers=KEY).json()["episodes"][0]
        assert entry["score"] == 4
        assert entry["score_provisional"] is True

    def test_the_final_score_wins_over_the_guess(self, tmp_path, store: MemoryStore) -> None:
        """The case an `$or` selector would get wrong."""
        self._seed(store, "overrated", final=3, guess=9)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?min_score=7", headers=KEY).json()
        assert body["total"] == 0, "matched on the triage guess it later disproved"

    def test_it_filters_on_the_effective_score(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, "high", final=9)
        self._seed(store, "mid", final=6)
        self._seed(store, "guessed-high", guess=8)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?min_score=7", headers=KEY).json()
        assert {e["title"] for e in body["episodes"]} == {"high", "guessed-high"}
        assert body["total"] == 2

    def test_an_unscored_episode_is_excluded(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            make_episode(guid="raw", title="raw", status=S.NEW, published_at=datetime.now(UTC))
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?min_score=1", headers=KEY).json()
        assert body["total"] == 0

    def test_it_combines_with_the_other_filters(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, "keep", final=9)
        store.seed(
            make_episode(
                guid="other",
                title="other",
                status=S.READY_FOR_DIGEST,
                slug="priority-show",
                published_at=datetime.now(UTC),
                tier1={"relevance_score": 9},
            )
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?min_score=7&podcast=test-show", headers=KEY).json()
        assert {e["title"] for e in body["episodes"]} == {"keep"}

    def test_it_pages(self, tmp_path, store: MemoryStore) -> None:
        for i in range(5):
            self._seed(store, f"ep{i}", final=9)
        with self._client(tmp_path, store) as client:
            first = client.get("/api/v1/episodes?min_score=7&limit=2", headers=KEY).json()
            second = client.get("/api/v1/episodes?min_score=7&limit=2&skip=2", headers=KEY).json()
        assert first["total"] == second["total"] == 5
        assert len(first["episodes"]) == len(second["episodes"]) == 2
        assert {e["title"] for e in first["episodes"]}.isdisjoint(
            {e["title"] for e in second["episodes"]}
        )

    def test_no_filter_keeps_the_cheap_path(self, tmp_path, store: MemoryStore) -> None:
        """Without a score floor the database still does the paging."""
        self._seed(store, "a", final=9)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes", headers=KEY).json()
        assert "truncated" not in body

    def test_it_filters_the_low_end_too(self, tmp_path, store: MemoryStore) -> None:
        """ "What did triage reject?" is as real a question as "what is good?"."""
        self._seed(store, "rejected", guess=2)
        self._seed(store, "greyzone", guess=5)
        self._seed(store, "good", final=9)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?min_score=0&max_score=3", headers=KEY).json()
        assert {e["title"] for e in body["episodes"]} == {"rejected"}

    def test_it_filters_a_band(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store, "low", guess=3)
        self._seed(store, "middling", guess=5)
        self._seed(store, "high", final=9)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?min_score=4&max_score=6", headers=KEY).json()
        assert {e["title"] for e in body["episodes"]} == {"middling"}

    def test_an_unjudged_episode_is_not_a_low_score(self, tmp_path, store: MemoryStore) -> None:
        """Nothing has looked at it yet, which is not the same as being rejected."""
        store.seed(
            make_episode(guid="raw", title="raw", status=S.NEW, published_at=datetime.now(UTC))
        )
        self._seed(store, "rejected", guess=1)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?min_score=0&max_score=3", headers=KEY).json()
        assert {e["title"] for e in body["episodes"]} == {"rejected"}

    def test_the_page_offers_the_filter_and_keeps_the_pager_together(
        self, tmp_path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/episodes").text
        assert 'id="fScore"' in page
        assert "min_score" in page and "max_score" in page
        # The low end is reachable, not just floors.
        assert "below 4" in page
        # Prev, Next and the count wrap as one unit rather than splitting.
        pager = page[page.index('<span class="pager">') :]
        assert pager.index('id="epPrev"') < pager.index('id="epNext"') < pager.index('id="epCount"')
        assert "white-space: nowrap" in page


class TestTranscriptionTelemetry:
    """Local transcription, recorded durably and reported in its own units.

    19 episodes had been transcribed with nothing kept about any of them: how
    long the archive really took, which podcast is expensive, whether the model
    suits the machine — all unanswerable an hour later, while 1,163 llm_call
    rows answered exactly those questions for the model side.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _run(
        self,
        store: MemoryStore,
        guid: str,
        *,
        audio: int,
        elapsed: float,
        slug: str = "test-show",
        model: str = "small.en",
    ) -> None:
        store.seed(
            {
                "_id": f"asrrun:{guid}",
                "type": "asr_run",
                "episode_id": f"episode:{guid}",
                "podcast_slug": slug,
                "model": model,
                "device": "cpu",
                "audio_duration_s": audio,
                "elapsed_s": elapsed,
                "realtime_factor": round(audio / elapsed, 2),
                "ts": iso_now(),
            }
        )

    def test_transcription_is_reported_separately_from_calls(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Folding it into `calls` would average minutes against seconds."""
        self._run(store, "a", audio=3600, elapsed=900)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/telemetry/costs?days=30", headers=KEY).json()
        assert body["asr"]["runs"] == 1
        # The model-call totals are untouched by it.
        assert body["totals"]["calls"] == 0

    def test_it_reports_the_realtime_factor(self, tmp_path, store: MemoryStore) -> None:
        """The number that says whether the model and machine suit each other."""
        self._run(store, "a", audio=3600, elapsed=900)
        with self._client(tmp_path, store) as client:
            asr = client.get("/api/v1/telemetry/costs", headers=KEY).json()["asr"]
        assert asr["realtime_factor"] == 4.0
        assert asr["audio_hours"] == 1.0
        assert asr["compute_hours"] == 0.25

    def test_it_groups_by_model_and_by_podcast(self, tmp_path, store: MemoryStore) -> None:
        self._run(store, "a", audio=3600, elapsed=900, slug="test-show")
        self._run(store, "b", audio=1800, elapsed=900, slug="priority-show", model="medium.en")
        with self._client(tmp_path, store) as client:
            asr = client.get("/api/v1/telemetry/costs", headers=KEY).json()["asr"]
        assert set(asr["by_model"]) == {"small.en on cpu", "medium.en on cpu"}
        assert set(asr["by_podcast"]) == {"test-show", "priority-show"}
        # Slower model, lower factor — the comparison the grouping exists for.
        assert asr["by_model"]["medium.en on cpu"]["realtime_factor"] == 2.0

    def test_nothing_transcribed_is_not_an_error(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            asr = client.get("/api/v1/telemetry/costs", headers=KEY).json()["asr"]
        assert asr["runs"] == 0
        assert asr["realtime_factor"] is None

    def test_the_tab_shows_both_halves(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            page = client.get("/admin/logs").text
        assert "Model work" in page
        assert "Local transcription" in page
        assert 'id="asrRows"' in page
        assert "realtime factor" in page.lower()


class TestFeedHealthCoversEveryPodcast:
    """Feed health read config.yaml, which is only half the list.

    A podcast added in the console lives in the database and nowhere else, so
    the panel reported 14 healthy feeds while 16 were being polled — and a
    failing feed among the other seven could never have appeared there.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_a_console_added_podcast_is_counted(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            before = len(client.get("/api/v1/status", headers=KEY).json()["feeds"])
            assert (
                client.post(
                    "/api/v1/podcasts",
                    headers=KEY,
                    json={
                        "slug": "added-here",
                        "name": "Added Here",
                        "feed_url": "https://added.example.com/feed.xml",
                    },
                ).status_code
                == 201
            )
            feeds = client.get("/api/v1/status", headers=KEY).json()["feeds"]

        assert len(feeds) == before + 1
        assert "added-here" in {f["slug"] for f in feeds}

    def test_its_failures_are_visible(self, tmp_path, store: MemoryStore) -> None:
        """The point of the panel: a broken feed has to be able to show up."""
        with self._client(tmp_path, store) as client:
            client.post(
                "/api/v1/podcasts",
                headers=KEY,
                json={
                    "slug": "broken",
                    "name": "Broken",
                    "feed_url": "https://broken.example.com/feed.xml",
                },
            )
            doc = next(d for d in store.docs_of_type("podcast") if d.get("slug") == "broken")
            doc.update({"consecutive_failures": 6, "last_error": "410 Gone"})
            store.seed(doc)
            feeds = client.get("/api/v1/status", headers=KEY).json()["feeds"]

        broken = next(f for f in feeds if f["slug"] == "broken")
        assert broken["consecutive_failures"] == 6
        assert broken["circuit_open"] is True

    def test_a_disabled_podcast_is_not_counted(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            client.patch("/api/v1/podcasts/test-show", headers=KEY, json={"enabled": False})
            feeds = client.get("/api/v1/status", headers=KEY).json()["feeds"]
        assert "test-show" not in {f["slug"] for f in feeds}


class TestFilteringByWhetherThereIsASummary:
    """ "Published" does not mean "summarised".

    The grey zone and the archive both list episodes without one, so a reader
    looking for something to read has to be able to separate the two — and,
    conversely, to find the ones worth asking for a summary of.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed(self, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="written",
                title="written",
                status=S.PUBLISHED,
                tier1={
                    "relevance_score": 8,
                    "summary_md": "the summary",
                    "summary_basis": "transcript",
                },
            ),
            make_episode(
                guid="listed",
                title="listed",
                status=S.PUBLISHED,
                digest_id="archive:test-show:2026-02",
                tier0={"relevance_guess": 7, "confidence": 9, "route": "ESCALATE"},
            ),
            make_episode(guid="fresh", title="fresh", status=S.NEW),
        )

    def _titles(self, tmp_path, store: MemoryStore, query: str) -> set[str]:
        with self._client(tmp_path, store) as client:
            body = client.get(f"/api/v1/episodes?{query}", headers=KEY).json()
        return {e["title"] for e in body["episodes"]}

    def test_it_finds_the_ones_with_a_summary(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        assert self._titles(tmp_path, store, "summarised=true") == {"written"}

    def test_it_finds_the_ones_without(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        assert self._titles(tmp_path, store, "summarised=false") == {"listed", "fresh"}

    def test_omitting_it_filters_nothing(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        assert len(self._titles(tmp_path, store, "limit=50")) == 3

    def test_an_empty_summary_does_not_count_as_one(self, tmp_path, store: MemoryStore) -> None:
        """A tier1 block can exist while the summary itself is empty — the same
        definition the table renders and the drawer's guard protects."""
        store.seed(
            make_episode(
                guid="hollow", title="hollow", status=S.SCORED_LOW, tier1={"relevance_score": 2}
            )
        )
        assert self._titles(tmp_path, store, "summarised=false") == {"hollow"}

    def test_the_count_describes_the_filtered_set(self, tmp_path, store: MemoryStore) -> None:
        """The pager reads `total`. Reporting the unfiltered count there would
        offer pages that render empty."""
        self._seed(store)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/episodes?summarised=false", headers=KEY).json()
        assert body["total"] == 2
        assert len(body["episodes"]) == 2

    def test_it_combines_with_another_filter(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        found = self._titles(tmp_path, store, "summarised=false&status=PUBLISHED")
        assert found == {"listed"}


class TestTheReadyCountMeansWhatItSays:
    """ "Ready for the next digest" must be a number that can reach zero.

    An episode summarised on request after it was already listed keeps its
    claim, so nothing will ever pick it up again. Counting it as awaiting is
    the same lie as labelling a published archive episode "queued": work
    implied on a decision already taken, in a number that never drains.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _depths(self, tmp_path, store: MemoryStore) -> dict[str, Any]:
        with self._client(tmp_path, store) as client:
            return client.get("/api/v1/status", headers=KEY).json()

    def test_an_unclaimed_episode_is_awaiting_a_digest(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            make_episode(guid="free", status=S.READY_FOR_DIGEST, digest_id=None),
            make_episode(guid="grey", status=S.DIGEST_DIRECT, digest_id=None),
        )
        assert self._depths(tmp_path, store)["queue_depths"]["awaiting_digest"] == 2

    def test_an_episode_already_written_somewhere_is_not(
        self, tmp_path, store: MemoryStore
    ) -> None:
        store.seed(
            make_episode(guid="free", status=S.READY_FOR_DIGEST, digest_id=None),
            make_episode(
                guid="listed",
                status=S.READY_FOR_DIGEST,
                digest_id="archive:test-show:2026-06",
            ),
        )
        assert self._depths(tmp_path, store)["queue_depths"]["awaiting_digest"] == 1

    def test_the_routine_count_excludes_it_too(self, tmp_path, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="listed",
                status=S.READY_FOR_DIGEST,
                digest_id="archive:test-show:2026-06",
            )
        )
        depths = self._depths(tmp_path, store)
        assert depths["queue_depths"]["awaiting_digest"] == 0
        assert depths["queue_depths_routine"]["awaiting_digest"] == 0


class TestRepeatedFailuresAreThrottled:
    """Constant-time comparison defeats timing, not volume.

    This listens on the LAN with nothing in front of it, so one compromised
    device on the network can sit and guess at whatever rate it likes.
    """

    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_a_wrong_key_is_still_just_unauthorised_at_first(
        self, tmp_path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            for _ in range(MAX_FAILURES):
                assert client.get("/api/v1/status", headers={"X-API-Key": "no"}).status_code == 401

    def test_the_attempt_after_the_limit_is_refused(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            for _ in range(MAX_FAILURES):
                client.get("/api/v1/status", headers={"X-API-Key": "no"})
            refused = client.get("/api/v1/status", headers={"X-API-Key": "no"})
        assert refused.status_code == 429
        assert refused.headers["Retry-After"] == str(WINDOW_S)

    def test_the_right_key_is_refused_too_once_throttled(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Otherwise the throttle is an oracle: it would say "that one was
        different" about exactly the guess that mattered."""
        with self._client(tmp_path, store) as client:
            for _ in range(MAX_FAILURES):
                client.get("/api/v1/status", headers={"X-API-Key": "no"})
            assert client.get("/api/v1/status", headers=KEY).status_code == 429

    def test_the_refusal_says_nothing_about_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            for _ in range(MAX_FAILURES):
                client.get("/api/v1/status", headers={"X-API-Key": "no"})
            body = client.get("/api/v1/status", headers={"X-API-Key": "no"}).json()
        assert "too many failed attempts" in body["detail"]
        for leak in ("close", "correct", "length", "test-admin-key"):
            assert leak not in body["detail"]

    def test_getting_it_right_clears_the_count(self, tmp_path, store: MemoryStore) -> None:
        """A person who mistypes twice and then succeeds is not mid-attack."""
        with self._client(tmp_path, store) as client:
            for _ in range(MAX_FAILURES - 1):
                client.get("/api/v1/status", headers={"X-API-Key": "no"})
            assert client.get("/api/v1/status", headers=KEY).status_code == 200
            for _ in range(MAX_FAILURES - 1):
                assert client.get("/api/v1/status", headers={"X-API-Key": "no"}).status_code == 401

    def test_failures_are_forgotten_once_the_window_passes(
        self, tmp_path, store: MemoryStore, monkeypatch
    ) -> None:
        clock = {"t": 1000.0}
        monkeypatch.setattr(auth, "time", type("_", (), {"monotonic": lambda: clock["t"]}))
        with self._client(tmp_path, store) as client:
            for _ in range(MAX_FAILURES):
                client.get("/api/v1/status", headers={"X-API-Key": "no"})
            assert client.get("/api/v1/status", headers={"X-API-Key": "no"}).status_code == 429
            clock["t"] += WINDOW_S + 1
            assert client.get("/api/v1/status", headers={"X-API-Key": "no"}).status_code == 401

    def test_one_address_does_not_throttle_another(self, tmp_path, store: MemoryStore) -> None:
        app = build_app(make_settings(tmp_path), store=store, llm=FakeLLM())
        with TestClient(app, client=("10.0.0.1", 1234)) as noisy:
            for _ in range(MAX_FAILURES + 1):
                noisy.get("/api/v1/status", headers={"X-API-Key": "no"})
        with TestClient(app, client=("10.0.0.2", 1234)) as other:
            assert other.get("/api/v1/status", headers=KEY).status_code == 200

    def test_health_is_never_throttled(self, tmp_path, store: MemoryStore) -> None:
        """It has no key to get wrong, and a monitor must not be locked out."""
        with self._client(tmp_path, store) as client:
            for _ in range(MAX_FAILURES + 1):
                client.get("/api/v1/status", headers={"X-API-Key": "no"})
            assert client.get("/healthz").status_code in (200, 503)

    def test_the_tracked_set_cannot_grow_without_limit(self) -> None:
        """Its keys are chosen by whoever connects, so it is attacker-sized."""
        auth.reset_throttle()
        for i in range(auth.MAX_TRACKED + 50):
            auth._record_failure(f"10.1.{i // 256}.{i % 256}", 1000.0)
        assert len(auth._failures) == auth.MAX_TRACKED
