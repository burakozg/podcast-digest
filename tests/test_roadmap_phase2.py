"""Roadmap phase 2: A2 archive-aware prompts, C2 profile versioning + re-score,
E4 notifications, E2 glance endpoint.

Backup/restore (F5) is shell plus CouchDB and is verified by a real round-trip
against a live database rather than here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.config import NotificationConfig
from podcast_agent.db import MemoryStore, save_transcript
from podcast_agent.llm.prompts import load_prompt
from podcast_agent.main import build_app
from podcast_agent.models import Tier1Result
from podcast_agent.notify import Notifier
from podcast_agent.state import EpisodeStatus
from podcast_agent.summarize.tier1 import PROMPT_VERSIONS, Tier1Stage
from podcast_agent.utils import describe_age, episode_age_days

S = EpisodeStatus
KEY = {"X-API-Key": "test-admin-key"}
LONG_TEXT = "A substantive transcript sentence about ICS security. " * 40


# --- A2: archive-aware summarisation ----------------------------------------


class TestEpisodeAge:
    def test_age_in_days(self) -> None:
        now = datetime(2026, 7, 30, tzinfo=UTC)
        published = (now - timedelta(days=45)).isoformat()
        assert episode_age_days(published, now=now) == 45

    def test_unknown_date(self) -> None:
        assert episode_age_days(None) is None
        assert episode_age_days("not a date") is None

    def test_future_dates_clamp_to_zero(self) -> None:
        """Feeds do publish dates in the future; a negative age would read oddly."""
        now = datetime(2026, 7, 30, tzinfo=UTC)
        ahead = (now + timedelta(days=3)).isoformat()
        assert episode_age_days(ahead, now=now) == 0

    @pytest.mark.parametrize(
        ("days", "expected_fragment"),
        [
            (0, "last day"),
            (5, "5 days ago"),
            (21, "3 weeks ago"),
            (200, "months ago"),
            (800, "years ago"),
        ],
    )
    def test_phrasing(self, days: int, expected_fragment: str) -> None:
        assert expected_fragment in describe_age(days)

    def test_only_old_episodes_are_labelled_archive(self) -> None:
        """The archive label is the cue the prompt keys off, so it must not fire
        for an episode that is merely a few weeks old."""
        assert "archive" not in describe_age(20)
        assert "archive" not in describe_age(60)
        assert "archive" in describe_age(200)
        assert "archive" in describe_age(900)

    def test_unknown_age_is_empty_not_misleading(self) -> None:
        assert describe_age(None) == ""


class TestArchiveAwarePrompts:
    def test_tier1_uses_v2(self) -> None:
        assert PROMPT_VERSIONS["tier1"] == "v2"
        assert PROMPT_VERSIONS["tier1_reduce"] == "v2"

    def test_unchanged_map_prompt_keeps_v1(self) -> None:
        """An unchanged prompt keeps its version, so telemetry stays attributable."""
        assert PROMPT_VERSIONS["tier1_map"] == "v1"

    def test_v1_prompts_are_still_present(self) -> None:
        """Shipped versions are never edited in place — v1 must remain loadable."""
        assert load_prompt("tier1", "v1").system
        assert load_prompt("tier1_reduce", "v1").system

    @pytest.mark.parametrize("name", ["tier1", "tier1_reduce"])
    def test_v2_instructs_historical_framing(self, name: str) -> None:
        system = load_prompt(name, "v2").system
        assert "publication date" in system
        assert "at the time of recording" in system

    async def test_age_reaches_the_prompt(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        old = datetime.now(UTC) - timedelta(days=400)
        episode = make_episode(status=S.TRANSCRIBED, published_at=old)
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)

        await Tier1Stage(settings, store, fake_llm).summarize(
            (await store.get(episode["_id"])) or {}
        )

        user_prompt = fake_llm.calls[0]["user"]
        assert "archive episode" in user_prompt
        assert fake_llm.calls[0]["prompt_version"] == "tier1_v2"

    async def test_recent_episode_is_not_called_an_archive(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        episode = make_episode(
            status=S.TRANSCRIBED, published_at=datetime.now(UTC) - timedelta(days=2)
        )
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)
        await Tier1Stage(settings, store, fake_llm).summarize(
            (await store.get(episode["_id"])) or {}
        )
        assert "archive episode" not in fake_llm.calls[0]["user"]


# --- C2: profile versioning and re-score -------------------------------------


class TestProfileVersion:
    def test_stable_across_reordering(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        before = settings.interest_profile_version()
        settings.interest_profile.reverse()
        assert settings.interest_profile_version() == before

    @pytest.mark.parametrize("field", ["weight", "description", "label", "key"])
    def test_changes_when_any_prompt_visible_field_changes(
        self, tmp_path: Path, field: str
    ) -> None:
        """Every one of these reaches the prompt, so each can move a score."""
        settings = make_settings(tmp_path)
        before = settings.interest_profile_version()
        item = settings.interest_profile[0]
        setattr(item, field, 3 if field == "weight" else "changed_value")
        assert settings.interest_profile_version() != before

    def test_version_is_short_and_stable_in_shape(self, tmp_path: Path) -> None:
        version = make_settings(tmp_path).interest_profile_version()
        assert len(version) == 12 and version.isalnum()

    async def test_stamped_on_tier1(self, settings, store: MemoryStore, fake_llm: FakeLLM) -> None:
        episode = make_episode(status=S.TRANSCRIBED)
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)
        await Tier1Stage(settings, store, fake_llm).summarize(
            (await store.get(episode["_id"])) or {}
        )
        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["tier1"]["profile_version"] == settings.interest_profile_version()


class TestRescore:
    def _runner(self, settings, store, llm):
        from podcast_agent.digest.generate import DigestGenerator
        from podcast_agent.ingest.feeds import Ingestor
        from podcast_agent.net import UrlGuard, build_client
        from podcast_agent.pipeline.runner import PipelineRunner
        from podcast_agent.transcripts.acquire import TranscriptAcquirer
        from podcast_agent.transcripts.stage import TranscriptStage
        from podcast_agent.triage.tier0 import Tier0Stage

        client = build_client()
        guard = UrlGuard(settings.security)
        return PipelineRunner(
            settings,
            store,
            ingestor=Ingestor(settings, store, client, guard),
            tier0=Tier0Stage(settings, store, llm),
            transcripts=TranscriptStage(
                settings,
                store,
                TranscriptAcquirer(settings, store, client, guard, None),  # type: ignore[arg-type]
            ),
            tier1=Tier1Stage(settings, store, llm),
            digest=DigestGenerator(settings, store),
        )

    async def _seed_scored(
        self, store: MemoryStore, *, guid: str, status: EpisodeStatus, score: int, version: str
    ) -> dict[str, Any]:
        episode = make_episode(
            guid=guid,
            status=status,
            tier1={
                "relevance_score": score,
                "summary_basis": "transcript",
                "profile_version": version,
                "summary_md": "old summary",
            },
        )
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)
        return episode

    async def test_detects_episodes_scored_under_an_old_profile(
        self, settings, store: MemoryStore
    ) -> None:
        await self._seed_scored(
            store, guid="stale", status=S.SCORED_LOW, score=3, version="stale00000"
        )
        await self._seed_scored(
            store,
            guid="current",
            status=S.SCORED_LOW,
            score=3,
            version=settings.interest_profile_version(),
        )
        stale = await self._runner(settings, store, FakeLLM()).stale_episodes(limit=10)
        assert [d["guid"] for d in stale] == ["stale"]

    async def test_promotes_when_the_new_score_clears_the_threshold(
        self, settings, store: MemoryStore
    ) -> None:
        await self._seed_scored(
            store, guid="low", status=S.SCORED_LOW, score=2, version="stale00000"
        )
        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=9, summary_md="Now relevant."))
        runner = self._runner(settings, store, llm)

        result = await runner.run_rescore(limit=10)

        assert result["rescored"] == 1
        assert result["promoted"] == 1
        doc = store.docs_of_type("episode")[0]
        assert doc["status"] == S.READY_FOR_DIGEST.value
        assert doc["tier1"]["relevance_score"] == 9
        assert doc["tier1"]["previous_score"] == 2
        assert doc["tier1"]["profile_version"] == settings.interest_profile_version()

    async def test_demotes_when_the_new_score_falls_below(
        self, settings, store: MemoryStore
    ) -> None:
        await self._seed_scored(
            store, guid="high", status=S.READY_FOR_DIGEST, score=9, version="stale00000"
        )
        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=1, summary_md="Not any more."))
        runner = self._runner(settings, store, llm)

        result = await runner.run_rescore(limit=10)

        assert result["demoted"] == 1
        assert store.docs_of_type("episode")[0]["status"] == S.SCORED_LOW.value

    async def test_skips_episodes_already_on_the_current_profile(
        self, settings, store: MemoryStore
    ) -> None:
        current = settings.interest_profile_version()
        await self._seed_scored(
            store, guid="fresh", status=S.READY_FOR_DIGEST, score=8, version=current
        )
        llm = FakeLLM()
        result = await self._runner(settings, store, llm).run_rescore(limit=10)
        assert result["candidates"] == 0
        assert llm.calls == []

    async def test_force_rescore_ignores_the_version(self, settings, store: MemoryStore) -> None:
        current = settings.interest_profile_version()
        await self._seed_scored(
            store, guid="fresh", status=S.READY_FOR_DIGEST, score=8, version=current
        )
        llm = FakeLLM()
        result = await self._runner(settings, store, llm).run_rescore(limit=10, force=True)
        assert result["rescored"] == 1

    async def test_legacy_episodes_without_a_version_are_stale(
        self, settings, store: MemoryStore
    ) -> None:
        """Episodes scored before versioning existed must not be invisible."""
        episode = make_episode(
            guid="legacy",
            status=S.SCORED_LOW,
            tier1={"relevance_score": 3, "summary_basis": "transcript"},
        )
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)
        stale = await self._runner(settings, store, FakeLLM()).stale_episodes(limit=10)
        assert len(stale) == 1

    async def test_published_episodes_are_never_rescored(
        self, settings, store: MemoryStore
    ) -> None:
        """The digest file on disk would disagree with the database."""
        await self._seed_scored(
            store, guid="published", status=S.PUBLISHED, score=9, version="stale00000"
        )
        llm = FakeLLM()
        result = await self._runner(settings, store, llm).run_rescore(limit=10)
        assert result["candidates"] == 0
        assert llm.calls == []

    async def test_reuses_the_stored_transcript(self, settings, store: MemoryStore) -> None:
        """The whole point of C2: re-scoring must not re-transcribe."""
        await self._seed_scored(store, guid="x", status=S.SCORED_LOW, score=2, version="stale00000")
        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=7, summary_md="Rescored."))
        await self._runner(settings, store, llm).run_rescore(limit=10)
        doc = store.docs_of_type("episode")[0]
        # Basis is still transcript-derived, and no transcript attempt was made.
        assert doc["tier1"]["summary_basis"] == "transcript"
        assert doc["attempts"]["transcript"] == 0

    async def test_limit_is_respected(self, settings, store: MemoryStore) -> None:
        for index in range(5):
            await self._seed_scored(
                store,
                guid=f"e{index}",
                status=S.SCORED_LOW,
                score=2,
                version="stale00000",
            )
        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=7, summary_md="ok"))
        result = await self._runner(settings, store, llm).run_rescore(limit=2)
        assert result["rescored"] == 2


# --- E4: notifications --------------------------------------------------------


def notif_config(**over: Any) -> NotificationConfig:
    base = {
        "enabled": True,
        "ntfy_url": "http://ntfy.example.com",
        "topic": "podcast-alerts",
        "min_score": 9,
    }
    base.update(over)
    return NotificationConfig(**base)  # type: ignore[arg-type]


def scored_episode(score: int, **over: Any) -> dict[str, Any]:
    return make_episode(
        tier1={
            "relevance_score": score,
            "why_it_matters": "Directly relevant to your OT remit.",
            "key_takeaways": ["Segment OT networks", "Patch the HMI", "Third bullet"],
        },
        **over,
    )


class TestNotifier:
    async def test_disabled_by_default(self) -> None:
        async with httpx.AsyncClient() as client:
            assert Notifier(NotificationConfig(), client).enabled is False

    async def test_threshold_is_strict(self) -> None:
        """A notification that fires weekly stops being read."""
        async with httpx.AsyncClient() as client:
            notifier = Notifier(notif_config(), client)
            assert notifier.should_notify(9) is True
            assert notifier.should_notify(10) is True
            assert notifier.should_notify(8) is False

    @respx.mock
    async def test_sends_expected_payload(self) -> None:
        route = respx.post("http://ntfy.example.com/podcast-alerts").mock(
            return_value=httpx.Response(200)
        )
        async with httpx.AsyncClient() as client:
            sent = await Notifier(notif_config(), client).notify_episode(scored_episode(9))

        assert sent is True
        request = route.calls[0].request
        assert "9/10" in request.headers["title"]
        assert "Test Show" in request.headers["title"]
        body = request.content.decode()
        assert "Directly relevant" in body
        # Only the first couple of takeaways — this is a push, not a digest.
        assert body.count("•") == 2

    @respx.mock
    async def test_below_threshold_sends_nothing(self) -> None:
        route = respx.post("http://ntfy.example.com/podcast-alerts").mock(
            return_value=httpx.Response(200)
        )
        async with httpx.AsyncClient() as client:
            sent = await Notifier(notif_config(), client).notify_episode(scored_episode(8))
        assert sent is False
        assert route.call_count == 0

    @respx.mock
    async def test_failure_is_swallowed(self) -> None:
        """The summary is already stored; a dead ntfy must not fail the run."""
        respx.post("http://ntfy.example.com/podcast-alerts").mock(return_value=httpx.Response(500))
        async with httpx.AsyncClient() as client:
            assert (
                await Notifier(notif_config(), client).notify_episode(scored_episode(10)) is False
            )

    @respx.mock
    async def test_connection_error_is_swallowed(self) -> None:
        respx.post("http://ntfy.example.com/podcast-alerts").mock(
            side_effect=httpx.ConnectError("refused")
        )
        async with httpx.AsyncClient() as client:
            assert (
                await Notifier(notif_config(), client).notify_episode(scored_episode(10)) is False
            )

    @respx.mock
    async def test_non_ascii_title_is_transliterated(self) -> None:
        """httpx encodes headers as ASCII, so an accent or an emoji in a title
        would otherwise raise and silently lose the notification."""
        route = respx.post("http://ntfy.example.com/podcast-alerts").mock(
            return_value=httpx.Response(200)
        )
        episode = scored_episode(9, title="Ep 42 — 🎙️ Ångström security")
        async with httpx.AsyncClient() as client:
            assert await Notifier(notif_config(), client).notify_episode(episode) is True
        assert route.call_count == 1
        title = route.calls[0].request.headers["title"]
        assert title.isascii()
        # Transliterated, not dropped.
        assert "Angstrom security" in title

    @respx.mock
    async def test_token_becomes_a_bearer_header(self) -> None:
        route = respx.post("http://ntfy.example.com/podcast-alerts").mock(
            return_value=httpx.Response(200)
        )
        async with httpx.AsyncClient() as client:
            await Notifier(notif_config(), client, "tk-secret").notify_episode(scored_episode(9))
        assert route.calls[0].request.headers["authorization"] == "Bearer tk-secret"

    @respx.mock
    async def test_tier1_fires_the_notification(self, tmp_path: Path, store: MemoryStore) -> None:
        route = respx.post("http://ntfy.example.com/podcast-alerts").mock(
            return_value=httpx.Response(200)
        )
        settings = make_settings(tmp_path)
        episode = make_episode(status=S.TRANSCRIBED)
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)

        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=10, summary_md="Exceptional."))
        async with httpx.AsyncClient() as client:
            stage = Tier1Stage(settings, store, llm, Notifier(notif_config(), client))
            await stage.summarize((await store.get(episode["_id"])) or {})

        assert route.call_count == 1

    @respx.mock
    async def test_tier1_does_not_notify_for_ordinary_scores(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        route = respx.post("http://ntfy.example.com/podcast-alerts").mock(
            return_value=httpx.Response(200)
        )
        settings = make_settings(tmp_path)
        episode = make_episode(status=S.TRANSCRIBED)
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)

        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=7, summary_md="Fine."))
        async with httpx.AsyncClient() as client:
            stage = Tier1Stage(settings, store, llm, Notifier(notif_config(), client))
            await stage.summarize((await store.get(episode["_id"])) or {})

        assert route.call_count == 0


# --- E2: glance endpoint ------------------------------------------------------


class TestGlance:
    def _client(self, tmp_path: Path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_empty_state_is_still_a_sentence(self, tmp_path: Path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/glance", headers=KEY).json()
        assert body["headline"] == "Podcast digest: 0 new summaries"
        assert body["top_pick"] is None

    def test_reports_count_and_best_item(self, tmp_path: Path, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="a",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                tier1={"relevance_score": 6},
            ),
            make_episode(
                guid="b",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 27, tzinfo=UTC),
                tier1={"relevance_score": 9},
            ),
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/glance", headers=KEY).json()

        assert "2 new summaries" in body["headline"]
        # Highest score wins, not the most recent.
        assert body["top_pick"]["score"] == 9
        assert "9/10" in body["headline"]

    def test_singular_grammar(self, tmp_path: Path, store: MemoryStore) -> None:
        store.seed(
            make_episode(guid="only", status=S.READY_FOR_DIGEST, tier1={"relevance_score": 7})
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/glance", headers=KEY).json()
        assert "1 new summary" in body["headline"]

    def test_headline_fits_a_small_display(self, tmp_path: Path, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="long",
                status=S.READY_FOR_DIGEST,
                title="A" * 400,
                tier1={"relevance_score": 9},
            )
        )
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/glance", headers=KEY).json()
        assert len(body["headline"]) <= 120

    def test_requires_the_api_key(self, tmp_path: Path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/glance").status_code == 401
