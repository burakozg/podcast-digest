"""Stage tests: Tier-0 triage, transcript acquisition, Tier-1 summarisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.db import MemoryStore, load_transcript, save_transcript
from podcast_agent.llm.base import LLMUnavailable
from podcast_agent.models import ChunkBullets, Route, Tier0Result, Tier1Result
from podcast_agent.net import UrlGuard, build_client
from podcast_agent.state import EpisodeStatus
from podcast_agent.summarize.tier1 import Tier1Stage
from podcast_agent.transcripts.acquire import TranscriptAcquirer, TranscriptUnavailable
from podcast_agent.transcripts.asr import ASRResult, ASRUnavailable
from podcast_agent.transcripts.stage import CRASH_BUDGET, TranscriptStage
from podcast_agent.triage.tier0 import Tier0Stage

S = EpisodeStatus


# --- Tier-0 -----------------------------------------------------------------


class TestTier0Stage:
    async def test_triage_records_result_and_moves_to_triaged(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        episode = make_episode()
        store.seed(episode)
        stage = Tier0Stage(settings, store, fake_llm)

        decision = await stage.triage(episode)

        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["status"] == S.TRIAGED.value
        assert doc["tier0"]["relevance_guess"] == 8
        assert doc["tier0"]["confidence"] == 9
        assert doc["tier0"]["route"] == Route.ESCALATE.value
        assert doc["tier0"]["rule"] == "confident_relevant"
        assert doc["attempts"]["tier0"] == 1
        assert decision.route is Route.ESCALATE

    async def test_telemetry_is_stored_on_the_episode(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        episode = make_episode()
        store.seed(episode)
        await Tier0Stage(settings, store, fake_llm).triage(episode)
        tier0 = (await store.get(episode["_id"]))["tier0"]  # type: ignore[index]
        assert tier0["model"] == "test-model"
        assert tier0["latency_ms"] == 42
        assert tier0["prompt_version"] == "tier0_v1"

    async def test_prompt_contains_interest_profile_and_episode_data(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        episode = make_episode()
        store.seed(episode)
        await Tier0Stage(settings, store, fake_llm).triage(episode)
        call = fake_llm.calls[0]
        assert "ot_ics" in call["system"]
        assert "OT/ICS security" in call["system"]
        # Untrusted content is fenced and labelled as data, not instructions.
        assert "<episode_data>" in call["user"]
        assert "UNTRUSTED DATA" in call["system"]
        assert "PLC malware" in call["user"]

    async def test_invented_interest_keys_are_dropped(self, settings, store: MemoryStore) -> None:
        """A hallucinated key must not reach the digest, and must not cause a retry."""
        llm = FakeLLM(
            lambda *a: Tier0Result(
                relevance_guess=8,
                confidence=9,
                matched_interests=["ot_ics", "not_a_real_key", "OT_ICS"],
            )
        )
        episode = make_episode()
        store.seed(episode)
        await Tier0Stage(settings, store, llm).triage(episode)
        tier0 = (await store.get(episode["_id"]))["tier0"]  # type: ignore[index]
        assert tier0["matched_interests"] == ["ot_ics"]

    async def test_always_escalate_show_bypasses_low_relevance(
        self, settings, store: MemoryStore
    ) -> None:
        llm = FakeLLM(lambda *a: Tier0Result(relevance_guess=0, confidence=10))
        episode = make_episode(slug="priority-show", guid="p1")
        store.seed(episode)
        decision = await Tier0Stage(settings, store, llm).triage(episode)
        assert decision.route is Route.ESCALATE
        assert decision.rule == "always_escalate"

    async def test_model_route_is_recorded_but_not_obeyed(
        self, settings, store: MemoryStore
    ) -> None:
        llm = FakeLLM(
            lambda *a: Tier0Result(relevance_guess=1, confidence=10, route=Route.ESCALATE)
        )
        episode = make_episode()
        store.seed(episode)
        await Tier0Stage(settings, store, llm).triage(episode)
        tier0 = (await store.get(episode["_id"]))["tier0"]  # type: ignore[index]
        assert tier0["model_suggested_route"] == "ESCALATE"
        assert tier0["route"] == "DROP"

    @pytest.mark.parametrize(
        ("relevance", "confidence", "expected"),
        [
            (9, 9, S.AWAITING_TRANSCRIPT),
            (1, 9, S.DROPPED),
            (5, 9, S.DIGEST_DIRECT),
            (9, 2, S.AWAITING_TRANSCRIPT),
        ],
    )
    async def test_dispatch_applies_the_stored_route(
        self,
        settings,
        store: MemoryStore,
        relevance: int,
        confidence: int,
        expected: EpisodeStatus,
    ) -> None:
        llm = FakeLLM(lambda *a: Tier0Result(relevance_guess=relevance, confidence=confidence))
        episode = make_episode()
        store.seed(episode)
        stage = Tier0Stage(settings, store, llm)
        await stage.triage(episode)

        triaged = await store.get(episode["_id"])
        assert triaged is not None
        assert await stage.dispatch(triaged) is expected
        assert (await store.get(episode["_id"]))["status"] == expected.value  # type: ignore[index]

    async def test_llm_failure_propagates_without_mutating_the_episode(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        """A dead tier must leave the episode queued, not consumed."""
        fake_llm.fail_with = LLMUnavailable("all endpoints down")
        episode = make_episode()
        store.seed(episode)
        with pytest.raises(LLMUnavailable):
            await Tier0Stage(settings, store, fake_llm).triage(episode)
        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["status"] == S.NEW.value
        assert doc["tier0"] is None


# --- transcripts ------------------------------------------------------------


class FakeASR:
    def __init__(self, text: str = "", fail: Exception | None = None) -> None:
        self._text = text
        self._fail = fail
        self.calls: list[Path] = []

    @property
    def name(self) -> str:
        return "fake-asr"

    async def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult:
        self.calls.append(audio_path)
        if self._fail:
            raise self._fail
        return ASRResult(text=self._text, language="en", duration_s=1800)

    async def close(self) -> None:
        return None


def build_acquirer(settings, store, client, asr) -> TranscriptAcquirer:
    return TranscriptAcquirer(settings, store, client, UrlGuard(settings.security), asr)


LONG_TEXT = "This is a real transcript sentence with substance. " * 40
VTT_BODY = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n" + LONG_TEXT


class TestTranscriptAcquisition:
    @respx.mock
    async def test_prefers_feed_transcript(self, settings, store: MemoryStore) -> None:
        respx.get("https://transcript-host.net/t.vtt").mock(
            return_value=httpx.Response(200, text=VTT_BODY, headers={"content-type": "text/vtt"})
        )
        episode = make_episode(
            feed_transcripts=[{"url": "https://transcript-host.net/t.vtt", "type": "text/vtt"}]
        )
        asr = FakeASR("should not be used")
        async with build_client() as client:
            result = await build_acquirer(settings, store, client, asr).acquire(episode)
        assert result.source == "feed"
        assert "-->" not in result.text
        assert asr.calls == []  # ASR never invoked when a transcript exists

    @respx.mock
    async def test_falls_through_short_transcript_to_asr(
        self, settings, store: MemoryStore
    ) -> None:
        """A 'transcript coming soon' stub must not be accepted as a transcript."""
        respx.get("https://transcript-host.net/stub.txt").mock(
            return_value=httpx.Response(200, text="Transcript coming soon.")
        )
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"\xff\xfb" + b"0" * 5000, headers={"content-type": "audio/mpeg"}
            )
        )
        episode = make_episode(
            feed_transcripts=[{"url": "https://transcript-host.net/stub.txt", "type": "text/plain"}]
        )
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            result = await build_acquirer(settings, store, client, asr).acquire(episode)
        assert result.source == "asr"
        assert len(asr.calls) == 1

    @respx.mock
    async def test_asr_when_no_feed_transcript(self, settings, store: MemoryStore) -> None:
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"audio-bytes" * 500, headers={"content-type": "audio/mpeg"}
            )
        )
        episode = make_episode()
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            result = await build_acquirer(settings, store, client, asr).acquire(episode)
        assert result.source == "asr"
        assert result.detected_language == "en"
        assert result.duration_s == 1800

    @respx.mock
    async def test_audio_is_deleted_after_transcription(self, settings, store: MemoryStore) -> None:
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"audio" * 1000, headers={"content-type": "audio/mpeg"}
            )
        )
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            await build_acquirer(settings, store, client, asr).acquire(make_episode())
        assert asr.calls[0].exists() is False

    @respx.mock
    async def test_audio_kept_when_configured(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(tmp_path, asr={"keep_audio": True})
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"audio" * 1000, headers={"content-type": "audio/mpeg"}
            )
        )
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            await build_acquirer(settings, store, client, asr).acquire(make_episode())
        assert asr.calls[0].exists() is True

    @respx.mock
    async def test_oversized_declared_enclosure_is_skipped(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, asr={"max_audio_mb": 1})
        episode = make_episode(enclosure_bytes=500_000_000)
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            with pytest.raises(TranscriptUnavailable, match="cap"):
                await build_acquirer(settings, store, client, asr).acquire(episode)
        assert asr.calls == []

    @respx.mock
    async def test_download_over_cap_is_aborted_midstream(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """A lying Content-Length must not defeat the cap (§10.2)."""
        settings = make_settings(tmp_path, asr={"max_audio_mb": 1})
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"x" * (2 * 1024 * 1024), headers={"content-type": "audio/mpeg"}
            )
        )
        episode = make_episode(enclosure_bytes=None)
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            with pytest.raises(TranscriptUnavailable):
                await build_acquirer(settings, store, client, asr).acquire(episode)
        assert asr.calls == []

    @respx.mock
    async def test_non_audio_content_type_is_refused(self, settings, store: MemoryStore) -> None:
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"<html>gotcha</html>", headers={"content-type": "text/html"}
            )
        )
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            with pytest.raises(TranscriptUnavailable):
                await build_acquirer(settings, store, client, asr).acquire(make_episode())
        assert asr.calls == []

    @respx.mock
    async def test_transcript_on_unlisted_domain_is_skipped(
        self, settings, store: MemoryStore
    ) -> None:
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"audio" * 1000, headers={"content-type": "audio/mpeg"}
            )
        )
        episode = make_episode(
            feed_transcripts=[{"url": "https://evil-host.org/t.txt", "type": "text/plain"}]
        )
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            result = await build_acquirer(settings, store, client, asr).acquire(episode)
        # Fell through to ASR rather than fetching the disallowed URL.
        assert result.source == "asr"

    @respx.mock
    async def test_configured_selector_scrape(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": "https://example.com/feed.xml",
                    "transcript_selector": "div.transcript",
                }
            ],
        )
        respx.get("https://example.com/ep1").mock(
            return_value=httpx.Response(
                200,
                text=f"<html><body><div class='transcript'>{LONG_TEXT}</div>"
                "<div class='ads'>buy things</div></body></html>",
                headers={"content-type": "text/html"},
            )
        )
        asr = FakeASR("unused")
        async with build_client() as client:
            result = await build_acquirer(settings, store, client, asr).acquire(make_episode())
        assert result.source == "scrape"
        assert "buy things" not in result.text
        assert asr.calls == []

    @respx.mock
    async def test_the_scrape_follows_a_sibling_page_rule(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The linked page is show notes; the transcript is one path segment away.

        Without the rewrite the selector is applied to the wrong page and
        matches nothing, which looks exactly like a show that publishes no
        transcript — and costs it an hour of local transcription, or a
        description-only summary.
        """
        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": "https://example.com/feed.xml",
                    "transcript_selector": "div.transcript",
                    "transcript_url_sub": ["/ep1", "/ep1/transcript"],
                }
            ],
        )
        notes = respx.get("https://example.com/ep1").mock(
            return_value=httpx.Response(
                200,
                text="<div class='notes'>show notes only</div>",
                headers={"content-type": "text/html"},
            )
        )
        transcript = respx.get("https://example.com/ep1/transcript").mock(
            return_value=httpx.Response(
                200,
                text=f"<div class='transcript'>{LONG_TEXT}</div>",
                headers={"content-type": "text/html"},
            )
        )
        asr = FakeASR("unused")
        async with build_client() as client:
            result = await build_acquirer(settings, store, client, asr).acquire(make_episode())

        assert result.source == "scrape"
        assert transcript.call_count == 1
        assert notes.call_count == 0, "the notes page should not be fetched at all"
        assert asr.calls == []

    @respx.mock
    async def test_scraping_never_happens_without_a_configured_selector(
        self, settings, store: MemoryStore
    ) -> None:
        """§4: generic scraping must never be attempted."""
        page = respx.get("https://example.com/ep1").mock(
            return_value=httpx.Response(200, text=f"<div>{LONG_TEXT}</div>")
        )
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"audio" * 1000, headers={"content-type": "audio/mpeg"}
            )
        )
        asr = FakeASR(LONG_TEXT)
        async with build_client() as client:
            await build_acquirer(settings, store, client, asr).acquire(make_episode())
        assert page.call_count == 0


class TestTranscriptCrashSafety:
    """A crash must bound the loop without blaming the episode it caught."""

    async def test_crash_is_recorded_before_the_risky_work(
        self, settings, store: MemoryStore
    ) -> None:
        episode = make_episode(status=S.AWAITING_TRANSCRIPT)
        store.seed(episode)
        seen: list[int] = []

        class _CrashingAcquirer:
            async def acquire(self, doc, *, allow_asr=True):
                fresh = await store.get(doc["_id"])
                seen.append(int((fresh.get("attempts") or {}).get("transcript_crash") or 0))
                raise AssertionError("process died here")

        stage = TranscriptStage(settings, store, _CrashingAcquirer())
        with pytest.raises(AssertionError):
            await stage.process(episode)

        assert seen == [1], "the in-flight marker must be persisted before acquisition"
        survived = await store.get(episode["_id"])
        assert (survived.get("attempts") or {}).get("transcript_crash") == 1

    async def test_exhausted_crash_budget_retires_without_touching_the_audio(
        self, settings, store: MemoryStore
    ) -> None:
        episode = make_episode(status=S.AWAITING_TRANSCRIPT)
        episode["attempts"] = {"transcript_crash": CRASH_BUDGET}
        store.seed(episode)
        called = False

        class _NeverCalled:
            async def acquire(self, doc, *, allow_asr=True):
                nonlocal called
                called = True
                raise AssertionError("must not be reached")

        stage = TranscriptStage(settings, store, _NeverCalled())
        assert await stage.process(episode) is S.TRANSCRIPT_FAILED
        assert not called

    @respx.mock
    async def test_success_releases_the_marker(self, settings, store: MemoryStore) -> None:
        """A clean run must leave no crash residue, or bystanders accumulate.

        One-minute episodes were retired for OOMs they could not have caused,
        because the marker was never given back on a clean exit.
        """
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"audio" * 1000, headers={"content-type": "audio/mpeg"}
            )
        )
        episode = make_episode(status=S.AWAITING_TRANSCRIPT)
        episode["attempts"] = {"transcript_crash": 2}
        store.seed(episode)
        async with build_client() as client:
            stage = TranscriptStage(
                settings, store, build_acquirer(settings, store, client, FakeASR(LONG_TEXT))
            )
            assert await stage.process(episode) is S.TRANSCRIBED

        done = await store.get(episode["_id"])
        assert (done.get("attempts") or {}).get("transcript_crash") == 2, (
            "a successful run must hand the marker back, not leave it claimed"
        )
        assert (done.get("attempts") or {}).get("transcript") == 1


class TestTranscriptStage:
    @respx.mock
    async def test_success_stores_gzipped_transcript(self, settings, store: MemoryStore) -> None:
        respx.get("https://cdn-host.net/ep1.mp3").mock(
            return_value=httpx.Response(
                200, content=b"audio" * 1000, headers={"content-type": "audio/mpeg"}
            )
        )
        episode = make_episode(status=S.AWAITING_TRANSCRIPT)
        store.seed(episode)
        async with build_client() as client:
            stage = TranscriptStage(
                settings, store, build_acquirer(settings, store, client, FakeASR(LONG_TEXT))
            )
            outcome = await stage.process(episode)

        assert outcome is S.TRANSCRIBED
        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["status"] == S.TRANSCRIBED.value
        assert doc["transcript_source"] == "asr"
        assert doc["transcript_bytes_gz"] < doc["transcript_chars"]
        assert await load_transcript(store, episode["_id"]) is not None

    @respx.mock
    async def test_retries_stay_queued_until_the_budget_is_spent(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, pipeline={"max_retries": 3})
        respx.get("https://cdn-host.net/ep1.mp3").mock(return_value=httpx.Response(500))
        episode = make_episode(status=S.AWAITING_TRANSCRIPT)
        store.seed(episode)

        async with build_client() as client:
            stage = TranscriptStage(
                settings, store, build_acquirer(settings, store, client, FakeASR(LONG_TEXT))
            )
            for attempt in (1, 2):
                current = await store.get(episode["_id"])
                assert current is not None
                assert await stage.process(current) is S.AWAITING_TRANSCRIPT
                doc = await store.get(episode["_id"])
                assert doc is not None
                assert doc["attempts"]["transcript"] == attempt

            # Third failure exhausts max_retries and falls back to description-only.
            current = await store.get(episode["_id"])
            assert current is not None
            assert await stage.process(current) is S.TRANSCRIPT_FAILED

        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["status"] == S.TRANSCRIPT_FAILED.value
        assert doc["last_error"]["stage"] == "transcript"

    async def test_backend_outage_does_not_consume_retry_budget(
        self, settings, store: MemoryStore
    ) -> None:
        """An operator problem (missing faster-whisper) is not the episode's fault."""
        episode = make_episode(status=S.AWAITING_TRANSCRIPT)
        store.seed(episode)
        async with build_client() as client:
            acquirer = build_acquirer(
                settings, store, client, FakeASR(fail=ASRUnavailable("not installed"))
            )
            stage = TranscriptStage(settings, store, acquirer)
            with respx.mock:
                respx.get("https://cdn-host.net/ep1.mp3").mock(
                    return_value=httpx.Response(
                        200, content=b"audio" * 1000, headers={"content-type": "audio/mpeg"}
                    )
                )
                with pytest.raises(ASRUnavailable):
                    await stage.process(episode)
        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["status"] == S.AWAITING_TRANSCRIPT.value
        assert doc["attempts"]["transcript"] == 0


# --- Tier-1 -----------------------------------------------------------------


class TestTier1Stage:
    async def test_single_call_summary(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        episode = make_episode(status=S.TRANSCRIBED, transcript_source="asr")
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)

        outcome = await Tier1Stage(settings, store, fake_llm).summarize(
            (await store.get(episode["_id"])) or {}
        )

        assert outcome is S.READY_FOR_DIGEST
        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["status"] == S.READY_FOR_DIGEST.value
        assert doc["tier1"]["relevance_score"] == 8
        assert doc["tier1"]["summary_basis"] == "transcript"
        assert doc["tier1"]["chunks"] == 0
        assert doc["tier1"]["llm_calls"] == 1
        assert len(fake_llm.calls) == 1

    async def test_published_transcript_basis_is_labelled(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        episode = make_episode(status=S.TRANSCRIBED, transcript_source="feed")
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)
        await Tier1Stage(settings, store, fake_llm).summarize(
            (await store.get(episode["_id"])) or {}
        )
        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["tier1"]["summary_basis"] == "published_transcript"

    async def test_description_only_fallback(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        """§4: TRANSCRIPT_FAILED still gets a summary, honestly labelled."""
        episode = make_episode(status=S.TRANSCRIPT_FAILED)
        store.seed(episode)

        await Tier1Stage(settings, store, fake_llm).summarize(episode)

        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["tier1"]["summary_basis"] == "description_only"
        assert "description_only" in fake_llm.calls[0]["user"]

    async def test_basis_is_not_taken_from_the_model(self, settings, store: MemoryStore) -> None:
        """Untrusted output must not be able to relabel its own provenance."""
        # Tier1Result has no summary_basis field at all, so a model claiming one
        # is simply ignored by validation.
        llm = FakeLLM(
            lambda *a: Tier1Result.model_validate(
                {"relevance_score": 9, "summary_basis": "transcript", "summary_md": "text"}
            )
        )
        episode = make_episode(status=S.TRANSCRIPT_FAILED)
        store.seed(episode)
        await Tier1Stage(settings, store, llm).summarize(episode)
        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["tier1"]["summary_basis"] == "description_only"

    async def test_low_score_goes_to_scored_low(self, settings, store: MemoryStore) -> None:
        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=2, summary_md="Not relevant."))
        episode = make_episode(status=S.TRANSCRIBED)
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)
        outcome = await Tier1Stage(settings, store, llm).summarize(
            (await store.get(episode["_id"])) or {}
        )
        assert outcome is S.SCORED_LOW
        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["status"] == S.SCORED_LOW.value
        # Kept for the audit trail, not discarded.
        assert doc["tier1"]["relevance_score"] == 2

    async def test_threshold_boundary_is_inclusive(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, pipeline={"digest_threshold": 5})
        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=5, summary_md="Borderline."))
        episode = make_episode(status=S.TRANSCRIBED)
        store.seed(episode)
        await save_transcript(store, episode["_id"], LONG_TEXT)
        assert (
            await Tier1Stage(settings, store, llm).summarize(
                (await store.get(episode["_id"])) or {}
            )
            is S.READY_FOR_DIGEST
        )

    async def test_map_reduce_for_long_transcripts(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(
            tmp_path, pipeline={"max_input_tokens": 1000, "chunk_target_tokens": 500}
        )
        calls: list[str] = []

        def handler(tier: str, system: str, user: str, model: type[Any]) -> Any:
            calls.append(model.__name__)
            if model is ChunkBullets:
                return ChunkBullets(bullets=[f"point {len(calls)}"], entities=["Modbus"])
            return Tier1Result(relevance_score=9, summary_md="Synthesised summary.")

        llm = FakeLLM(handler)
        episode = make_episode(status=S.TRANSCRIBED)
        store.seed(episode)
        # ~20k chars → well over the 1000-token (4k char) budget.
        await save_transcript(store, episode["_id"], "Paragraph of speech. " * 1000)

        await Tier1Stage(settings, store, llm).summarize((await store.get(episode["_id"])) or {})

        doc = await store.get(episode["_id"])
        assert doc is not None
        assert doc["tier1"]["chunks"] > 1
        assert doc["tier1"]["llm_calls"] == doc["tier1"]["chunks"] + 1  # maps + reduce
        assert calls[-1] == "Tier1Result"  # reduce runs last
        assert calls.count("ChunkBullets") == doc["tier1"]["chunks"]

    async def test_map_reduce_costs_are_summed(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(
            tmp_path, pipeline={"max_input_tokens": 1000, "chunk_target_tokens": 500}
        )

        def handler(tier: str, system: str, user: str, model: type[Any]) -> Any:
            if model is ChunkBullets:
                return ChunkBullets(bullets=["point"])
            return Tier1Result(relevance_score=9, summary_md="Summary.")

        llm = FakeLLM(handler)
        episode = make_episode(status=S.TRANSCRIBED)
        store.seed(episode)
        await save_transcript(store, episode["_id"], "Paragraph of speech. " * 1000)
        await Tier1Stage(settings, store, llm).summarize((await store.get(episode["_id"])) or {})
        doc = await store.get(episode["_id"])
        assert doc is not None
        # Latency is the whole-episode total across every call, not just the last.
        assert doc["tier1"]["latency_ms"] == 42 * doc["tier1"]["llm_calls"]

    async def test_description_only_never_uses_map_reduce(
        self, tmp_path: Path, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        """A long description is still one call — chunking it would be absurd."""
        settings = make_settings(
            tmp_path, pipeline={"max_input_tokens": 1000, "chunk_target_tokens": 500}
        )
        # ~40k chars of description, far over the 1000-token budget.
        episode = make_episode(status=S.TRANSCRIPT_FAILED, description="word " * 8000)
        store.seed(episode)
        await Tier1Stage(settings, store, fake_llm).summarize(episode)
        assert len(fake_llm.calls) == 1

    async def test_empty_content_raises(
        self, settings, store: MemoryStore, fake_llm: FakeLLM
    ) -> None:
        episode = make_episode(status=S.TRANSCRIPT_FAILED, description="")
        store.seed(episode)
        with pytest.raises(ValueError, match="nothing to summarise"):
            await Tier1Stage(settings, store, fake_llm).summarize(episode)


class TestSlightlyOffSpecOutputIsUsed:
    """A cap bounds output size; it is not a requirement the model must meet.

    `Field(max_length=...)` rejects, and the constraint is checked *before* any
    mode="after" validator — so the cleaning validators that truncate never ran
    on the values that needed truncating. A model returning 27 entities against
    a cap of 25 failed the call, retried, failed again, and lost the summary.
    """

    def test_too_many_entities_are_trimmed_not_rejected(self) -> None:
        from podcast_agent.models import ChunkBullets

        result = ChunkBullets(entities=[f"e{i}" for i in range(27)], bullets=[])
        assert len(result.entities) == 25
        assert result.entities[0] == "e0"

    def test_too_many_bullets_are_trimmed(self) -> None:
        from podcast_agent.models import ChunkBullets

        assert len(ChunkBullets(bullets=[f"b{i}" for i in range(30)]).bullets) == 15

    @pytest.mark.parametrize(
        ("field", "value", "cap"),
        [
            ("matched_interests", ["k"] * 40, 20),
            ("key_takeaways", ["t"] * 30, 12),
            ("entities", ["e"] * 90, 40),
        ],
    )
    def test_every_capped_list_on_a_summary_is_trimmed(
        self, field: str, value: list[str], cap: int
    ) -> None:
        from podcast_agent.models import Tier1Result

        result = Tier1Result(relevance_score=8, **{field: value})
        assert len(getattr(result, field)) <= cap

    def test_an_over_long_string_is_trimmed_not_rejected(self) -> None:
        """Same trap: the cleaning validator that truncates never got to run."""
        from podcast_agent.models import Tier1Result

        result = Tier1Result(relevance_score=8, why_it_matters="x" * 1500)
        assert 0 < len(result.why_it_matters) <= 1000

    def test_triage_output_is_trimmed_too(self) -> None:
        from podcast_agent.models import Tier0Result

        result = Tier0Result(
            relevance_guess=5,
            confidence=5,
            matched_interests=["k"] * 40,
            reasoning="r" * 900,
        )
        assert len(result.matched_interests) <= 20
        assert len(result.reasoning) <= 600

    def test_a_response_within_the_caps_is_untouched(self) -> None:
        """Trimming must not quietly reshape a well-formed answer."""
        from podcast_agent.models import ChunkBullets

        entities = [f"e{i}" for i in range(5)]
        assert ChunkBullets(entities=entities, bullets=["a", "b"]).entities == entities


class TestHopelessTranscriptFailuresDoNotRetry:
    """Retrying only makes sense when something could have worked.

    "This podcast publishes no transcript, no scrape selector is configured, and
    local transcription is off" is not a transient condition. Retrying it three
    times produced three identical warnings and delayed the description-only
    summary the episode was always going to get.
    """

    def _stage(self, tmp_path: Path, store: MemoryStore):
        from podcast_agent.transcripts.acquire import TranscriptAcquirer
        from podcast_agent.transcripts.stage import TranscriptStage

        settings = make_settings(tmp_path)
        return TranscriptStage(
            settings,
            store,
            TranscriptAcquirer(  # type: ignore[arg-type]
                settings, store, build_client(), UrlGuard(settings.security), None
            ),
        )

    async def test_nothing_to_try_fails_immediately(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        episode = make_episode(
            guid="bare", status=EpisodeStatus.AWAITING_TRANSCRIPT, feed_transcripts=[]
        )
        store.seed(episode)
        outcome = await self._stage(tmp_path, store).process(episode, allow_asr=False)

        assert outcome is EpisodeStatus.TRANSCRIPT_FAILED
        doc = next(iter(store.docs_of_type("episode")))
        assert doc["attempts"]["transcript"] == 1, "should not have burned three attempts"

    async def test_the_error_says_what_is_missing_in_plain_words(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        episode = make_episode(
            guid="bare", status=EpisodeStatus.AWAITING_TRANSCRIPT, feed_transcripts=[]
        )
        store.seed(episode)
        await self._stage(tmp_path, store).process(episode, allow_asr=False)

        doc = next(iter(store.docs_of_type("episode")))
        assert "local transcription is off" in doc["last_error"]["message"]

    async def test_a_transcript_that_could_not_be_fetched_still_retries(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """A published transcript that 500s today may work tomorrow."""
        episode = make_episode(
            guid="published",
            status=EpisodeStatus.AWAITING_TRANSCRIPT,
            feed_transcripts=[{"url": "https://transcript-host.net/a.txt", "type": "text/plain"}],
        )
        store.seed(episode)
        with respx.mock:
            respx.get("https://transcript-host.net/a.txt").mock(return_value=httpx.Response(503))
            outcome = await self._stage(tmp_path, store).process(episode, allow_asr=False)

        assert outcome is EpisodeStatus.AWAITING_TRANSCRIPT, "a fetch failure is transient"
        doc = next(iter(store.docs_of_type("episode")))
        assert doc["attempts"]["transcript"] == 1


class TestTranscriptionReportsProgress:
    """An hour of audio is ten to twenty minutes of silence otherwise.

    The only evidence of life used to be the line logged before decoding
    started, which is indistinguishable from a hung process — and looks exactly
    like "Whisper is not creating logs".
    """

    class _FakeSegment:
        def __init__(self, text: str, end: float) -> None:
            self.text, self.end = text, end

    class _FakeInfo:
        duration = 3600.0
        language = "en"

    def _model(self, segments: list[Any]) -> Any:
        class _Model:
            def transcribe(self, *_a: Any, **_k: Any) -> Any:
                return iter(segments), TestTranscriptionReportsProgress._FakeInfo()

        return _Model()

    def _backend(self, tmp_path: Path):
        from podcast_agent.transcripts.asr import LocalFasterWhisperBackend

        return LocalFasterWhisperBackend(make_settings(tmp_path).asr)

    def test_a_completion_line_always_reports_speed(self, tmp_path: Path) -> None:
        """The number that says whether this model and machine are a sane pair."""
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        backend = self._backend(tmp_path)
        segments = [self._FakeSegment("hello", 10.0), self._FakeSegment("world", 20.0)]
        result = backend._transcribe_sync(self._model(segments), Path("x.wav"), "en")

        assert result.text == "hello world"
        done = next(e for e in buffer.tail(limit=10) if e["event"] == "asr.complete")
        assert done["audio_duration_s"] == 3600
        assert done["realtime_factor"] is not None

    def test_a_short_file_reports_no_progress_lines(self, tmp_path: Path) -> None:
        """Throttled by wall time, so a quick job stays quiet."""
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        backend = self._backend(tmp_path)
        backend._transcribe_sync(self._model([self._FakeSegment("a", 1.0)]), Path("x.wav"), "en")
        assert not [e for e in buffer.tail(limit=10) if e["event"] == "asr.progress"]

    def test_a_long_job_reports_where_it_has_got_to(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging
        from podcast_agent.transcripts import asr as asr_module

        # Every segment appears to take a minute of wall time.
        ticks = iter([0.0] + [60.0 * i for i in range(1, 20)])
        monkeypatch.setattr(asr_module.time, "monotonic", lambda: next(ticks))

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        backend = self._backend(tmp_path)
        backend._transcribe_sync(
            self._model([self._FakeSegment("s", 600.0 * i) for i in range(1, 4)]),
            Path("x.wav"),
            "en",
        )
        progress = [e for e in buffer.tail(limit=20) if e["event"] == "asr.progress"]
        assert progress, "a long job must say where it has got to"
        assert progress[0]["percent"] is not None
        assert progress[0]["audio_position_s"] > 0


class TestATranscriptionIsRecorded:
    """The write itself, not a seeded row.

    Without this the telemetry tests would pass against a system that never
    writes an asr_run at all — which is exactly the state this replaced.
    """

    class _Backend:
        name = "local:fake"

        async def transcribe(self, audio_path: Path, *, language: str | None = None):
            from podcast_agent.transcripts.asr import ASRResult

            return ASRResult(
                text="a transcript long enough to be accepted " * 40,
                language="en",
                duration_s=3600,
                elapsed_s=900.0,
            )

        async def close(self) -> None:
            return None

    async def _acquire(self, tmp_path: Path, store: MemoryStore):
        from podcast_agent.transcripts.acquire import TranscriptAcquirer

        settings = make_settings(tmp_path)
        episode = make_episode(
            guid="needs-asr", status=EpisodeStatus.AWAITING_TRANSCRIPT, feed_transcripts=[]
        )
        store.seed(episode)
        acquirer = TranscriptAcquirer(
            settings, store, build_client(), UrlGuard(settings.security), self._Backend()
        )
        with respx.mock:
            respx.get("https://cdn-host.net/ep1.mp3").mock(
                return_value=httpx.Response(200, content=b"x" * 2048)
            )
            await acquirer.acquire(episode, allow_asr=True)
        return store

    async def test_a_run_is_written(self, tmp_path: Path, store: MemoryStore) -> None:
        store = await self._acquire(tmp_path, store)
        runs = store.docs_of_type("asr_run")
        assert len(runs) == 1
        run = runs[0]
        assert run["audio_duration_s"] == 3600
        assert run["elapsed_s"] == 900.0
        assert run["realtime_factor"] == 4.0
        assert run["podcast_slug"] == "test-show"
        assert run["model"] and run["device"]

    async def test_telemetry_failing_does_not_lose_the_transcript(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """An episode that was just transcribed must not be failed by bookkeeping."""
        from podcast_agent.transcripts.acquire import TranscriptAcquirer

        class Broken(MemoryStore):
            async def create(self, doc: Any) -> bool:
                if doc.get("type") == "asr_run":
                    raise RuntimeError("couch is down")
                return await super().create(doc)

        settings = make_settings(tmp_path)
        broken = Broken()
        episode = make_episode(
            guid="needs-asr", status=EpisodeStatus.AWAITING_TRANSCRIPT, feed_transcripts=[]
        )
        broken.seed(episode)
        acquirer = TranscriptAcquirer(
            settings, broken, build_client(), UrlGuard(settings.security), self._Backend()
        )
        with respx.mock:
            respx.get("https://cdn-host.net/ep1.mp3").mock(
                return_value=httpx.Response(200, content=b"x" * 2048)
            )
            result = await acquirer.acquire(episode, allow_asr=True)
        assert result.source == "asr"
