"""End-to-end pipeline tests: fixture RSS + canned LLM responses, no network.

Exercises the property the design leans on hardest (§10.3): every stage is driven
by document status, so the whole pipeline is resumable and one bad episode cannot
block the queue.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.db import MemoryStore, load_transcript
from podcast_agent.digest.generate import DigestGenerator
from podcast_agent.ingest.feeds import Ingestor
from podcast_agent.joblock import lock_id
from podcast_agent.llm.base import LLMUnavailable
from podcast_agent.models import ChunkBullets, Tier0Result, Tier1Result
from podcast_agent.net import UrlGuard, build_client
from podcast_agent.pipeline.runner import JobBusy, PipelineRunner, PipelineStats
from podcast_agent.retention import RetentionJob
from podcast_agent.state import EpisodeStatus
from podcast_agent.summarize.tier1 import Tier1Stage
from podcast_agent.transcripts.acquire import TranscriptAcquirer
from podcast_agent.transcripts.asr import ASRResult
from podcast_agent.transcripts.stage import TranscriptStage
from podcast_agent.triage.tier0 import Tier0Stage

S = EpisodeStatus

FEED_URL = "https://example.com/feed.xml"
TRANSCRIPT_TEXT = "A substantive transcript sentence about ICS security. " * 40


def build_feed() -> str:
    """Feed with recent pubDates so the backfill guard never interferes."""
    now = datetime.now(UTC)
    items = []
    for index, (title, description) in enumerate(
        [
            ("Deep dive on PLC malware", "Detailed walkthrough of Modbus abuse and segmentation."),
            ("Vague marketing episode", "Join us for a chat!"),
            ("Totally unrelated cooking show", "We bake bread and discuss sourdough starters."),
        ]
    ):
        published = (now - timedelta(days=index + 1)).strftime("%a, %d %b %Y %H:%M:%S +0000")
        items.append(
            f"""<item>
              <title>{title}</title>
              <link>https://example.com/ep{index}</link>
              <guid isPermaLink="false">e2e-{index}</guid>
              <pubDate>{published}</pubDate>
              <description>{description}</description>
              <itunes:duration>1800</itunes:duration>
              <enclosure url="https://cdn-host.net/ep{index}.mp3" length="20000000" type="audio/mpeg"/>
            </item>"""
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
      <channel><title>Test Show</title>{"".join(items)}</channel>
    </rss>"""


class FakeASR:
    def __init__(self, text: str = TRANSCRIPT_TEXT) -> None:
        self._text = text
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake-asr"

    async def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult:
        self.calls += 1
        return ASRResult(text=self._text, language="en", duration_s=1800)

    async def close(self) -> None:
        return None


def routing_llm() -> FakeLLM:
    """Route each fixture episode down a different branch of the pipeline."""

    def handler(tier: str, system: str, user: str, model: type[Any]) -> Any:
        if model is Tier0Result:
            if "PLC malware" in user:
                return Tier0Result(relevance_guess=9, confidence=9, matched_interests=["ot_ics"])
            if "Vague marketing" in user:
                # Thin description → must escalate, never drop (§4).
                return Tier0Result(relevance_guess=5, confidence=2)
            return Tier0Result(relevance_guess=1, confidence=9)  # confidently irrelevant
        if model is ChunkBullets:
            return ChunkBullets(bullets=["A point"], entities=["Modbus"])
        score = 9 if "PLC" in user or "ICS" in user else 3
        return Tier1Result(
            relevance_score=score,
            matched_interests=["ot_ics"] if score > 5 else [],
            why_it_matters="Relevant to your OT remit." if score > 5 else "Marginal.",
            summary_md="A **detailed** summary of the discussion.",
            key_takeaways=["Segment networks"],
            entities=["Modbus"],
        )

    return FakeLLM(handler)


def build_runner(settings, store: MemoryStore, client, llm, asr) -> PipelineRunner:
    guard = UrlGuard(settings.security)
    return PipelineRunner(
        settings,
        store,
        ingestor=Ingestor(settings, store, client, guard),
        tier0=Tier0Stage(settings, store, llm),
        transcripts=TranscriptStage(
            settings, store, TranscriptAcquirer(settings, store, client, guard, asr)
        ),
        tier1=Tier1Stage(settings, store, llm),
        digest=DigestGenerator(settings, store),
    )


@pytest.fixture
def e2e_settings(tmp_path: Path):
    return make_settings(
        tmp_path,
        podcasts=[
            {
                "slug": "test-show",
                "name": "Test Show",
                "feed_url": FEED_URL,
                "priority": "med",
                "asr_enabled": True,
            }
        ],
    )


def mock_http() -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=build_feed()))
    for index in range(3):
        respx.get(f"https://cdn-host.net/ep{index}.mp3").mock(
            return_value=httpx.Response(
                200, content=b"audio" * 2000, headers={"content-type": "audio/mpeg"}
            )
        )


class TestFullRun:
    @respx.mock
    async def test_ingest_then_pipeline_then_digest(self, e2e_settings, store: MemoryStore) -> None:
        mock_http()
        asr = FakeASR()
        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, routing_llm(), asr)

            ingest_stats = await runner.run_ingest()
            assert ingest_stats["episodes_created"] == 3

            pipeline_stats = await runner.run_pipeline()
            assert pipeline_stats["triaged"] == 3
            assert pipeline_stats["dropped"] == 1  # cooking show
            assert pipeline_stats["escalated"] == 2  # relevant + low-confidence
            assert pipeline_stats["transcripts_ok"] == 2
            assert pipeline_stats["errors"] == 0
            assert pipeline_stats["stages_deferred"] == []

            digest = await runner.run_digest()

        assert asr.calls == 2  # only escalated episodes cost ASR time
        assert digest.file_path is not None
        text = digest.file_path.read_text()
        assert "Deep dive on PLC malware" in text
        # The confidently-irrelevant episode is in the audit table, not dropped silently.
        assert "cooking show" in text.lower()
        assert "## Everything else scanned" in text

    @respx.mock
    async def test_thin_description_reaches_a_full_summary(
        self, e2e_settings, store: MemoryStore
    ) -> None:
        """The core requirement: a marketing blurb must not lose a good episode."""
        mock_http()
        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, routing_llm(), FakeASR())
            await runner.run_ingest()
            await runner.run_pipeline()

        vague = next(e for e in store.docs_of_type("episode") if "Vague" in e["title"])
        assert vague["tier0"]["rule"] == "low_confidence"
        assert vague["transcript_source"] == "asr"
        assert vague["tier1"]["summary_basis"] == "transcript"
        assert await load_transcript(store, vague["_id"]) is not None

    @respx.mock
    async def test_second_pipeline_run_is_a_no_op(self, e2e_settings, store: MemoryStore) -> None:
        mock_http()
        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, routing_llm(), FakeASR())
            await runner.run_ingest()
            await runner.run_pipeline()
            second = await runner.run_pipeline()
        assert second["triaged"] == 0
        assert second["transcripts_ok"] == 0
        assert second["summarized"] == 0


class TestResumability:
    @respx.mock
    async def test_pipeline_resumes_from_document_status(
        self, e2e_settings, store: MemoryStore
    ) -> None:
        """Simulates a crash after triage: the next run picks up mid-pipeline and
        does not re-bill the Tier-0 calls."""
        mock_http()
        llm = routing_llm()
        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, llm, FakeASR())
            await runner.run_ingest()
            # Run only the triage stage, as if the process died right after.
            await runner._stage_triage(PipelineStats())
            tier0_calls = len(llm.calls)
            assert all(e["status"] == S.TRIAGED.value for e in store.docs_of_type("episode"))

            stats = await runner.run_pipeline()

        # No episode was triaged twice.
        assert stats["triaged"] == 0
        assert stats["dispatched"] == 3
        tier0_after = sum(1 for c in llm.calls if c["response_model"] == "Tier0Result")
        assert tier0_after == tier0_calls

    @respx.mock
    async def test_poison_pill_does_not_block_the_queue(
        self, e2e_settings, store: MemoryStore
    ) -> None:
        """§10.3: one exploding episode is marked ERROR; the rest still process."""
        mock_http()
        calls = {"n": 0}

        def handler(tier: str, system: str, user: str, model: type[Any]) -> Any:
            if model is Tier0Result:
                calls["n"] += 1
                if "Vague marketing" in user:
                    raise RuntimeError("simulated model explosion")
                return Tier0Result(relevance_guess=9, confidence=9)
            return Tier1Result(relevance_score=8, summary_md="Summary.")

        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, FakeLLM(handler), FakeASR())
            await runner.run_ingest()
            stats = await runner.run_pipeline()

        assert stats["errors"] == 1
        assert stats["triaged"] == 2  # the other two still went through
        failed = next(e for e in store.docs_of_type("episode") if "Vague" in e["title"])
        assert failed["status"] == S.ERROR.value
        assert failed["last_error"]["stage"] == "tier0"
        assert "simulated model explosion" in failed["last_error"]["message"]
        assert "Traceback" in failed["last_error"]["traceback"]

    @respx.mock
    async def test_dead_llm_defers_the_stage_and_keeps_work_queued(
        self, e2e_settings, store: MemoryStore
    ) -> None:
        """§10.6: with cloud fallback off, work must queue rather than be lost."""
        mock_http()
        llm = FakeLLM()
        llm.fail_with = LLMUnavailable("local model unreachable")
        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, llm, FakeASR())
            await runner.run_ingest()
            stats = await runner.run_pipeline()

        assert "triage" in stats["stages_deferred"]
        assert stats["errors"] == 0  # not the episodes' fault
        assert all(e["status"] == S.NEW.value for e in store.docs_of_type("episode"))

    @respx.mock
    async def test_batch_caps_are_respected(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "T", "feed_url": FEED_URL}],
            pipeline={"max_triage_per_run": 2},
        )
        mock_http()
        async with build_client() as client:
            runner = build_runner(settings, store, client, routing_llm(), FakeASR())
            await runner.run_ingest()
            stats = await runner.run_pipeline()
        assert stats["triaged"] == 2
        assert sum(1 for e in store.docs_of_type("episode") if e["status"] == S.NEW.value) == 1


class TestJobExclusion:
    async def test_overlapping_run_is_rejected(self, e2e_settings, store: MemoryStore) -> None:
        """§11: the same job must never run twice concurrently."""
        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, FakeLLM(), FakeASR())
            await runner._locks["pipeline"].acquire()
            try:
                with pytest.raises(JobBusy):
                    await runner.run_pipeline()
                assert runner.is_running("pipeline") is True
            finally:
                runner._locks["pipeline"].release()
            assert runner.is_running("pipeline") is False

    async def test_a_second_process_is_rejected_too(self, e2e_settings, store: MemoryStore) -> None:
        """The case the in-process lock cannot see.

        Two runners against one database is what a stray second instance, a CLI
        invocation or a container replica looks like. The asyncio locks are
        per-object, so both would have run — two backfills paying for the same
        episodes, two digests racing for the same claims.
        """
        async with build_client() as client:
            first = build_runner(e2e_settings, store, client, FakeLLM(), FakeASR())
            second = build_runner(e2e_settings, store, client, FakeLLM(), FakeASR())

            started = asyncio.Event()
            release = asyncio.Event()

            async def _hold() -> None:
                async with first._exclusive("pipeline"):
                    started.set()
                    await release.wait()

            holding = asyncio.create_task(_hold())
            await started.wait()
            try:
                with pytest.raises(JobBusy, match="another process"):
                    async with second._exclusive("pipeline"):
                        pass
            finally:
                release.set()
                await holding

            # And once the first lets go, the second may proceed.
            async with second._exclusive("pipeline"):
                pass

    async def test_the_lease_is_released_even_when_the_job_raises(
        self, e2e_settings, store: MemoryStore
    ) -> None:
        """Otherwise one crash locks the job out for a whole TTL."""
        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, FakeLLM(), FakeASR())
            with pytest.raises(RuntimeError):
                async with runner._exclusive("digest"):
                    raise RuntimeError("boom")
            assert await store.get(lock_id("digest")) is None

    @respx.mock
    async def test_last_run_summary_is_recorded(self, e2e_settings, store: MemoryStore) -> None:
        mock_http()
        async with build_client() as client:
            runner = build_runner(e2e_settings, store, client, routing_llm(), FakeASR())
            await runner.run_ingest()
        assert "ingest" in runner.last_runs
        assert runner.last_runs["ingest"]["episodes_created"] == 3
        assert "wall_ms" in runner.last_runs["ingest"]


class TestRetention:
    async def test_old_transcripts_are_deleted_but_summaries_kept(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, retention={"transcript_days": 180})
        from podcast_agent.db import save_transcript
        from podcast_agent.utils import iso

        old = make_episode(
            guid="old",
            status=S.PUBLISHED,
            transcript_at=iso(datetime.now(UTC) - timedelta(days=200)),
            tier1={"relevance_score": 9, "summary_md": "Kept forever."},
        )
        recent = make_episode(
            guid="recent",
            status=S.PUBLISHED,
            transcript_at=iso(datetime.now(UTC) - timedelta(days=10)),
        )
        store.seed(old, recent)
        await save_transcript(store, old["_id"], TRANSCRIPT_TEXT)
        await save_transcript(store, recent["_id"], TRANSCRIPT_TEXT)

        stats = await RetentionJob(settings, store).run()

        assert stats["transcripts_deleted"] == 1
        assert stats["transcript_bytes_freed"] > 0
        assert await load_transcript(store, old["_id"]) is None
        assert await load_transcript(store, recent["_id"]) is not None
        # The summary is the durable artefact and must survive.
        kept = await store.get(old["_id"])
        assert kept is not None
        assert kept["tier1"]["summary_md"] == "Kept forever."
        assert kept["transcript_expired_at"]

    async def test_zero_days_keeps_transcripts_indefinitely(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The corpus setting.

        Entity timelines and any retrieval over the archive reach back exactly
        as far as the transcripts still exist, so expiring them silently caps
        what those can ever answer. At ~14 KB gzipped an episode the storage
        this was protecting is not worth the capability it costs.
        """
        settings = make_settings(tmp_path, retention={"transcript_days": 0})
        from podcast_agent.db import save_transcript
        from podcast_agent.utils import iso

        ancient = make_episode(
            guid="ancient",
            status=S.PUBLISHED,
            transcript_at=iso(datetime.now(UTC) - timedelta(days=4000)),
        )
        store.seed(ancient)
        await save_transcript(store, ancient["_id"], TRANSCRIPT_TEXT)

        stats = await RetentionJob(settings, store).run()

        assert stats["transcripts_deleted"] == 0
        assert await load_transcript(store, ancient["_id"]) is not None
        # Not merely undeleted — never considered, so nothing marks it expired.
        kept = await store.get(ancient["_id"])
        assert kept is not None
        assert "transcript_expired_at" not in kept

    async def test_zero_days_does_not_stop_the_other_sweeps(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Telemetry still expires; only transcripts are the durable corpus."""
        settings = make_settings(tmp_path, retention={"transcript_days": 0, "llm_call_days": 365})
        store.seed({"_id": "llmcall:old", "type": "llm_call", "ts": "2020-01-01T00:00:00+00:00"})
        stats = await RetentionJob(settings, store).run()
        assert stats["llm_calls_deleted"] == 1

    async def test_old_telemetry_is_deleted(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(tmp_path, retention={"llm_call_days": 365})
        store.seed(
            {"_id": "llmcall:old", "type": "llm_call", "ts": "2020-01-01T00:00:00+00:00"},
            {"_id": "llmcall:new", "type": "llm_call", "ts": "2026-07-29T00:00:00+00:00"},
        )
        stats = await RetentionJob(settings, store).run()
        assert stats["llm_calls_deleted"] == 1
        assert await store.get("llmcall:old") is None
        assert await store.get("llmcall:new") is not None

    async def test_orphan_audio_is_swept(self, tmp_path: Path, store: MemoryStore) -> None:
        """Audio left behind by a run that died mid-transcription."""
        import os
        import time

        settings = make_settings(tmp_path)
        audio_dir = settings.output.work_dir / "audio"
        audio_dir.mkdir(parents=True)
        stale = audio_dir / "stale.audio"
        stale.write_bytes(b"leftover")
        old_time = time.time() - 24 * 3600
        os.utime(stale, (old_time, old_time))
        fresh = audio_dir / "fresh.audio"
        fresh.write_bytes(b"in use")

        stats = await RetentionJob(settings, store).run()

        assert stats["orphan_audio_deleted"] == 1
        assert not stale.exists()
        assert fresh.exists()

    async def test_missing_work_dir_is_not_an_error(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        stats = await RetentionJob(make_settings(tmp_path), store).run()
        assert stats["orphan_audio_deleted"] == 0
        assert stats["error_count"] == 0
