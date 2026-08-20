"""Archive backfill tests (roadmap A1).

The properties that matter are economic and isolating: backfill must never spend
ASR time, never leak into the weekly digest, never run without confirmation, and
never let routine ingestion wander into the archive by accident.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from helpers import FakeLLM, make_episode, make_settings
from pydantic import ValidationError

from podcast_agent.backfill.control import get_state as get_control_state
from podcast_agent.backfill.control import is_paused, rewind_cursors
from podcast_agent.backfill.control import set_paused as set_control_paused
from podcast_agent.backfill.estimate import estimate_backfill
from podcast_agent.backfill.ingest import (
    BACKFILL_ORIGIN,
    BackfillIngestor,
    floor_from_anchor,
    floor_month,
    month_bounds,
    month_key,
    previous_month,
)
from podcast_agent.backfill.process import BackfillProcessor, BackfillProcessStats
from podcast_agent.db import MemoryStore, save_transcript
from podcast_agent.digest.archive import ArchiveDigestGenerator
from podcast_agent.digest.generate import DigestGenerator
from podcast_agent.ingest.feeds import Ingestor
from podcast_agent.models import Tier0Result, Tier1Result
from podcast_agent.net import UrlGuard, build_client
from podcast_agent.state import EpisodeStatus
from podcast_agent.summarize.tier1 import Tier1Stage
from podcast_agent.transcripts.acquire import TranscriptAcquirer
from podcast_agent.transcripts.asr import ASRUnavailable
from podcast_agent.transcripts.stage import TranscriptStage
from podcast_agent.triage.tier0 import Tier0Stage
from podcast_agent.utils import podcast_doc_id

S = EpisodeStatus
FEED_URL = "https://example.com/feed.xml"

#: Parses, contains nothing. Enough for a walk that is being tested for where it
#: stops rather than for what it ingests.
EMPTY_FEED = '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title></channel></rss>'


async def _seed_podcast_docs(settings, store: MemoryStore) -> None:
    """Create the podcast documents console overrides attach to."""
    async with build_client() as client:
        await Ingestor(settings, store, client, UrlGuard(settings.security)).seed_podcast_docs()


NOW = datetime(2026, 7, 30, tzinfo=UTC)
TRANSCRIPT = "A substantive archive transcript sentence about ICS security. " * 40


def feed_with(entries: list[dict[str, Any]]) -> str:
    items = []
    for e in entries:
        transcript = (
            f'<podcast:transcript url="{e["transcript"]}" type="text/plain"/>'
            if e.get("transcript")
            else ""
        )
        items.append(f"""<item>
          <title>{e["title"]}</title>
          <link>https://example.com/{e["guid"]}</link>
          <guid isPermaLink="false">{e["guid"]}</guid>
          <pubDate>{e["published"].strftime("%a, %d %b %Y %H:%M:%S +0000")}</pubDate>
          <description>{e.get("description", "A description.")}</description>
          <itunes:duration>1800</itunes:duration>
          <enclosure url="https://cdn-host.net/{e["guid"]}.mp3" length="1000" type="audio/mpeg"/>
          {transcript}
        </item>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
         xmlns:podcast="https://podcastindex.org/namespace/1.0">
      <channel><title>Test Show</title>{"".join(items)}</channel>
    </rss>"""


def archive_settings(tmp_path: Path, **over: Any):
    """Settings for a podcast the archive actually walks.

    `backfill_mode` defaults to `skip` — adding a podcast should not silently
    start walking its back catalogue — so a fixture about the walk has to opt in
    the way a person would.
    """
    podcasts = over.pop(
        "podcasts",
        [
            {
                "slug": "test-show",
                "name": "Test Show",
                "feed_url": FEED_URL,
                "backfill_mode": "full",
            }
        ],
    )
    return make_settings(tmp_path, podcasts=podcasts, **over)


def ingestor(settings, store) -> BackfillIngestor:
    return BackfillIngestor(settings, store, build_client(), UrlGuard(settings.security))


class TestMonthArithmetic:
    def test_previous_month_wraps_the_year(self) -> None:
        assert previous_month("2026-01") == "2025-12"
        assert previous_month("2026-07") == "2026-06"

    def test_month_bounds_are_half_open(self) -> None:
        start, end = month_bounds("2026-02")
        assert start == datetime(2026, 2, 1, tzinfo=UTC)
        assert end == datetime(2026, 3, 1, tzinfo=UTC)

    def test_december_bounds_roll_into_january(self) -> None:
        _, end = month_bounds("2026-12")
        assert end == datetime(2027, 1, 1, tzinfo=UTC)

    def test_floor_month_counts_back(self) -> None:
        assert floor_month(NOW, 12) == "2025-07"
        assert floor_month(NOW, 1) == "2026-06"

    def test_month_key(self) -> None:
        assert month_key(NOW) == "2026-07"


class TestCursorWalk:
    @respx.mock
    async def test_starts_at_the_month_before_now(self, tmp_path: Path, store: MemoryStore) -> None:
        """The current month belongs to routine ingestion, not the archive."""
        settings = archive_settings(tmp_path)
        respx.get(FEED_URL).mock(
            return_value=httpx.Response(
                200,
                text=feed_with(
                    [
                        {
                            "guid": "june",
                            "title": "June episode",
                            "published": datetime(2026, 6, 15, tzinfo=UTC),
                            "transcript": "https://transcript-host.net/june.txt",
                        },
                        {
                            "guid": "july",
                            "title": "July episode",
                            "published": datetime(2026, 7, 15, tzinfo=UTC),
                            "transcript": "https://transcript-host.net/july.txt",
                        },
                    ]
                ),
            )
        )
        store.seed({"_id": podcast_doc_id("test-show"), "type": "podcast", "slug": "test-show"})

        stats = await ingestor(settings, store).run(now=NOW)

        assert stats.episodes_created == 1
        titles = {e["title"] for e in store.docs_of_type("episode")}
        assert titles == {"June episode"}

    @respx.mock
    async def test_cursor_advances_backwards_each_run(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = archive_settings(tmp_path)
        entries = [
            {
                "guid": f"m{month}",
                "title": f"Month {month}",
                "published": datetime(2026, month, 15, tzinfo=UTC),
                "transcript": f"https://transcript-host.net/{month}.txt",
            }
            for month in (4, 5, 6)
        ]
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed_with(entries)))
        store.seed({"_id": podcast_doc_id("test-show"), "type": "podcast", "slug": "test-show"})

        back = ingestor(settings, store)
        await back.run(now=NOW)
        assert (await store.get(podcast_doc_id("test-show")))["backfill_cursor"] == "2026-05"  # type: ignore[index]
        await back.run(now=NOW)
        assert (await store.get(podcast_doc_id("test-show")))["backfill_cursor"] == "2026-04"  # type: ignore[index]

        assert {e["archive_month"] for e in store.docs_of_type("episode")} == {
            "2026-06",
            "2026-05",
        }

    @respx.mock
    async def test_stops_at_the_configured_floor(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = archive_settings(tmp_path, backfill={"months": 1})
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed_with([])))
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "backfill_cursor": "2025-01",
            }
        )
        stats = await ingestor(settings, store).run(now=NOW)
        assert stats.shows_complete == 1
        assert stats.months_processed == 0

    @respx.mock
    async def test_marks_the_show_complete_at_the_floor(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = archive_settings(tmp_path, backfill={"months": 1, "months_per_run": 2})
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed_with([])))
        store.seed({"_id": podcast_doc_id("test-show"), "type": "podcast", "slug": "test-show"})
        await ingestor(settings, store).run(now=NOW)
        pdoc = await store.get(podcast_doc_id("test-show"))
        assert pdoc is not None
        assert pdoc["backfill_complete"] is True


class TestTranscriptOnlyPolicy:
    @respx.mock
    async def test_records_episodes_without_a_transcript_too(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Knowing an episode exists and spending on it are separate decisions.

        These used to be discarded at ingestion, which is why a weekly podcast
        could show one episode: its whole back catalogue had been dropped with
        no record that it existed. They are now indexed; `require_transcript`
        governs what is *spent*, further down.
        """
        settings = archive_settings(tmp_path)
        respx.get(FEED_URL).mock(
            return_value=httpx.Response(
                200,
                text=feed_with(
                    [
                        {
                            "guid": "with",
                            "title": "Has transcript",
                            "published": datetime(2026, 6, 10, tzinfo=UTC),
                            "transcript": "https://transcript-host.net/a.txt",
                        },
                        {
                            "guid": "without",
                            "title": "No transcript",
                            "published": datetime(2026, 6, 11, tzinfo=UTC),
                        },
                    ]
                ),
            )
        )
        store.seed({"_id": podcast_doc_id("test-show"), "type": "podcast", "slug": "test-show"})

        stats = await ingestor(settings, store).run(now=NOW)

        assert stats.episodes_created == 2
        assert stats.without_transcript == 1
        assert {e["title"] for e in store.docs_of_type("episode")} == {
            "Has transcript",
            "No transcript",
        }
        # Recorded, but nothing to summarise from: acquisition has no candidate.
        bare = next(e for e in store.docs_of_type("episode") if e["title"] == "No transcript")
        assert bare["feed_transcripts"] == []

    @respx.mock
    async def test_policy_can_be_disabled(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = archive_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": FEED_URL,
                    "backfill_mode": "full",
                    "asr_enabled": True,
                }
            ],
        )
        respx.get(FEED_URL).mock(
            return_value=httpx.Response(
                200,
                text=feed_with(
                    [
                        {
                            "guid": "without",
                            "title": "No transcript",
                            "published": datetime(2026, 6, 11, tzinfo=UTC),
                        }
                    ]
                ),
            )
        )
        store.seed({"_id": podcast_doc_id("test-show"), "type": "podcast", "slug": "test-show"})
        stats = await ingestor(settings, store).run(now=NOW)
        assert stats.episodes_created == 1

    async def test_processing_never_permits_asr(self, tmp_path: Path, store: MemoryStore) -> None:
        """A backfill episode must not be able to start a transcription job."""
        settings = archive_settings(tmp_path)
        calls: list[bool] = []

        class RecordingStage:
            async def process(self, episode: dict[str, Any], *, allow_asr: bool = True):
                calls.append(allow_asr)
                return S.TRANSCRIPT_FAILED

        store.seed(
            make_episode(
                guid="a",
                status=S.AWAITING_TRANSCRIPT,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-06",
            )
        )
        processor = BackfillProcessor(
            settings,
            store,
            tier0=Tier0Stage(settings, store, FakeLLM()),
            transcripts=RecordingStage(),  # type: ignore[arg-type]
            tier1=Tier1Stage(settings, store, FakeLLM()),
        )
        await processor.run()
        assert calls == [False]


class TestPipelineIsolation:
    @respx.mock
    async def test_regular_ingest_never_walks_backwards(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Regression: once a show had any history the cutoff became None, so the
        NEXT routine poll ingested the whole feed page — months of back catalogue,
        triaged silently at LLM cost. Reaching back is backfill's job alone."""
        settings = make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "Test Show", "feed_url": FEED_URL}],
        )
        old = [
            {
                "guid": f"old{n}",
                "title": f"Old {n}",
                "published": datetime.now(UTC) - timedelta(days=120 + n),
            }
            for n in range(5)
        ]
        recent = [
            {
                "guid": "recent",
                "title": "Recent",
                "published": datetime.now(UTC) - timedelta(days=1),
            }
        ]
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed_with(recent + old)))

        async with build_client() as client:
            ing = Ingestor(settings, store, client, UrlGuard(settings.security))
            await ing.seed_podcast_docs()
            first = await ing.run()
            second = await ing.run()

        assert first.episodes_created == 1  # only the recent one
        # The second poll must not discover the back catalogue.
        assert second.episodes_created == 0

    async def test_regular_queues_exclude_archive_episodes(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = archive_settings(tmp_path)
        store.seed(
            make_episode(guid="normal", status=S.NEW),
            make_episode(
                guid="archive", status=S.NEW, origin=BACKFILL_ORIGIN, archive_month="2026-06"
            ),
        )
        runner = _runner(settings, store, FakeLLM())
        queued = await runner._queue(S.NEW, 50)
        assert [d["guid"] for d in queued] == ["normal"]

    async def test_weekly_digest_excludes_archive_episodes(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """A 2019 episode must never appear in this week's digest."""
        settings = archive_settings(tmp_path)
        published = datetime(2026, 7, 28, tzinfo=UTC)
        store.seed(
            make_episode(
                guid="normal",
                status=S.READY_FOR_DIGEST,
                published_at=published,
                tier1={"relevance_score": 8, "summary_basis": "transcript"},
            ),
            make_episode(
                guid="archive",
                status=S.READY_FOR_DIGEST,
                published_at=published,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-07",
                tier1={"relevance_score": 9, "summary_basis": "transcript"},
            ),
        )
        result = await DigestGenerator(settings, store).generate(
            since=datetime(2026, 7, 20, tzinfo=UTC)
        )
        assert len(result.episode_ids) == 1
        assert result.file_path is not None
        assert "archive" not in result.file_path.read_text()


class TestBackfillProcessing:
    async def test_tier0_only_shows_are_never_summarised(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """A stale daily-news episode gets an index line, not four minutes of LLM."""
        settings = archive_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": FEED_URL,
                    "backfill_mode": "tier0_only",
                }
            ],
        )
        store.seed(
            make_episode(
                guid="news",
                status=S.NEW,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-06",
                backfill_mode="tier0_only",
            )
        )
        llm = FakeLLM(lambda *a: Tier0Result(relevance_guess=9, confidence=9))
        processor = _processor(settings, store, llm)
        stats = await processor.run()

        assert stats.listed == 1
        doc = store.docs_of_type("episode")[0]
        assert doc["status"] == S.DIGEST_DIRECT.value
        assert doc["tier1"] is None

    async def test_full_mode_shows_are_summarised(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = archive_settings(tmp_path)
        episode = make_episode(
            guid="deep",
            status=S.TRANSCRIBED,
            origin=BACKFILL_ORIGIN,
            archive_month="2026-06",
            backfill_mode="full",
            transcript_source="feed",
        )
        store.seed(episode)
        await save_transcript(store, episode["_id"], TRANSCRIPT)

        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=8, summary_md="Archive summary."))
        stats = await _processor(settings, store, llm).run()

        assert stats.summarized == 1
        assert store.docs_of_type("episode")[0]["status"] == S.READY_FOR_DIGEST.value

    async def test_stricter_threshold_applies(self, tmp_path: Path, store: MemoryStore) -> None:
        """Score 6 clears the weekly threshold of 5 but not the archive's 7."""
        settings = archive_settings(
            tmp_path, pipeline={"digest_threshold": 5}, backfill={"digest_threshold": 7}
        )
        episode = make_episode(
            guid="mid",
            status=S.TRANSCRIBED,
            origin=BACKFILL_ORIGIN,
            archive_month="2026-06",
            transcript_source="feed",
        )
        store.seed(episode)
        await save_transcript(store, episode["_id"], TRANSCRIPT)

        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=6, summary_md="Middling."))
        stats = await _processor(settings, store, llm).run()

        assert stats.scored_low == 1
        assert store.docs_of_type("episode")[0]["status"] == S.SCORED_LOW.value


class TestArchiveDigest:
    async def _seed_month(self, store: MemoryStore, **over: Any) -> None:
        store.seed(
            make_episode(
                guid="sum",
                status=S.READY_FOR_DIGEST,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-06",
                published_at=datetime(2026, 6, 12, tzinfo=UTC),
                tier1={
                    "relevance_score": 9,
                    "summary_basis": "published_transcript",
                    "why_it_matters": "Still relevant to OT work.",
                    "summary_md": "A thorough discussion of PLC security.",
                    "key_takeaways": ["Segment OT networks"],
                    "matched_interests": ["ot_ics"],
                },
                **over,
            ),
            make_episode(
                guid="listed",
                status=S.DIGEST_DIRECT,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-06",
                published_at=datetime(2026, 6, 3, tzinfo=UTC),
                tier0={"relevance_guess": 5, "confidence": 8, "route": "DIGEST_DIRECT"},
            ),
        )

    async def test_writes_one_file_per_show_month(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = archive_settings(tmp_path)
        await self._seed_month(store)
        result = await ArchiveDigestGenerator(settings, store).generate()

        path = settings.output.digest_dir / "archive" / "test-show" / "2026-06.md"
        assert path.exists()
        assert result.files_written == ["archive/test-show/2026-06.md"]

        text = path.read_text()
        assert "type: podcast-archive" in text
        assert "month: 2026-06" in text
        assert "June 2026" in text
        assert "## Worth reading" in text
        assert "## Indexed only" in text
        # Honest about what it is.
        assert "as of the publication" in text

    async def test_incomplete_months_are_not_written(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """A file that silently omits still-queued episodes is worse than none."""
        settings = archive_settings(tmp_path)
        await self._seed_month(store)
        store.seed(
            make_episode(
                guid="pending",
                status=S.AWAITING_TRANSCRIPT,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-06",
            )
        )
        result = await ArchiveDigestGenerator(settings, store).generate()
        assert result.files_written == []
        assert result.months_skipped_incomplete == 1

    async def test_episodes_are_claimed_once(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = archive_settings(tmp_path)
        await self._seed_month(store)
        generator = ArchiveDigestGenerator(settings, store)
        await generator.generate()
        again = await generator.generate()
        assert again.files_written == []
        for doc in store.docs_of_type("episode"):
            assert doc["digest_id"] == "archive:test-show:2026-06"
            assert doc["status"] == S.PUBLISHED.value

    async def test_dry_run_writes_nothing(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = archive_settings(tmp_path)
        await self._seed_month(store)
        result = await ArchiveDigestGenerator(settings, store).generate(dry_run=True)
        assert not (settings.output.digest_dir / "archive").exists()
        assert result.episodes_published == 0


class TestEstimateAndConfirmation:
    async def test_refuses_to_run_without_confirmation(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Nobody should start a multi-hour archive job by fat-fingering a URL."""
        runner = _runner(archive_settings(tmp_path), store, FakeLLM())
        with pytest.raises(ValueError, match="confirm=true"):
            await runner.run_backfill(dry_run=False, confirm=False)

    async def test_estimate_says_so_when_there_is_no_telemetry(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Inventing a number would be worse than admitting ignorance."""
        estimate = await estimate_backfill(
            archive_settings(tmp_path), store, episodes_to_ingest=100, tier0_only_share=0.5
        )
        assert estimate["estimated_wall_hours"] == "no telemetry yet"
        assert "No LLM calls recorded" in estimate["note"]

    async def test_estimate_uses_real_telemetry(self, tmp_path: Path, store: MemoryStore) -> None:
        for index in range(20):
            store.seed(
                {
                    "_id": f"llmcall:{index}",
                    "type": "llm_call",
                    "tier": "tier0" if index % 2 else "tier1",
                    "latency_ms": 10_000 if index % 2 else 60_000,
                    "cost_usd": 0.0,
                    "ts": f"2026-07-{index + 1:02d}T00:00:00+00:00",
                }
            )
        estimate = await estimate_backfill(
            archive_settings(tmp_path), store, episodes_to_ingest=100, tier0_only_share=1.0
        )
        # All tier0_only: 100 triage calls at 10s, no Tier-1.
        assert estimate["tier0_calls"] == 100
        assert estimate["tier1_calls"] == 0
        assert estimate["estimated_wall_hours"] == pytest.approx(100 * 10_000 / 3_600_000, rel=0.1)

    async def test_estimate_reports_asr_is_impossible(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        estimate = await estimate_backfill(
            archive_settings(tmp_path), store, episodes_to_ingest=10, tier0_only_share=0.0
        )
        # No longer a single number: whether a podcast's back catalogue is
        # transcribed is that podcast's own setting.
        assert "per podcast" in str(estimate["asr_jobs"])


class TestASleepingTranscriberIsNotTheEpisodesFault:
    """A dead ASR backend must defer the stage, never condemn the episodes.

    `TranscriptStage.process` already gets this right: on `ASRUnavailable` it
    puts the episode back to AWAITING_TRANSCRIPT, spends none of its retry
    budget, and re-raises so the caller can stop. The routine pipeline honours
    that re-raise. Backfill did not — its blanket `except Exception` marked the
    episode ERROR, undoing the stage's work one frame up and leaving every
    archive episode the walk touched needing a hand-written retry.

    That mattered because the machine answering is a laptop that sleeps.
    """

    def _processor(self, settings, store, *, acquirer) -> BackfillProcessor:
        llm = FakeLLM(lambda *a: Tier1Result(relevance_score=8, summary_md="x"))
        return BackfillProcessor(
            settings,
            store,
            tier0=Tier0Stage(settings, store, llm),
            transcripts=TranscriptStage(settings, store, acquirer),
            tier1=Tier1Stage(settings, store, llm),
        )

    def _settings(self, tmp_path: Path):
        return archive_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": FEED_URL,
                    "backfill_mode": "full",
                    "asr_enabled": True,
                }
            ],
        )

    async def test_the_episode_stays_queued_instead_of_erroring(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        calls = 0

        class DeadASR:
            async def acquire(self, episode: dict[str, Any], *, allow_asr: bool = True) -> Any:
                nonlocal calls
                calls += 1
                raise ASRUnavailable("remote:http://transcriber.local:8000 unreachable")

        store.seed(
            make_episode(
                guid="asleep",
                status=S.AWAITING_TRANSCRIPT,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-06",
                backfill_mode="full",
            )
        )
        settings = self._settings(tmp_path)
        stats = await self._processor(settings, store, acquirer=DeadASR()).run()

        doc = store.docs_of_type("episode")[0]
        assert doc["status"] == S.AWAITING_TRANSCRIPT.value
        assert doc.get("last_error") is None
        # The whole point: no retry budget spent on an operator problem.
        assert (doc.get("attempts") or {}).get("transcript", 0) == 0
        assert stats.errors == 0
        assert "transcribe" in stats.deferred
        assert calls == 1

    async def test_the_stage_stops_rather_than_walking_the_queue(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """One dead endpoint is learned once, not once per episode."""
        calls = 0

        class DeadASR:
            async def acquire(self, episode: dict[str, Any], *, allow_asr: bool = True) -> Any:
                nonlocal calls
                calls += 1
                raise ASRUnavailable("unreachable")

        for n in range(3):
            store.seed(
                make_episode(
                    guid=f"asleep-{n}",
                    status=S.AWAITING_TRANSCRIPT,
                    origin=BACKFILL_ORIGIN,
                    archive_month="2026-06",
                    backfill_mode="full",
                )
            )
        settings = self._settings(tmp_path)
        stats = await self._processor(settings, store, acquirer=DeadASR()).run()

        assert calls == 1
        assert stats.errors == 0
        assert all(
            d["status"] == S.AWAITING_TRANSCRIPT.value for d in store.docs_of_type("episode")
        )


# --- helpers ------------------------------------------------------------------


def _processor(settings, store, llm) -> BackfillProcessor:
    client = build_client()
    guard = UrlGuard(settings.security)
    return BackfillProcessor(
        settings,
        store,
        tier0=Tier0Stage(settings, store, llm),
        transcripts=TranscriptStage(
            settings,
            store,
            TranscriptAcquirer(settings, store, client, guard, None),  # type: ignore[arg-type]
        ),
        tier1=Tier1Stage(settings, store, llm),
    )


def _runner(settings, store, llm):
    from podcast_agent.pipeline.runner import PipelineRunner

    client = build_client()
    guard = UrlGuard(settings.security)
    tier0 = Tier0Stage(settings, store, llm)
    transcripts = TranscriptStage(
        settings,
        store,
        TranscriptAcquirer(settings, store, client, guard, None),  # type: ignore[arg-type]
    )
    tier1 = Tier1Stage(settings, store, llm)
    return PipelineRunner(
        settings,
        store,
        ingestor=Ingestor(settings, store, client, guard),
        tier0=tier0,
        transcripts=transcripts,
        tier1=tier1,
        digest=DigestGenerator(settings, store),
        backfill_ingest=BackfillIngestor(settings, store, client, guard),
        backfill_process=BackfillProcessor(
            settings, store, tier0=tier0, transcripts=transcripts, tier1=tier1
        ),
        archive=ArchiveDigestGenerator(settings, store),
    )


class TestArchiveCrashSafety:
    """Stopping mid-run is expected — the whole design is resumable — so an
    interrupted archive write must not produce a duplicate file on the next run.
    """

    async def _seed(self, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="one",
                status=S.READY_FOR_DIGEST,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-06",
                published_at=datetime(2026, 6, 12, tzinfo=UTC),
                tier1={"relevance_score": 9, "summary_basis": "published_transcript"},
            )
        )

    async def test_interrupted_claiming_is_reconciled_not_duplicated(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = archive_settings(tmp_path)
        await self._seed(store)
        generator = ArchiveDigestGenerator(settings, store)
        await generator.generate()

        archive_dir = settings.output.digest_dir / "archive" / "test-show"
        assert [p.name for p in archive_dir.iterdir()] == ["2026-06.md"]

        # Simulate a kill between writing the file and claiming the episodes.
        doc = store.docs_of_type("episode")[0]
        stored = await store.get(doc["_id"])
        assert stored is not None
        stored["digest_id"] = None
        stored["status"] = S.READY_FOR_DIGEST.value
        await store.put(stored)
        archive_doc = await store.get("archive:test-show:2026-06")
        assert archive_doc is not None
        archive_doc["marking_complete"] = False
        await store.put(archive_doc)

        result = await generator.generate()

        # No second file for the same month.
        assert sorted(p.name for p in archive_dir.iterdir()) == ["2026-06.md"]
        assert result.reconciled == 1
        finished = await store.get(doc["_id"])
        assert finished is not None
        assert finished["status"] == S.PUBLISHED.value

    async def test_archive_doc_records_what_was_written(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = archive_settings(tmp_path)
        await self._seed(store)
        await ArchiveDigestGenerator(settings, store).generate()

        doc = await store.get("archive:test-show:2026-06")
        assert doc is not None
        assert doc["type"] == "archive"
        assert doc["file_path"] == "archive/test-show/2026-06.md"
        assert doc["marking_complete"] is True
        assert len(doc["episode_ids"]) == 1


class TestPauseControl:
    """Pause is the control that makes a multi-hour job usable: it must stop
    promptly, lose no work, and never start the unattended walk by surprise."""

    async def test_defaults_to_paused(self, store: MemoryStore) -> None:
        """A deployed config must not silently begin walking the archive."""
        state = await get_control_state(store)
        assert state["paused"] is True
        assert state["note"] == "never started"

    async def test_resume_and_pause_round_trip(self, store: MemoryStore) -> None:
        await set_control_paused(store, False, note="starting")
        assert await is_paused(store) is False
        await set_control_paused(store, True, note="enough")
        state = await get_control_state(store)
        assert state["paused"] is True
        assert state["note"] == "enough"
        assert state["updated_at"]

    async def test_scheduled_run_does_nothing_while_paused(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        runner = _runner(archive_settings(tmp_path), store, FakeLLM())
        assert await runner.run_backfill_scheduled() == {"skipped": "paused"}

    @respx.mock
    async def test_scheduled_run_proceeds_once_resumed(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed_with([])))
        await set_control_paused(store, False)
        runner = _runner(archive_settings(tmp_path), store, FakeLLM())
        result = await runner.run_backfill_scheduled()
        assert result.get("skipped") is None
        assert result["dry_run"] is False

    @respx.mock
    async def test_pausing_stops_ingestion_between_shows(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Between shows the cursor is already written, so nothing is lost."""
        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "a",
                    "name": "A",
                    "feed_url": "https://a.example.com/f.xml",
                    "backfill_mode": "full",
                },
                {
                    "slug": "b",
                    "name": "B",
                    "feed_url": "https://b.example.com/f.xml",
                    "backfill_mode": "full",
                },
            ],
        )
        for host in ("a", "b"):
            respx.get(f"https://{host}.example.com/f.xml").mock(
                return_value=httpx.Response(200, text=feed_with([]))
            )
        back = BackfillIngestor(settings, store, build_client(), UrlGuard(settings.security))

        calls = {"n": 0}

        async def stop_after_first() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        stats = await back.run(now=NOW, should_stop=stop_after_first)

        assert stats.stopped_early is True
        assert stats.feeds_examined == 1  # the second show was never fetched

    async def test_pausing_stops_processing_between_episodes(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The in-flight episode finishes; the next one is not started."""
        settings = archive_settings(tmp_path)
        for index in range(4):
            store.seed(
                make_episode(
                    guid=f"e{index}",
                    status=S.NEW,
                    origin=BACKFILL_ORIGIN,
                    archive_month="2026-06",
                    published_at=datetime(2026, 6, index + 1, tzinfo=UTC),
                )
            )
        llm = FakeLLM(lambda *a: Tier0Result(relevance_guess=2, confidence=9))
        processor = _processor(settings, store, llm)

        seen = {"n": 0}

        async def stop_after_two() -> bool:
            seen["n"] += 1
            return seen["n"] > 2

        stats = await processor.run(should_stop=stop_after_two)

        assert stats.stopped_early is True
        assert stats.triaged == 2
        # The untouched episodes are still queued, not lost or half-written.
        remaining = [d for d in store.docs_of_type("episode") if d["status"] == S.NEW.value]
        assert len(remaining) == 2

    @respx.mock
    async def test_manual_run_is_not_blocked_by_pause(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Pausing stops the unattended walk; an explicit run is still explicit."""
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed_with([])))
        await set_control_paused(store, True)
        runner = _runner(archive_settings(tmp_path), store, FakeLLM())
        result = await runner.run_backfill(dry_run=True)
        assert result["dry_run"] is True


class TestRecentWorkComesFirst:
    """The present outranks the past.

    A digest is a weekly artefact, so an episode published this morning is worth
    more than one from 2019 — and a five-hour local transcription of a fresh
    episode outranks a thousand archive items that would each finish in seconds.
    Speed is not the ordering; recency is.
    """

    def _pending(self, status: S, **over: Any) -> dict[str, Any]:
        return make_episode(
            guid=over.pop("guid", "fresh"),
            title=over.pop("title", "Fresh episode"),
            status=status,
            published_at=datetime.now(UTC) - timedelta(hours=2),
            **over,
        )

    async def test_scheduled_backfill_waits_for_recent_episodes(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path)
        await set_control_paused(store, False)
        store.seed(self._pending(S.NEW))
        result = await _runner(settings, store, FakeLLM()).run_backfill_scheduled()
        assert result["skipped"] == "recent episodes still in the pipeline"
        assert result["pending"] == 1

    @pytest.mark.parametrize(
        "status",
        [S.NEW, S.TRIAGED, S.AWAITING_TRANSCRIPT, S.TRANSCRIBED, S.TRANSCRIPT_FAILED, S.SUMMARIZED],
    )
    async def test_every_unfinished_stage_holds_history_back(
        self, tmp_path: Path, store: MemoryStore, status: S
    ) -> None:
        settings = make_settings(tmp_path)
        await set_control_paused(store, False)
        store.seed(self._pending(status))
        result = await _runner(settings, store, FakeLLM()).run_backfill_scheduled()
        assert result.get("skipped") == "recent episodes still in the pipeline", (
            f"{status.value} did not hold back the archive walk"
        )

    async def test_an_episode_queued_for_local_transcription_blocks_the_archive(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The explicit requirement: slow ASR beats fast archive items.

        An archive month of transcript-only episodes would finish in seconds
        while this one episode occupies a GPU for an hour. It still waits.
        """
        settings = make_settings(tmp_path)
        store.seed(self._pending(S.AWAITING_TRANSCRIPT, title="Needs ASR"))
        result = await _runner(settings, store, FakeLLM()).run_backfill(dry_run=False, confirm=True)
        assert result["skipped"] == "recent episodes still in the pipeline"
        assert result["waiting_on"][0]["title"] == "Needs ASR"
        assert result["waiting_on"][0]["status"] == S.AWAITING_TRANSCRIPT.value

    async def test_archive_episodes_do_not_block_themselves(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Backfill creates episodes in these same statuses.

        Counting them as "recent work" would mean the walk stopped itself dead
        after its first ingested episode and never resumed.
        """
        settings = make_settings(tmp_path)
        await set_control_paused(store, False)
        for i in range(3):
            store.seed(
                make_episode(
                    guid=f"archive-{i}",
                    title=f"Archive {i}",
                    status=S.NEW,
                    published_at=datetime(2019, 3, 1, tzinfo=UTC),
                    origin=BACKFILL_ORIGIN,
                )
            )
        result = await _runner(settings, store, FakeLLM()).run_backfill_scheduled()
        assert "skipped" not in result

    async def test_a_hard_failed_episode_does_not_block_history_forever(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """ERROR needs a person. One poisoned document must not halt the archive."""
        settings = make_settings(tmp_path)
        await set_control_paused(store, False)
        store.seed(self._pending(S.ERROR, title="Broken"))
        result = await _runner(settings, store, FakeLLM()).run_backfill_scheduled()
        assert "skipped" not in result

    async def test_settled_episodes_do_not_block(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(tmp_path)
        await set_control_paused(store, False)
        for i, status in enumerate((S.PUBLISHED, S.DROPPED, S.SCORED_LOW, S.READY_FOR_DIGEST)):
            store.seed(self._pending(status, guid=f"done-{i}"))
        result = await _runner(settings, store, FakeLLM()).run_backfill_scheduled()
        assert "skipped" not in result

    async def test_a_dry_run_still_reports_an_estimate(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Dry runs write nothing and spend nothing, so they need not wait."""
        settings = make_settings(tmp_path)
        store.seed(self._pending(S.AWAITING_TRANSCRIPT))
        result = await _runner(settings, store, FakeLLM()).run_backfill(dry_run=True)
        assert "skipped" not in result
        assert "estimate" in result

    async def test_force_overrides_the_wait(self, tmp_path: Path, store: MemoryStore) -> None:
        """A machine's owner is not locked out of their archive by one stuck episode."""
        settings = make_settings(tmp_path)
        store.seed(self._pending(S.AWAITING_TRANSCRIPT))
        result = await _runner(settings, store, FakeLLM()).run_backfill(
            dry_run=False, confirm=True, force=True
        )
        assert "skipped" not in result

    async def test_a_walk_in_flight_yields_when_a_fresh_episode_lands(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Handover happens at the next episode boundary, not at the end of history."""
        from podcast_agent.pipeline.runner import RecentWorkCheck

        check = RecentWorkCheck(store, cache_seconds=0.0)
        assert await check.should_stop() is False

        store.seed(self._pending(S.NEW))
        assert await check.should_stop() is True

    async def test_the_check_survives_a_storage_failure(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """A database blip must not silently halt a long-running walk."""
        from podcast_agent.pipeline.runner import RecentWorkCheck

        class Broken:
            async def find(self, *a: Any, **k: Any) -> Any:
                raise RuntimeError("couch is down")

        check = RecentWorkCheck(Broken(), cache_seconds=0.0)  # type: ignore[arg-type]
        assert await check.should_stop() is False


class TestQueueOrdering:
    """Within the routine pipeline, newest first at every stage."""

    async def test_the_queue_hands_back_the_newest_episode_first(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path)
        now = datetime.now(UTC)
        for days in (30, 1, 14, 3):
            store.seed(
                make_episode(
                    guid=f"ep-{days}",
                    title=f"{days} days old",
                    status=S.NEW,
                    published_at=now - timedelta(days=days),
                )
            )
        runner = _runner(settings, store, FakeLLM())
        queued = await runner._queue(S.NEW, limit=10)
        assert [d["title"] for d in queued] == [
            "1 days old",
            "3 days old",
            "14 days old",
            "30 days old",
        ]

    async def test_a_capped_run_spends_its_budget_on_the_newest(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The point of the ordering: when capacity is short, recency wins."""
        settings = make_settings(tmp_path)
        now = datetime.now(UTC)
        for days in (60, 45, 2, 1):
            store.seed(
                make_episode(
                    guid=f"ep-{days}",
                    title=f"{days} days old",
                    status=S.NEW,
                    published_at=now - timedelta(days=days),
                )
            )
        runner = _runner(settings, store, FakeLLM())
        queued = await runner._queue(S.NEW, limit=2)
        assert [d["title"] for d in queued] == ["1 days old", "2 days old"]

    async def test_archive_episodes_are_not_in_the_routine_queue(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path)
        store.seed(
            make_episode(
                guid="archive",
                title="Archive item",
                status=S.NEW,
                published_at=datetime.now(UTC),
                origin=BACKFILL_ORIGIN,
            )
        )
        store.seed(
            make_episode(
                guid="fresh",
                title="Fresh item",
                status=S.NEW,
                published_at=datetime.now(UTC) - timedelta(days=1),
            )
        )
        runner = _runner(settings, store, FakeLLM())
        queued = await runner._queue(S.NEW, limit=10)
        assert [d["title"] for d in queued] == ["Fresh item"]


class TestEveryEpisodeCarriesAnOrigin:
    """Selecting routine episodes, by a field rather than by its absence.

    History: `{"origin": {"$ne": "backfill"}}` matched no routine episode at
    all, because CouchDB has no index entry for a document lacking the field and
    every comparison against a missing field fails — negative ones included.
    The `$or` with `$exists: false` that replaced it was correct but
    unindexable, so every pipeline query scanned a range and filtered in memory.

    Both problems came from routine episodes having no `origin`. They have one
    now, so the selector is a plain equality match, and the guarantee that makes
    it safe is tested here rather than assumed.
    """

    async def test_ingestion_stamps_an_origin(self, tmp_path: Path) -> None:
        """The half of the guarantee that covers everything written from now on."""
        import inspect

        from podcast_agent.ingest import feeds

        source = inspect.getsource(feeds.Ingestor._ingest_entry)
        assert '"origin": ROUTINE_ORIGIN' in source

    async def test_the_selector_is_an_equality_match(self) -> None:
        """An `$or` or a `$ne` here means the scans are back."""
        from podcast_agent.state import ROUTINE_ONLY

        assert ROUTINE_ONLY == {"origin": "routine"}

    async def test_the_queue_finds_a_freshly_ingested_episode(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path)
        store.seed(
            make_episode(guid="fresh", title="fresh", status=S.NEW, published_at=datetime.now(UTC))
        )
        queued = await _runner(settings, store, FakeLLM())._queue(S.NEW, limit=10)
        assert [d["guid"] for d in queued] == ["fresh"]

    async def test_archive_material_is_still_excluded(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The clause must keep doing the job it was added for."""
        settings = make_settings(tmp_path)
        store.seed(
            make_episode(guid="fresh", title="fresh", status=S.NEW, published_at=datetime.now(UTC))
        )
        store.seed(
            make_episode(
                guid="archive",
                title="archive",
                status=S.NEW,
                published_at=datetime.now(UTC),
                origin=BACKFILL_ORIGIN,
            )
        )
        queued = await _runner(settings, store, FakeLLM())._queue(S.NEW, limit=10)
        assert [d["guid"] for d in queued] == ["fresh"]

    async def test_an_episode_predating_the_field_is_migrated(self, store: MemoryStore) -> None:
        """The other half: everything already in the database."""
        from podcast_agent.migrate import backfill_origins

        doc = make_episode(guid="old", title="old", status=S.NEW)
        doc.pop("origin")
        store.seed(doc)

        result = await backfill_origins(store)
        assert result == {"updated": 1, "remaining": 0}
        migrated = next(iter(store.docs_of_type("episode")))
        assert migrated["origin"] == "routine"

    async def test_migration_does_not_mislabel_archive_material(self, store: MemoryStore) -> None:
        """A mislabelled archive episode would leak into the weekly digest."""
        from podcast_agent.migrate import backfill_origins

        doc = make_episode(guid="old-archive", title="old", status=S.NEW)
        doc.pop("origin")
        doc["archive_month"] = "2025-09"
        store.seed(doc)

        await backfill_origins(store)
        migrated = next(iter(store.docs_of_type("episode")))
        assert migrated["origin"] == BACKFILL_ORIGIN

    async def test_migration_is_idempotent(self, store: MemoryStore) -> None:
        """It runs on every boot, so a completed one must cost a single query."""
        from podcast_agent.migrate import backfill_origins

        doc = make_episode(guid="old", title="old", status=S.NEW)
        doc.pop("origin")
        store.seed(doc)

        assert (await backfill_origins(store))["updated"] == 1
        assert (await backfill_origins(store))["updated"] == 0

    async def test_a_migrated_episode_is_then_visible_to_the_queue(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """End to end: the reason the migration runs before the scheduler."""
        from podcast_agent.migrate import backfill_origins

        settings = make_settings(tmp_path)
        doc = make_episode(guid="old", title="old", status=S.NEW, published_at=datetime.now(UTC))
        doc.pop("origin")
        store.seed(doc)

        runner = _runner(settings, store, FakeLLM())
        assert await runner._queue(S.NEW, limit=10) == []
        await backfill_origins(store)
        assert [d["guid"] for d in await runner._queue(S.NEW, limit=10)] == ["old"]


class TestPerPodcastArchiveWindow:
    """How far back the walk reaches, chosen per podcast.

    An evergreen interview show can be worth three years of archive while a
    daily news show is not worth one, so this is not a single global number.
    """

    def _settings(self, tmp_path: Path, **months: Any):
        return make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": FEED_URL,
                    "backfill_mode": "full",
                    **({"backfill_months": months["test"]} if "test" in months else {}),
                },
                {
                    "slug": "other-show",
                    "name": "Other Show",
                    "feed_url": "https://other.example/feed.xml",
                    "backfill_mode": "full",
                    **({"backfill_months": months["other"]} if "other" in months else {}),
                },
            ],
        )

    def test_it_defaults_to_the_configured_window(self, tmp_path: Path) -> None:
        """ "They are all defaulted to 12" — with nothing set per podcast."""
        settings = self._settings(tmp_path)
        for podcast in settings.podcasts:
            assert podcast.backfill_months is None
        assert settings.backfill.months == 12

    @pytest.mark.parametrize("months", [12, 24, 36])
    def test_every_offered_choice_is_accepted(self, tmp_path: Path, months: int) -> None:
        settings = self._settings(tmp_path, test=months)
        assert settings.podcasts[0].backfill_months == months

    @pytest.mark.parametrize("months", [1, 6, 18, 48, 0, -12])
    def test_a_value_outside_the_list_is_refused(self, tmp_path: Path, months: int) -> None:
        with pytest.raises(ValidationError):
            self._settings(tmp_path, test=months)

    def test_one_podcast_can_reach_further_than_another(self, tmp_path: Path) -> None:
        """The whole point: the setting is per podcast, not global."""
        settings = self._settings(tmp_path, test=36)
        default = settings.backfill.months
        now = datetime(2026, 7, 15, tzinfo=UTC)

        by_slug = {p.slug: p for p in settings.podcasts}
        reach = {
            slug: floor_month(now, p.backfill_months or default) for slug, p in by_slug.items()
        }
        assert reach["test-show"] == "2023-07"  # 36 months
        assert reach["other-show"] == "2025-07"  # inherits 12

    async def test_the_registry_carries_it_through(self, tmp_path: Path) -> None:
        from podcast_agent.podcasts import PodcastRegistry

        registry = PodcastRegistry(self._settings(tmp_path, test=24))
        by_slug = {p.slug: p for p in registry.all_podcasts()}
        assert by_slug["test-show"].backfill_months == 24
        assert by_slug["other-show"].backfill_months is None

    async def test_a_console_override_reaches_the_registry(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Set from the Podcasts page, effective at the next run."""
        from podcast_agent.podcasts import PodcastRegistry, set_overrides

        settings = self._settings(tmp_path)
        registry = PodcastRegistry(settings)
        await _seed_podcast_docs(settings, store)
        await registry.refresh(store)
        await set_overrides(store, "test-show", {"backfill_months": 36})
        await registry.refresh(store)

        record = next(p for p in registry.all_podcasts() if p.slug == "test-show")
        assert record.backfill_months == 36
        assert "backfill_months" in record.overridden

    async def test_reverting_the_override_returns_it_to_inheriting(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        from podcast_agent.podcasts import PodcastRegistry, clear_override, set_overrides

        settings = self._settings(tmp_path)
        registry = PodcastRegistry(settings)
        await _seed_podcast_docs(settings, store)
        await registry.refresh(store)
        await set_overrides(store, "test-show", {"backfill_months": 24})
        await clear_override(store, "test-show", "backfill_months")
        await registry.refresh(store)

        record = next(p for p in registry.all_podcasts() if p.slug == "test-show")
        assert record.backfill_months is None

    @respx.mock
    async def test_the_walk_stops_at_this_podcasts_own_floor(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The floor is what actually bounds a walk, so it is what is checked.

        A podcast whose cursor already predates its own window is complete, even
        though a wider-windowed podcast alongside it is not.
        """
        settings = self._settings(tmp_path, test=12)
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                # Older than a 12-month window from now, inside a 36-month one.
                "backfill_cursor": month_key(datetime.now(UTC) - timedelta(days=700)),
            }
        )
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=EMPTY_FEED))
        respx.get("https://other.example/feed.xml").mock(
            return_value=httpx.Response(200, text=EMPTY_FEED)
        )
        async with build_client() as client:
            ingestor = BackfillIngestor(settings, store, client, UrlGuard(settings.security))
            stats = await ingestor.run(dry_run=False)
        assert stats.shows_complete >= 1


class TestIndexedButNotSummarised:
    """Every archive episode is recorded and triaged; only some are summarised.

    The cost boundary moved from ingestion to dispatch. If it were not enforced
    here, an escalated episode with no transcript would go to
    AWAITING_TRANSCRIPT, fail acquisition, land at TRANSCRIPT_FAILED and then
    draw a Tier-1 call to summarise its description — the most expensive stage,
    for the least material, on the ~92% of archive episodes that publish no
    transcript.
    """

    def _episode(self, guid: str, *, transcripts: list[dict[str, str]], route: str):
        return make_episode(
            guid=guid,
            title=guid,
            status=S.TRIAGED,
            published_at=datetime(2026, 6, 10, tzinfo=UTC),
            origin=BACKFILL_ORIGIN,
            backfill_mode="full",
            feed_transcripts=transcripts,
            tier0={"relevance_guess": 9, "confidence": 9, "route": route},
        )

    async def _dispatch(self, tmp_path: Path, store: MemoryStore, *, asr_enabled: bool = False):
        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": "https://example.com/feed.xml",
                    "backfill_mode": "full",
                    "asr_enabled": asr_enabled,
                }
            ],
        )
        llm = FakeLLM()
        processor = BackfillProcessor(
            settings,
            store,
            tier0=Tier0Stage(settings, store, llm),
            transcripts=TranscriptStage(
                settings,
                store,
                TranscriptAcquirer(  # type: ignore[arg-type]
                    settings, store, build_client(), UrlGuard(settings.security), None
                ),
            ),
            tier1=Tier1Stage(settings, store, llm),
        )
        stats = BackfillProcessStats()
        await processor._dispatch(stats)
        return stats

    async def test_an_escalation_without_a_transcript_becomes_an_index_entry(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        store.seed(self._episode("bare", transcripts=[], route="ESCALATE"))
        stats = await self._dispatch(tmp_path, store)

        doc = next(iter(store.docs_of_type("episode")))
        assert doc["status"] == S.DIGEST_DIRECT.value, "would have drawn a Tier-1 call"
        assert stats.listed_no_transcript == 1

    async def test_the_downgrade_is_recorded_on_the_episode(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Otherwise the document asserts something that never happened.

        The downgrade is decided after triage, so `tier0.route` still reads
        ESCALATE for an episode that was never escalated. A reader then finds an
        episode published with no summary and no explanation — which reads as
        work still pending rather than a decision already taken, and labelled
        280 finished archive episodes "queued".
        """
        store.seed(self._episode("bare", transcripts=[], route="ESCALATE"))
        await self._dispatch(tmp_path, store)

        doc = next(iter(store.docs_of_type("episode")))
        assert "no transcript" in doc["indexed_only"]
        assert "local transcription is off" in doc["indexed_only"]

    async def test_a_news_cadence_downgrade_says_so_instead(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Two different reasons to index rather than summarise, and the reader
        cannot act on the right one without knowing which applied."""
        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": "https://example.com/feed.xml",
                    "backfill_mode": "tier0_only",
                    "asr_enabled": True,
                }
            ],
        )
        store.seed(
            self._episode(
                "newsy",
                transcripts=[{"url": "https://transcript-host.net/a.txt", "type": "text/plain"}],
                route="ESCALATE",
            )
        )
        llm = FakeLLM()
        processor = BackfillProcessor(
            settings,
            store,
            tier0=Tier0Stage(settings, store, llm),
            transcripts=TranscriptStage(
                settings,
                store,
                TranscriptAcquirer(  # type: ignore[arg-type]
                    settings, store, build_client(), UrlGuard(settings.security), None
                ),
            ),
            tier1=Tier1Stage(settings, store, llm),
        )
        await processor._dispatch(BackfillProcessStats())

        doc = next(iter(store.docs_of_type("episode")))
        assert "news cadence" in doc["indexed_only"]

    async def test_an_episode_that_was_not_downgraded_carries_no_reason(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The marker means "a decision was taken here", so it must not appear
        on an episode that simply followed its triage verdict."""
        store.seed(
            self._episode(
                "rich",
                transcripts=[{"url": "https://transcript-host.net/a.txt", "type": "text/plain"}],
                route="ESCALATE",
            )
        )
        await self._dispatch(tmp_path, store)

        doc = next(iter(store.docs_of_type("episode")))
        assert "indexed_only" not in doc

    async def test_an_escalation_with_a_transcript_still_escalates(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The downgrade must not swallow the episodes the archive exists for."""
        store.seed(
            self._episode(
                "rich",
                transcripts=[{"url": "https://transcript-host.net/a.txt", "type": "text/plain"}],
                route="ESCALATE",
            )
        )
        await self._dispatch(tmp_path, store)

        doc = next(iter(store.docs_of_type("episode")))
        assert doc["status"] == S.AWAITING_TRANSCRIPT.value

    async def test_turning_on_local_transcription_lets_them_escalate(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The podcast toggle is what buys transcription of its back catalogue."""
        store.seed(self._episode("bare", transcripts=[], route="ESCALATE"))
        await self._dispatch(tmp_path, store, asr_enabled=True)

        doc = next(iter(store.docs_of_type("episode")))
        assert doc["status"] == S.AWAITING_TRANSCRIPT.value

    async def test_a_drop_verdict_is_untouched(self, tmp_path: Path, store: MemoryStore) -> None:
        store.seed(self._episode("dull", transcripts=[], route="DROP"))
        stats = await self._dispatch(tmp_path, store)

        doc = next(iter(store.docs_of_type("episode")))
        assert doc["status"] == S.DROPPED.value
        assert stats.listed_no_transcript == 0

    async def test_the_estimate_does_not_bill_for_summaries_that_cannot_happen(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Most archive episodes carry no transcript; ignoring that inflates it."""
        settings = make_settings(tmp_path)
        with_share = await estimate_backfill(
            settings,
            store,
            episodes_to_ingest=100,
            tier0_only_share=0.0,
            without_transcript_share=0.92,
        )
        without_share = await estimate_backfill(
            settings,
            store,
            episodes_to_ingest=100,
            tier0_only_share=0.0,
        )
        assert with_share["tier0_calls"] == without_share["tier0_calls"] == 100
        assert with_share["tier1_calls"] < without_share["tier1_calls"]
        assert with_share["indexed_only"] == 92


class TestRewind:
    """Re-reading months the walk has already passed.

    The cursor only moves backwards, so a month behind it is unreachable —
    including one passed under a policy that discarded most of what it saw.
    Widening the window does not help: that extends the far end, while the
    missing episodes sit in months already behind the cursor.
    """

    async def _seed(self, store: MemoryStore, **fields: Any) -> None:
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                **fields,
            }
        )

    async def test_it_clears_the_cursor(self, store: MemoryStore) -> None:
        await self._seed(store, backfill_cursor="2025-08", backfill_complete=True)
        result = await rewind_cursors(store)

        assert result["rewound"] == ["test-show"]
        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        assert doc["backfill_cursor"] is None
        assert doc["backfill_complete"] is False

    async def test_a_rewound_podcast_starts_from_the_top_again(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The cursor is what decides where the next run begins."""
        settings = make_settings(tmp_path)
        await self._seed(store, backfill_cursor="2025-08", backfill_complete=True)
        await rewind_cursors(store)

        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        # None means "start at the month before this one", per _backfill_show.
        assert doc.get("backfill_cursor") is None
        assert floor_month(datetime(2026, 7, 15, tzinfo=UTC), settings.backfill.months) == "2025-07"

    async def test_it_deletes_nothing(self, store: MemoryStore) -> None:
        """Existing episodes keep their status, summaries and digest claims."""
        await self._seed(store, backfill_cursor="2025-08")
        store.seed(
            make_episode(
                guid="already",
                title="Already summarised",
                status=S.PUBLISHED,
                published_at=datetime(2025, 9, 1, tzinfo=UTC),
                origin=BACKFILL_ORIGIN,
                digest_id="archive:test-show:2025-09",
            )
        )
        await rewind_cursors(store)

        episodes = store.docs_of_type("episode")
        assert len(episodes) == 1
        assert episodes[0]["status"] == S.PUBLISHED.value
        assert episodes[0]["digest_id"] == "archive:test-show:2025-09"

    async def test_one_podcast_can_be_rewound_alone(self, store: MemoryStore) -> None:
        await self._seed(store, backfill_cursor="2025-08")
        store.seed(
            {
                "_id": podcast_doc_id("other"),
                "type": "podcast",
                "slug": "other",
                "backfill_cursor": "2025-08",
            }
        )
        result = await rewind_cursors(store, slug="test-show")

        assert result["rewound"] == ["test-show"]
        other = await store.get(podcast_doc_id("other"))
        assert other is not None
        assert other["backfill_cursor"] == "2025-08"

    async def test_a_podcast_that_never_started_is_left_alone(self, store: MemoryStore) -> None:
        """Nothing to rewind, and rewriting it would churn revisions."""
        await self._seed(store)
        result = await rewind_cursors(store)
        assert result["rewound"] == []

    @respx.mock
    async def test_a_second_pass_adds_only_what_was_missing(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Ingestion is create-if-absent, so re-walking cannot duplicate."""
        settings = archive_settings(tmp_path)
        respx.get(FEED_URL).mock(
            return_value=httpx.Response(
                200,
                text=feed_with(
                    [
                        {
                            "guid": "a",
                            "title": "One",
                            "published": datetime(2026, 6, 10, tzinfo=UTC),
                        },
                    ]
                ),
            )
        )
        await self._seed(store)
        first = await ingestor(settings, store).run(now=NOW)
        assert first.episodes_created == 1

        await rewind_cursors(store)
        second = await ingestor(settings, store).run(now=NOW)
        assert second.episodes_created == 0
        assert second.episodes_existing == 1
        assert len(store.docs_of_type("episode")) == 1


class TestWhatPauseActuallyStops:
    """Pause is not "stop starting new rounds"; it is "stop".

    Three distinct effects, and the third is the surprising one: an archive
    episode is only ever advanced inside a backfill run, so pausing freezes the
    ones already ingested rather than letting them drain. That mattered little
    when ingestion only took transcript-bearing episodes; now that it records
    everything, a pause strands the whole backlog.
    """

    async def test_a_scheduled_round_does_not_start(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path)
        await set_control_paused(store, True)
        result = await _runner(settings, store, FakeLLM()).run_backfill_scheduled()
        assert result == {"skipped": "paused"}

    async def test_a_running_round_stops_at_the_next_episode(self, store: MemoryStore) -> None:
        """Checked between episodes, so the one in flight finishes."""
        from podcast_agent.backfill.control import PauseCheck

        check = PauseCheck(store, cache_seconds=0.0)
        assert await check.should_stop() is True  # defaults to paused
        await set_control_paused(store, False)
        assert await check.should_stop() is False

    async def test_already_ingested_archive_episodes_are_frozen_not_queued(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Nothing else in the system will pick them up.

        The routine pipeline filters every one of its queues to routine origin,
        so a paused walk means no job at all is looking at these.
        """
        settings = make_settings(tmp_path)
        for i in range(3):
            store.seed(
                make_episode(
                    guid=f"a{i}",
                    title=f"Archive {i}",
                    status=S.NEW,
                    published_at=datetime(2025, 6, 1, tzinfo=UTC),
                    origin=BACKFILL_ORIGIN,
                )
            )
        await set_control_paused(store, True)
        runner = _runner(settings, store, FakeLLM())

        # The archive walk will not run at all.
        assert (await runner.run_backfill_scheduled())["skipped"] == "paused"
        # And the routine pipeline does not consider them its work.
        assert await runner._queue(S.NEW, limit=10) == []

        still_new = [d for d in store.docs_of_type("episode") if d["status"] == S.NEW.value]
        assert len(still_new) == 3


class TestPodcastSettingsAreReadLive:
    """Changing a podcast's settings must affect history already ingested.

    Ingestion used to copy `backfill_mode` and `asr_enabled` onto each archive
    episode. The copy then outranked the console: turning on local transcription
    did nothing for episodes walked before the change, silently, with nothing on
    the page to say so. Six AI Security Brief episodes sat indexed-only for
    exactly this reason.
    """

    def _processor(self, settings, store: MemoryStore, registry):
        llm = FakeLLM()
        return BackfillProcessor(
            settings,
            store,
            tier0=Tier0Stage(settings, store, llm),
            transcripts=TranscriptStage(
                settings,
                store,
                TranscriptAcquirer(  # type: ignore[arg-type]
                    settings, store, build_client(), UrlGuard(settings.security), None, registry
                ),
            ),
            tier1=Tier1Stage(settings, store, llm),
            registry=registry,
        )

    def _episode(self, guid: str, status: S, **over: Any) -> dict[str, Any]:
        doc = make_episode(
            guid=guid,
            title=guid,
            status=status,
            published_at=datetime(2026, 6, 10, tzinfo=UTC),
            origin=BACKFILL_ORIGIN,
            **over,
        )
        # Shaped as the walk writes it now: no snapshot of podcast settings.
        doc.pop("backfill_mode", None)
        doc.pop("asr_enabled", None)
        return doc

    async def test_archive_mode_changed_after_ingest_is_honoured(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        from podcast_agent.podcasts import PodcastRegistry, set_overrides

        settings = make_settings(tmp_path)
        registry = PodcastRegistry(settings)
        await _seed_podcast_docs(settings, store)
        store.seed(
            self._episode(
                "escalated",
                S.TRIAGED,
                feed_transcripts=[
                    {"url": "https://transcript-host.net/a.txt", "type": "text/plain"}
                ],
                tier0={"relevance_guess": 9, "confidence": 9, "route": "ESCALATE"},
            )
        )
        # The podcast was ingested as "summarise"; the owner now wants index-only.
        await set_overrides(store, "test-show", {"backfill_mode": "tier0_only"})
        await registry.refresh(store)

        await self._processor(settings, store, registry)._dispatch(BackfillProcessStats())

        doc = next(iter(store.docs_of_type("episode")))
        assert doc["status"] == S.DIGEST_DIRECT.value, "still using the ingest-time mode"

    async def _allow_asr_seen(self, settings, store: MemoryStore, registry) -> bool:
        """Run the transcribe stage and report the allow_asr it decided on.

        That decision is the whole point: it is where the podcast's
        transcription toggle is consulted, and where the ingest-time copy used
        to win.
        """
        seen: list[bool] = []

        class Spy:
            async def process(self, episode: Any, *, allow_asr: bool) -> S:
                seen.append(allow_asr)
                return S.TRANSCRIPT_FAILED

        processor = BackfillProcessor(
            settings,
            store,
            tier0=Tier0Stage(settings, store, FakeLLM()),
            transcripts=Spy(),  # type: ignore[arg-type]
            tier1=Tier1Stage(settings, store, FakeLLM()),
            registry=registry,
        )
        await processor._transcribe(BackfillProcessStats())
        assert seen, "the transcribe stage did not run"
        return seen[0]

    async def test_transcription_turned_on_after_ingest_is_honoured(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The case that produced no summaries for six episodes."""
        from podcast_agent.podcasts import PodcastRegistry, set_overrides

        settings = make_settings(tmp_path)
        registry = PodcastRegistry(settings)
        await _seed_podcast_docs(settings, store)
        store.seed(self._episode("needs_asr", S.AWAITING_TRANSCRIPT))

        # Ingested while transcription was off; turned on afterwards.
        await set_overrides(store, "test-show", {"asr_enabled": True})
        await registry.refresh(store)

        assert await self._allow_asr_seen(settings, store, registry) is True

    async def test_transcription_turned_off_after_ingest_is_also_honoured(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """It has to work in both directions, or it is just a different bug."""
        from podcast_agent.podcasts import PodcastRegistry, set_overrides

        settings = make_settings(tmp_path)
        registry = PodcastRegistry(settings)
        await _seed_podcast_docs(settings, store)
        store.seed(self._episode("no_asr_now", S.AWAITING_TRANSCRIPT))

        await set_overrides(store, "test-show", {"asr_enabled": False})
        await registry.refresh(store)

        assert await self._allow_asr_seen(settings, store, registry) is False

    async def test_the_podcast_toggle_is_the_whole_gate(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """There is no global override left to reason about.

        `backfill.require_transcript` used to sit above this and win. It was
        removed: whether a back catalogue is worth hours of CPU differs per
        podcast, so that is where the decision lives.
        """
        from podcast_agent.podcasts import PodcastRegistry, set_overrides

        settings = make_settings(tmp_path)
        registry = PodcastRegistry(settings)
        await _seed_podcast_docs(settings, store)
        store.seed(self._episode("gated", S.AWAITING_TRANSCRIPT))
        await set_overrides(store, "test-show", {"asr_enabled": True})
        await registry.refresh(store)

        assert await self._allow_asr_seen(settings, store, registry) is True

    async def test_the_walk_no_longer_stamps_settings_onto_episodes(self, tmp_path: Path) -> None:
        """A field nothing reads is a trap for whoever reads it next."""
        import inspect

        from podcast_agent.backfill import ingest as ingest_module

        source = inspect.getsource(ingest_module.BackfillIngestor._backfill_show)
        assert '"asr_enabled": podcast.asr_enabled' not in source
        assert '"backfill_mode": podcast.backfill_mode' not in source


#: One episode a month across the whole window, so a walk always has something
#: to find in whichever month it is asked about.
_MONTHLY_ENTRIES = [
    {
        "guid": f"{year}-{month:02d}",
        "title": f"Episode {year}-{month:02d}",
        "published": datetime(year, month, 15, tzinfo=UTC),
        "transcript": f"https://transcript-host.net/{year}-{month:02d}.txt",
    }
    for year, month in [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 9)]
]


class TestTheWindowDoesNotDriftUnderTheWalk:
    """The floor used to be recomputed from `now` on every run.

    The cursor moves backwards, a floor derived from today moves forwards, and
    they can cross. On 2026-07-31 five shows sat at cursor 2025-07 with floor
    2025-07 — meaning "2025-07 is next". At midnight the floor became 2025-08,
    every one of them was declared finished, and 2025-07 was never fetched.
    """

    def test_the_floor_is_measured_from_the_anchor_not_today(self) -> None:
        assert floor_from_anchor("2026-07", 12) == "2025-07"
        # A month later, the same walk still reaches the same month.
        assert floor_from_anchor("2026-07", 12) != floor_month(datetime(2026, 8, 1, tzinfo=UTC), 12)

    def test_widening_the_window_still_deepens_the_floor(self) -> None:
        """Anchoring must not freeze the setting as well as the calendar."""
        assert floor_from_anchor("2026-07", 24) == "2024-07"
        assert floor_from_anchor("2026-07", 36) == "2023-07"

    async def test_a_month_at_the_boundary_is_not_stepped_over(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The exact loss, reproduced: a run on the far side of a rollover."""
        settings = archive_settings(tmp_path)
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "backfill_anchor": "2026-07",
                "backfill_cursor": "2025-07",
                "backfill_complete": False,
            }
        )
        with respx.mock:
            respx.get(FEED_URL).mock(
                return_value=httpx.Response(200, text=feed_with(_MONTHLY_ENTRIES))
            )
            # A calendar month after the anchor — the moment the old floor moved.
            await ingestor(settings, store).run(now=datetime(2026, 8, 1, tzinfo=UTC))

        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        # 2025-07 was processed rather than skipped, and only then is it done.
        assert doc["backfill_cursor"] == "2025-06"
        assert doc["backfill_complete"] is True


class TestCompletionIsRecordedNotJustCounted:
    """`backfill_complete` was written only when a month was processed.

    A show that arrived already past its floor was counted complete in the run
    summary while its document kept `complete: False` — and the console reads
    the document, so it said "in progress" for good.
    """

    async def test_a_show_already_past_its_floor_is_marked_complete(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = archive_settings(tmp_path)
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "backfill_anchor": "2026-07",
                "backfill_cursor": "2025-06",  # already below the 12-month floor
                "backfill_complete": False,
            }
        )
        with respx.mock:
            respx.get(FEED_URL).mock(
                return_value=httpx.Response(200, text=feed_with(_MONTHLY_ENTRIES))
            )
            stats = await ingestor(settings, store).run(now=datetime(2026, 8, 1, tzinfo=UTC))

        assert stats.shows_complete == 1
        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        assert doc["backfill_complete"] is True, "counted complete but never recorded"

    async def test_the_anchor_is_pinned_on_the_first_run(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = archive_settings(tmp_path)
        store.seed({"_id": podcast_doc_id("test-show"), "type": "podcast", "slug": "test-show"})
        with respx.mock:
            respx.get(FEED_URL).mock(
                return_value=httpx.Response(200, text=feed_with(_MONTHLY_ENTRIES))
            )
            await ingestor(settings, store).run(now=datetime(2026, 7, 15, tzinfo=UTC))

        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        assert doc["backfill_anchor"] == "2026-07"


class TestSummarisingIsNotStarvedByTranscription:
    """Yielding mid-stage abandons the run, which is right. Always abandoning
    the *same* stage is not.

    With transcribe ahead of summarize, a transcript queue 181 deep at minutes
    an episode never drained inside a run, so summarize never got a turn — and
    63 episodes sat fully transcribed, one cheap call short of done.
    """

    def test_summarise_runs_before_transcribe(self) -> None:
        source = (Path(__file__).parent.parent / "podcast_agent/backfill/process.py").read_text()
        run = source[source.index("async def run(") : source.index("async def _stop_requested")]
        assert run.index("self._summarize(stats)") < run.index("self._transcribe(stats)")

    async def test_a_transcribed_episode_is_summarised_even_with_a_deep_queue(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The behaviour the ordering exists for, driven end to end."""
        settings = archive_settings(tmp_path)
        store.seed(
            make_episode(
                guid="ready",
                status=S.TRANSCRIBED,
                origin=BACKFILL_ORIGIN,
                archive_month="2026-01",
            ),
            *[
                make_episode(
                    guid=f"waiting-{i}",
                    status=S.AWAITING_TRANSCRIPT,
                    origin=BACKFILL_ORIGIN,
                    archive_month="2026-01",
                )
                for i in range(5)
            ],
        )
        ready_id = next(e["_id"] for e in store.docs_of_type("episode") if e["guid"] == "ready")
        await save_transcript(store, ready_id, TRANSCRIPT)

        # Yields as soon as anything has been transcribed — which is what a
        # busy machine does, and what used to strand the summary.
        calls = {"n": 0}

        async def should_stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        processor = _processor(settings, store, FakeLLM())
        stats = await processor.run(should_stop=should_stop)

        assert stats.summarized + stats.scored_low == 1, "the ready episode was never scored"


class TestRepairingAnInterruptedWalk:
    """The migration for walks that were mid-flight when the floor drifted.

    Five shows here lost 2025-07 to it. The repair anchors each unfinished walk
    so its floor lands exactly on the month the cursor reached: finish the month
    you are on, then stop. That needs no knowledge of when the walk started, and
    re-reads nothing behind the cursor.
    """

    async def _run(self, store: MemoryStore, months: int = 12) -> dict[str, int]:
        from podcast_agent.migrate import anchor_backfill_walks

        return await anchor_backfill_walks(store, months)

    def _show(self, slug: str, **fields: Any) -> dict[str, Any]:
        return {"_id": podcast_doc_id(slug), "type": "podcast", "slug": slug, **fields}

    async def test_an_unfinished_walk_resumes_at_its_cursor(self, store: MemoryStore) -> None:
        store.seed(self._show("a", backfill_cursor="2025-07", backfill_complete=False))
        assert (await self._run(store))["anchored"] == 1

        doc = await store.get(podcast_doc_id("a"))
        assert doc is not None
        assert floor_from_anchor(doc["backfill_anchor"], 12) == "2025-07"

    async def test_the_repaired_show_then_walks_that_month(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """End to end: the month that was lost is actually fetched."""
        settings = archive_settings(tmp_path)
        store.seed(self._show("test-show", backfill_cursor="2025-07", backfill_complete=False))
        await self._run(store)

        with respx.mock:
            respx.get(FEED_URL).mock(
                return_value=httpx.Response(200, text=feed_with(_MONTHLY_ENTRIES))
            )
            await ingestor(settings, store).run(now=datetime(2026, 8, 1, tzinfo=UTC))

        months = {e["archive_month"] for e in store.docs_of_type("episode")}
        assert "2025-07" in months

    async def test_a_finished_walk_is_left_alone(self, store: MemoryStore) -> None:
        """Anchoring it to its cursor would make it re-walk that month."""
        store.seed(self._show("a", backfill_cursor="2025-06", backfill_complete=True))
        assert (await self._run(store))["anchored"] == 0
        doc = await store.get(podcast_doc_id("a"))
        assert doc is not None
        assert "backfill_anchor" not in doc

    async def test_a_show_that_never_started_is_left_alone(self, store: MemoryStore) -> None:
        store.seed(self._show("a"))
        assert (await self._run(store))["anchored"] == 0

    async def test_it_does_not_re_anchor_on_a_second_boot(self, store: MemoryStore) -> None:
        """Idempotent: a migration that must run exactly once runs zero times."""
        store.seed(self._show("a", backfill_cursor="2025-07", backfill_complete=False))
        assert (await self._run(store))["anchored"] == 1
        assert (await self._run(store))["anchored"] == 0

    async def test_it_honours_a_wider_window(self, store: MemoryStore) -> None:
        store.seed(self._show("a", backfill_cursor="2024-07", backfill_complete=False))
        await self._run(store, months=24)
        doc = await store.get(podcast_doc_id("a"))
        assert doc is not None
        assert floor_from_anchor(doc["backfill_anchor"], 24) == "2024-07"


class TestTranscriptionHasItsOwnBudget:
    """Acquisition is capped separately from summarisation.

    `_transcribe` sized its batch with `max_summaries_per_run`, so raising the
    summary budget — a cheap call — silently raised how many episodes the walk
    would transcribe, which on a podcast with local transcription on is hours
    of CPU rather than seconds of model time. The two costs differ by orders of
    magnitude and now have separate valves.
    """

    def _archive_episodes(self, store: MemoryStore, count: int) -> None:
        store.seed(
            *[
                make_episode(
                    guid=f"waiting-{i}",
                    status=S.AWAITING_TRANSCRIPT,
                    origin=BACKFILL_ORIGIN,
                    archive_month="2026-01",
                )
                for i in range(count)
            ]
        )

    async def test_the_batch_honours_the_transcript_cap(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = archive_settings(
            tmp_path,
            backfill={
                "months": 12,
                "digest_threshold": 7,
                "months_per_run": 1,
                "max_episodes_per_run": 200,
                "max_summaries_per_run": 20,
                "max_transcripts_per_run": 2,
            },
        )
        self._archive_episodes(store, 5)

        stats = await _processor(settings, store, FakeLLM()).run()

        # No transcript to be had for any of them, so every episode the stage
        # actually reached is counted — which is the batch size.
        assert stats.no_transcript == 2

    async def test_the_summary_budget_no_longer_moves_it(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The coupling that made this a bug: same transcript cap, summary
        budget raised, and the transcription batch must not follow it."""
        settings = archive_settings(
            tmp_path,
            backfill={
                "months": 12,
                "digest_threshold": 7,
                "months_per_run": 1,
                "max_episodes_per_run": 200,
                "max_summaries_per_run": 500,
                "max_transcripts_per_run": 2,
            },
        )
        self._archive_episodes(store, 5)

        stats = await _processor(settings, store, FakeLLM()).run()

        assert stats.no_transcript == 2
