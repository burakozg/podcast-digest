"""Processing archive episodes (roadmap A1).

Same stages as the routine pipeline, three differences that all follow from the
economics of an archive:

* **Transcription is opted into per podcast.** The podcast's own "transcribe
  locally" toggle is the only gate, read live, so a console change reaches
  episodes already ingested.
* **A stricter threshold.** Old material must earn its summary.
* **``tier0_only`` shows are never summarised.** A two-year-old daily news
  round-up is worth a line in an index, not four minutes of local LLM.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..db import Doc, Store, typed_sort
from ..episodes import mark_error, transition
from ..llm.base import LLMUnavailable
from ..logging_setup import get_logger
from ..models import Route
from ..podcasts import PodcastRegistry
from ..state import BACKFILL_ORIGIN, EpisodeStatus
from ..summarize.tier1 import Tier1Stage
from ..transcripts.stage import TranscriptStage
from ..triage.tier0 import Tier0Stage

log = get_logger(__name__)


@dataclass(slots=True)
class BackfillProcessStats:
    triaged: int = 0
    dropped: int = 0
    listed: int = 0
    #: Indexed rather than summarised because no transcript was published.
    listed_no_transcript: int = 0
    summarized: int = 0
    scored_low: int = 0
    transcripts_ok: int = 0
    no_transcript: int = 0
    errors: int = 0
    #: Stopped because backfill was paused, rather than because work ran out.
    stopped_early: bool = False
    deferred: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "triaged": self.triaged,
            "dropped": self.dropped,
            "listed": self.listed,
            "listed_no_transcript": self.listed_no_transcript,
            "summarized": self.summarized,
            "scored_low": self.scored_low,
            "transcripts_ok": self.transcripts_ok,
            "no_transcript": self.no_transcript,
            "errors": self.errors,
            "stopped_early": self.stopped_early,
            "deferred": self.deferred,
        }


class BackfillProcessor:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        *,
        tier0: Tier0Stage,
        transcripts: TranscriptStage,
        tier1: Tier1Stage,
        registry: PodcastRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._tier0 = tier0
        self._transcripts = transcripts
        self._tier1 = tier1
        # The podcast's archive mode and transcription toggle are read from here,
        # live, rather than from the episode document. Ingestion used to snapshot
        # them onto each episode, so changing either in the console had no effect
        # on history already taken — silently, with nothing to say so.
        self._registry = registry or PodcastRegistry(settings)
        self._should_stop: Callable[[], Awaitable[bool]] | None = None

    async def run(
        self, *, should_stop: Callable[[], Awaitable[bool]] | None = None
    ) -> BackfillProcessStats:
        stats = BackfillProcessStats()
        self._should_stop = should_stop
        # Cheapest first, most open-ended last — deliberately *not* pipeline
        # order.
        #
        # Yielding mid-stage abandons the rest of the run, which is right: the
        # signal is that routine intake needs the machine. But with transcribe
        # ahead of summarize, the abandoned stage was always the same one. A
        # transcript queue 181 deep at minutes an episode never drains inside a
        # run, so summarize never got a turn, and 63 episodes sat fully
        # transcribed and one cheap call short of done — indefinitely.
        #
        # Each stage still feeds the next; the work simply lands on the
        # following run, which for a job that fires every twenty minutes costs
        # nothing. It already worked this way whenever a stage hit its cap.
        await self._triage(stats)
        if not stats.stopped_early:
            await self._dispatch(stats)
        if not stats.stopped_early:
            await self._summarize(stats)
        if not stats.stopped_early:
            await self._transcribe(stats)
        log.info("backfill.process_complete", **stats.as_dict())
        return stats

    async def _stop_requested(self, stats: BackfillProcessStats, stage: str) -> bool:
        """Check between episodes so a pause costs at most one in-flight item."""
        if self._should_stop is None or not await self._should_stop():
            return False
        stats.stopped_early = True
        log.info("backfill.paused", stage=stage)
        return True

    async def _queue(self, status: EpisodeStatus, limit: int) -> list[Doc]:
        return await self._store.find(
            {"type": "episode", "origin": BACKFILL_ORIGIN, "status": status.value},
            sort=typed_sort("published_at", "desc"),
            limit=limit,
        )

    async def _triage(self, stats: BackfillProcessStats) -> None:
        episodes = await self._queue(
            EpisodeStatus.NEW, self._settings.backfill.max_episodes_per_run
        )
        for episode in episodes:
            if await self._stop_requested(stats, "triage"):
                return
            try:
                await self._tier0.triage(episode)
                stats.triaged += 1
            except LLMUnavailable as exc:
                stats.deferred.append("triage")
                log.error("backfill.triage_deferred", error=str(exc))
                return
            except Exception as exc:
                stats.errors += 1
                log.warning("backfill.triage_failed", episode_id=episode["_id"], error=str(exc))
                await mark_error(self._store, episode["_id"], "backfill_tier0", exc)

    async def _dispatch(self, stats: BackfillProcessStats) -> None:
        for episode in await self._queue(EpisodeStatus.TRIAGED, 1000):
            route = Route((episode.get("tier0") or {}).get("route") or Route.DROP.value)
            podcast = self._registry.podcast_by_slug(episode["podcast_slug"])
            mode = podcast.backfill_mode if podcast else "full"

            # Why the archive treated this episode differently from what triage
            # decided. Recorded on the document, because otherwise `tier0.route`
            # says ESCALATE for an episode that was never escalated: the reader
            # then sees an episode published with no summary and no explanation,
            # which reads as work still pending rather than a decision taken.
            downgraded: str | None = None

            # A news-cadence podcast never earns a summary from the archive, so
            # an ESCALATE verdict is downgraded to a one-line index entry.
            if route is Route.ESCALATE and mode == "tier0_only":
                route = Route.DIGEST_DIRECT
                downgraded = "news cadence: archive entries are indexed, not summarised"

            # Same downgrade when there is no transcript to be had: the podcast
            # publishes none and is not set to transcribe locally. Escalating it
            # would send it to AWAITING_TRANSCRIPT, fail acquisition, and then
            # spend a Tier-1 call summarising the description alone — the most
            # expensive stage, for the least material. It is indexed instead, at
            # triage cost only.
            if (
                route is Route.ESCALATE
                and not (podcast and podcast.asr_enabled)
                and not (episode.get("feed_transcripts") or [])
            ):
                route = Route.DIGEST_DIRECT
                downgraded = "no transcript published, and local transcription is off"
                stats.listed_no_transcript += 1

            target = {
                Route.DROP: EpisodeStatus.DROPPED,
                Route.DIGEST_DIRECT: EpisodeStatus.DIGEST_DIRECT,
                Route.ESCALATE: EpisodeStatus.AWAITING_TRANSCRIPT,
            }[route]

            def _record(doc: Doc, reason: str | None = downgraded) -> None:
                if reason:
                    doc["indexed_only"] = reason

            try:
                await transition(self._store, episode["_id"], target, mutate=_record)
                if target is EpisodeStatus.DROPPED:
                    stats.dropped += 1
                elif target is EpisodeStatus.DIGEST_DIRECT:
                    stats.listed += 1
            except Exception as exc:
                stats.errors += 1
                await mark_error(self._store, episode["_id"], "backfill_dispatch", exc)

    async def _transcribe(self, stats: BackfillProcessStats) -> None:
        episodes = await self._queue(
            EpisodeStatus.AWAITING_TRANSCRIPT, self._settings.backfill.max_transcripts_per_run
        )
        for episode in episodes:
            if await self._stop_requested(stats, "transcribe"):
                return
            try:
                # The podcast's own toggle, read live. An archive walk spans
                # months of audio, so this is the switch that decides whether a
                # podcast's back catalogue is worth transcribing — made per
                # podcast, because that is the only place the answer differs.
                podcast = self._registry.podcast_by_slug(episode["podcast_slug"])
                allow_asr = bool(podcast and podcast.asr_enabled)
                outcome = await self._transcripts.process(episode, allow_asr=allow_asr)
                if outcome is EpisodeStatus.TRANSCRIBED:
                    stats.transcripts_ok += 1
                elif outcome is EpisodeStatus.TRANSCRIPT_FAILED:
                    # Left at TRANSCRIPT_FAILED on purpose. The routine pipeline
                    # falls back to a description-only summary here, but for a
                    # years-old episode that is the least trustworthy artefact
                    # this system can produce, and the archive is optional.
                    stats.no_transcript += 1
            except Exception as exc:
                stats.errors += 1
                log.warning("backfill.transcript_failed", episode_id=episode["_id"], error=str(exc))
                await mark_error(self._store, episode["_id"], "backfill_transcript", exc)

    async def _summarize(self, stats: BackfillProcessStats) -> None:
        budget = self._settings.backfill.max_summaries_per_run
        episodes = await self._queue(EpisodeStatus.TRANSCRIBED, budget)
        for episode in episodes:
            if await self._stop_requested(stats, "summarize"):
                return
            try:
                outcome = await self._tier1.summarize(
                    episode, threshold=self._settings.backfill.digest_threshold
                )
                if outcome is EpisodeStatus.READY_FOR_DIGEST:
                    stats.summarized += 1
                else:
                    stats.scored_low += 1
            except LLMUnavailable as exc:
                stats.deferred.append("summarize")
                log.error("backfill.summarize_deferred", error=str(exc))
                return
            except Exception as exc:
                stats.errors += 1
                log.warning("backfill.summarize_failed", episode_id=episode["_id"], error=str(exc))
                await mark_error(self._store, episode["_id"], "backfill_tier1", exc)
