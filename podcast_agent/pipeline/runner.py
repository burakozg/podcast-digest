"""Pipeline orchestration (§4, §10.3).

Every stage is driven purely by document status, so the scheduler can die at any
point and the next run resumes from the documents themselves. Per-episode
exceptions are caught and recorded, so one poison-pill episode cannot block the
queue or crash a run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..backfill import ROUTINE_ONLY
from ..backfill.control import PauseCheck, is_paused
from ..backfill.estimate import estimate_backfill
from ..backfill.ingest import BackfillIngestor
from ..backfill.process import BackfillProcessor
from ..config import Settings
from ..db import Doc, Store, typed_sort
from ..digest.archive import ArchiveDigestGenerator
from ..digest.generate import DigestGenerator, DigestResult
from ..episodes import mark_error, transition
from ..ingest.feeds import Ingestor
from ..joblock import LeaseLost, held
from ..llm.base import LLMUnavailable
from ..logging_setup import bind_run, clear_run_context, get_logger
from ..podcasts import PodcastRegistry
from ..state import ACTIVE_STATUSES, EpisodeStatus
from ..summarize.tier1 import Tier1Stage
from ..transcripts.asr import ASRUnavailable
from ..transcripts.stage import TranscriptStage
from ..triage.tier0 import Tier0Stage
from ..utils import iso_now, new_run_id, run_doc_id

log = get_logger(__name__)

#: Types a run summary may carry into CouchDB. A stray object would fail the
#: write and lose the whole record for the sake of one field.
_JSONABLE = (str, int, float, bool, list, dict, type(None))


#: Statuses a fresh episode passes through before it is settled. While any
#: non-archive episode sits in one of these, routine intake is not finished.
_PENDING_STATUSES: tuple[str, ...] = tuple(sorted(s.value for s in ACTIVE_STATUSES))


async def pending_routine_episodes(store: Store, *, limit: int = 200) -> list[Doc]:
    """Recent-intake episodes still owed pipeline work, newest first.

    Deliberately excludes ERROR: an episode that failed hard needs a person, and
    letting one poisoned document block the archive walk forever would be a
    worse failure than the one it is reporting. Everything else here drains on
    its own — a transcript that cannot be acquired ends at TRANSCRIPT_FAILED and
    is then summarised from its description.
    """
    return await store.find(
        {
            "type": "episode",
            "status": {"$in": list(_PENDING_STATUSES)},
            **ROUTINE_ONLY,
        },
        fields=["_id", "podcast_slug", "title", "status", "published_at"],
        sort=typed_sort("published_at", "desc"),
        limit=limit,
    )


class RecentWorkCheck:
    """Asks whether recent-intake episodes are still owed work, with a cache.

    Consulted between archive episodes so a long walk yields as soon as a fresh
    intake lands, rather than finishing hours of history first. Cached because
    it runs in a tight loop; a stale answer for a few seconds only delays the
    handover by a few seconds.
    """

    def __init__(self, store: Store, *, cache_seconds: float = 10.0) -> None:
        self._store = store
        self._cache_seconds = cache_seconds
        self._last_checked: float = 0.0
        self._last_value = False

    async def should_stop(self) -> bool:
        now = time.monotonic()
        if self._last_checked and now - self._last_checked < self._cache_seconds:
            return self._last_value
        try:
            self._last_value = bool(await pending_routine_episodes(self._store, limit=1))
        except Exception as exc:
            # A storage blip must not silently halt a long run.
            log.warning("backfill.recent_work_check_failed", error=str(exc))
            self._last_value = False
        self._last_checked = now
        return self._last_value


class JobBusy(Exception):
    """This job is already running.

    Enforces the same "never overlap" guarantee as APScheduler's
    ``max_instances=1`` (§11), but across manual API triggers too.
    """


@dataclass(slots=True)
class PipelineStats:
    triaged: int = 0
    dispatched: int = 0
    dropped: int = 0
    escalated: int = 0
    digest_direct: int = 0
    transcripts_ok: int = 0
    transcripts_failed: int = 0
    transcripts_retrying: int = 0
    summarized: int = 0
    scored_low: int = 0
    errors: int = 0
    #: Stages cut short because a tier's whole LLM chain or the ASR backend was
    #: unavailable. Work stays queued (§10.6).
    stages_deferred: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "triaged": self.triaged,
            "dispatched": self.dispatched,
            "dropped": self.dropped,
            "escalated": self.escalated,
            "digest_direct": self.digest_direct,
            "transcripts_ok": self.transcripts_ok,
            "transcripts_failed": self.transcripts_failed,
            "transcripts_retrying": self.transcripts_retrying,
            "summarized": self.summarized,
            "scored_low": self.scored_low,
            "errors": self.errors,
            "stages_deferred": self.stages_deferred,
        }


class PipelineRunner:
    """Coordinates ingestion, the processing stages and digest generation."""

    def __init__(
        self,
        settings: Settings,
        store: Store,
        *,
        ingestor: Ingestor,
        tier0: Tier0Stage,
        transcripts: TranscriptStage,
        tier1: Tier1Stage,
        digest: DigestGenerator,
        registry: PodcastRegistry | None = None,
        backfill_ingest: BackfillIngestor | None = None,
        backfill_process: BackfillProcessor | None = None,
        archive: ArchiveDigestGenerator | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._ingestor = ingestor
        self._tier0 = tier0
        self._transcripts = transcripts
        self._tier1 = tier1
        self._digest = digest
        self._registry = registry or PodcastRegistry(settings)
        self._backfill_ingest = backfill_ingest
        self._backfill_process = backfill_process
        self._archive = archive
        self.last_runs: dict[str, dict[str, Any]] = {}
        #: Episodes being summarised on demand, so the same one cannot be
        #: triggered twice concurrently from an impatient click.
        self._episodes_in_flight: set[str] = set()
        #: Strong references to in-flight run-record writes.
        self._run_records: set[asyncio.Task[None]] = set()
        self._locks: dict[str, asyncio.Lock] = {
            job: asyncio.Lock() for job in ("ingest", "pipeline", "digest", "rescore", "backfill")
        }

    def is_running(self, job: str) -> bool:
        """Whether *this process* is running the job.

        Deliberately local and synchronous — it answers "is the button I just
        pressed already busy", which is a question about this server. A job held
        by another process is reported by the lease document instead; see
        :func:`podcast_agent.joblock.current_holders`.
        """
        lock = self._locks.get(job)
        return bool(lock and lock.locked())

    @asynccontextmanager
    async def _exclusive(self, job: str) -> AsyncIterator[None]:
        """Refuse to overlap this job, in this process and across the database.

        Two guards, and both earn their place. The in-process lock is instant
        and covers the case that actually happens hourly — a scheduled fire and
        an impatient click. The lease covers what the lock cannot see at all: a
        second instance, a CLI run, a container replica, any of which would
        otherwise run a second backfill against the same episodes and pay for
        every one of them twice.
        """
        lock = self._locks[job]
        if lock.locked():
            raise JobBusy(f"{job} is already running")
        async with lock:
            try:
                async with held(self._store, job):
                    yield
            except LeaseLost as exc:
                raise JobBusy(str(exc)) from exc

    # --- jobs ---------------------------------------------------------------

    async def run_ingest(self) -> dict[str, Any]:
        async with self._exclusive("ingest"):
            return await self._run_ingest()

    async def _run_ingest(self) -> dict[str, Any]:
        run_id = new_run_id()
        bind_run(run_id, job="ingest")
        started = time.perf_counter()
        try:
            await self._registry.refresh(self._store)
            await self._ingestor.seed_podcast_docs()
            stats = await self._ingestor.run()
            summary = {
                **stats.as_dict(),
                "wall_ms": int((time.perf_counter() - started) * 1000),
            }
            log.info("run.ingest_summary", **summary)
            self._remember("ingest", summary)
            return summary
        finally:
            clear_run_context()

    async def run_pipeline(self) -> dict[str, Any]:
        async with self._exclusive("pipeline"):
            return await self._run_pipeline()

    async def _run_pipeline(self) -> dict[str, Any]:
        run_id = new_run_id()
        bind_run(run_id, job="pipeline")
        started = time.perf_counter()
        stats = PipelineStats()
        try:
            # Pick up console changes (ASR toggles, disabled shows) each run.
            await self._registry.refresh(self._store)
            await self._stage_triage(stats)
            await self._stage_dispatch(stats)
            await self._stage_transcripts(stats)
            await self._stage_summarize(stats)
            summary = {
                **stats.as_dict(),
                "wall_ms": int((time.perf_counter() - started) * 1000),
            }
            # Single-line per-run summary (§10.1).
            log.info("run.pipeline_summary", **summary)
            self._remember("pipeline", summary)
            return summary
        finally:
            clear_run_context()

    async def run_digest(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        dry_run: bool = False,
    ) -> DigestResult:
        async with self._exclusive("digest"):
            return await self._run_digest(since=since, until=until, dry_run=dry_run)

    async def _run_digest(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        dry_run: bool = False,
    ) -> DigestResult:
        run_id = new_run_id()
        bind_run(run_id, job="digest")
        started = time.perf_counter()
        try:
            result = await self._digest.generate(since=since, until=until, dry_run=dry_run)
            summary = {
                **result.as_dict(),
                "wall_ms": int((time.perf_counter() - started) * 1000),
            }
            log.info("run.digest_summary", **summary)
            if not dry_run:
                self._remember("digest", summary)
            return result
        finally:
            clear_run_context()

    async def run_backfill(
        self, *, dry_run: bool = True, confirm: bool = False, force: bool = False
    ) -> dict[str, Any]:
        """Walk the archive backwards one month-window per show (roadmap A1).

        Defaults to a dry run. Actually spending compute requires ``confirm``,
        because the roadmap's guardrail is that nobody starts a multi-hour
        archive job by fat-fingering a URL.

        History waits for the present: while any recent-intake episode is still
        owed work — including one sitting in a queue for local transcription —
        the walk does not start, and a walk already running yields at its next
        episode boundary. ``force`` overrides that, because a machine's owner
        should not be locked out of their own archive by one stuck episode.
        Dry runs always proceed: they write nothing and spend nothing.
        """
        if not dry_run and not confirm:
            raise ValueError(
                "refusing to run backfill without confirm=true; "
                "run with dry_run=true first and read the estimate"
            )
        if self._backfill_ingest is None or self._backfill_process is None or self._archive is None:
            raise RuntimeError(
                "backfill is not configured on this runner (build_app wires it; "
                "a hand-built runner must pass backfill_ingest/backfill_process/archive)"
            )
        if not dry_run and not force:
            waiting = await pending_routine_episodes(self._store)
            if waiting:
                log.info("backfill.deferred_to_recent_work", pending=len(waiting))
                return {
                    "skipped": "recent episodes still in the pipeline",
                    "pending": len(waiting),
                    "waiting_on": [
                        {
                            "episode_id": doc.get("_id"),
                            "podcast_slug": doc.get("podcast_slug"),
                            "title": doc.get("title"),
                            "status": doc.get("status"),
                        }
                        for doc in waiting[:10]
                    ],
                }
        async with self._exclusive("backfill"):
            return await self._run_backfill(dry_run=dry_run, force=force)

    async def run_backfill_scheduled(self) -> dict[str, Any]:
        """Cron entry point: does nothing while paused, or while intake is behind.

        Manual runs are an explicit act and proceed regardless; the paused flag
        exists to stop the *unattended* walk, which is the one that can quietly
        consume a machine overnight.
        """
        if await is_paused(self._store):
            log.debug("backfill.scheduled_skip_paused")
            return {"skipped": "paused"}
        waiting = await pending_routine_episodes(self._store)
        if waiting:
            log.info(
                "backfill.scheduled_skip_recent_work",
                pending=len(waiting),
                oldest_status=waiting[-1].get("status"),
            )
            return {"skipped": "recent episodes still in the pipeline", "pending": len(waiting)}
        return await self.run_backfill(dry_run=False, confirm=True)

    async def _run_backfill(self, *, dry_run: bool, force: bool = False) -> dict[str, Any]:
        ingestor = self._backfill_ingest
        processor = self._backfill_process
        archive = self._archive
        assert ingestor is not None and processor is not None and archive is not None

        run_id = new_run_id()
        bind_run(run_id, job="backfill")
        started = time.perf_counter()
        try:
            await self._registry.refresh(self._store)
            # A pause takes effect at the next show/episode boundary, so the
            # in-flight item finishes instead of being thrown away. The same
            # boundary is where the walk hands back to a fresh intake.
            pause = PauseCheck(self._store)
            recent = RecentWorkCheck(self._store)

            async def should_stop() -> bool:
                if await pause.should_stop():
                    return True
                return not force and await recent.should_stop()

            ingest = await ingestor.run(
                dry_run=dry_run, should_stop=None if dry_run else should_stop
            )
            eligible = ingestor.eligible_podcasts()
            tier0_only = [p for p in eligible if p.backfill_mode == "tier0_only"]
            share = len(tier0_only) / len(eligible) if eligible else 0.0
            created = ingest.episodes_created or 1
            estimate = await estimate_backfill(
                self._settings,
                self._store,
                episodes_to_ingest=ingest.episodes_created,
                tier0_only_share=share,
                without_transcript_share=min(1.0, ingest.without_transcript / created),
            )

            summary: dict[str, Any] = {
                "dry_run": dry_run,
                "ingest": ingest.as_dict(),
                "estimate": estimate,
            }
            if not dry_run:
                process = await processor.run(should_stop=should_stop)
                summary["process"] = process.as_dict()
                # Archive files are still written for any month that completed,
                # so a paused run leaves finished months on disk rather than
                # holding them hostage until the walk resumes.
                summary["archive"] = (await archive.generate()).as_dict()
                stopped_early = ingest.stopped_early or process.stopped_early
                summary["paused_mid_run"] = stopped_early
                # Two different reasons to stop early, and the console should
                # not report "paused" for a walk that simply gave way to a
                # fresh episode.
                summary["stopped_early_reason"] = (
                    ("paused" if await is_paused(self._store) else "recent episodes arrived")
                    if stopped_early
                    else None
                )
            summary["wall_ms"] = int((time.perf_counter() - started) * 1000)

            log.info("run.backfill_summary", **summary)
            if not dry_run:
                self._remember("backfill", summary)
            return summary
        finally:
            clear_run_context()

    async def run_rescore(self, *, limit: int = 50, force: bool = False) -> dict[str, Any]:
        """Re-score episodes against the current interest profile (C2)."""
        async with self._exclusive("rescore"):
            return await self._run_rescore(limit=limit, force=force)

    async def _run_rescore(self, *, limit: int, force: bool) -> dict[str, Any]:
        run_id = new_run_id()
        bind_run(run_id, job="rescore")
        started = time.perf_counter()
        try:
            candidates = await self.stale_episodes(limit=limit, force=force)
            rescored = promoted = demoted = unchanged = errors = 0
            for episode in candidates:
                before = episode.get("status")
                try:
                    after = await self._tier1.rescore(episode)
                except LLMUnavailable as exc:
                    log.error("rescore.deferred", error=str(exc), remaining=len(candidates))
                    break
                except Exception as exc:
                    errors += 1
                    log.warning(
                        "rescore.episode_failed",
                        episode_id=episode["_id"],
                        error=str(exc),
                        exc_info=True,
                    )
                    await mark_error(self._store, episode["_id"], "rescore", exc)
                    continue
                rescored += 1
                if after.value == before:
                    unchanged += 1
                elif after is EpisodeStatus.READY_FOR_DIGEST:
                    promoted += 1
                else:
                    demoted += 1

            summary = {
                "candidates": len(candidates),
                "rescored": rescored,
                "promoted": promoted,
                "demoted": demoted,
                "unchanged": unchanged,
                "errors": errors,
                "profile_version": self._settings.interest_profile_version(),
                "wall_ms": int((time.perf_counter() - started) * 1000),
            }
            log.info("run.rescore_summary", **summary)
            self._remember("rescore", summary)
            return summary
        finally:
            clear_run_context()

    async def stale_episodes(self, *, limit: int = 50, force: bool = False) -> list[Doc]:
        """Scored, unpublished episodes whose score predates the current profile.

        Filtering happens in Python rather than Mango so that episodes scored
        before profile versioning existed (no ``profile_version`` at all) are
        also treated as stale.
        """
        current = self._settings.interest_profile_version()
        docs = await self._store.find(
            {
                "type": "episode",
                "status": {
                    "$in": [
                        EpisodeStatus.READY_FOR_DIGEST.value,
                        EpisodeStatus.SCORED_LOW.value,
                    ]
                },
                **ROUTINE_ONLY,
            },
            sort=typed_sort("published_at", "desc"),
            limit=1000,
        )
        stale = [
            d
            for d in docs
            if d.get("tier1") and (force or (d["tier1"] or {}).get("profile_version") != current)
        ]
        return stale[:limit]

    async def summarize_episode(
        self, episode_id: str, *, allow_asr: bool | None = None
    ) -> dict[str, Any]:
        """Acquire a transcript and summarise one episode, now.

        The owner override for "why has this not been summarised?" and for
        "do it properly this time". ``allow_asr=False`` restricts acquisition to
        a published transcript, which is the difference between seconds and
        potentially an hour of CPU.

        Always reaches a verdict: if no transcript can be had, the episode is
        summarised from its description and labelled ``description_only`` rather
        than being left queued with nothing to show.
        """
        if episode_id in self._episodes_in_flight:
            raise JobBusy(f"{episode_id} is already being summarised")

        episode = await self._store.get(episode_id)
        if episode is None:
            raise KeyError(episode_id)

        if allow_asr is None:
            # Default to whatever the show is configured for; an explicit
            # argument from the console still wins.
            await self._registry.refresh(self._store)
            allow_asr = self._registry.allows_asr(episode["podcast_slug"])

        current = EpisodeStatus(episode["status"])
        # Where this episode was already written, if anywhere. Held because the
        # episode must be put back afterwards: it is listed in a file, and a
        # summary added later does not make it un-listed.
        claim = episode.get("digest_id")
        already_added = bool(episode.get("summary_after_listing"))
        if (
            current is EpisodeStatus.PUBLISHED
            and (episode.get("tier1") or {}).get("summary_md")
            and not already_added
        ):
            # Same reasoning as re-scoring: the digest file on disk already
            # states a verdict, and rewriting only the database would make the
            # two disagree with no way to tell which is current.
            #
            # Narrowed to episodes that actually carry a summary. One that was
            # listed as an index entry — the grey zone, or an archive episode
            # with no transcript to be had — has nothing in any file to
            # contradict: the file said it was not summarised, which stays true
            # of the file. Refusing those left an episode the owner wants to
            # read permanently unreachable, which is a worse outcome than a
            # month-old archive listing being less complete than the database.
            raise ValueError(
                f"{episode_id} is already published in a digest; re-summarising "
                "would leave the written digest disagreeing with the database"
            )

        self._episodes_in_flight.add(episode_id)
        started = time.perf_counter()
        bind_run(new_run_id(), job="summarize_episode", episode_id=episode_id)
        try:
            if current is not EpisodeStatus.TRANSCRIBED:
                await transition(self._store, episode_id, EpisodeStatus.AWAITING_TRANSCRIPT)
                fresh = await self._store.get(episode_id) or episode
                outcome = await self._transcripts.process(fresh, allow_asr=allow_asr)
                if outcome is not EpisodeStatus.TRANSCRIBED:
                    # Retry budget may remain, but a manual request wants an
                    # answer now — take the honest description-only path.
                    reread = await self._store.get(episode_id) or fresh
                    if reread["status"] != EpisodeStatus.TRANSCRIPT_FAILED.value:
                        await transition(self._store, episode_id, EpisodeStatus.TRANSCRIPT_FAILED)

            final_doc = await self._store.get(episode_id)
            assert final_doc is not None
            status = await self._tier1.summarize(final_doc)

            if current is EpisodeStatus.PUBLISHED and claim:
                # Back where it was. Summarising does not un-publish an episode:
                # it is listed in a file that is never rewritten, and its claim
                # is still held, so no digest will take it. Leaving it at
                # READY_FOR_DIGEST would assert it was waiting for one — a queue
                # entry that can never drain, and the reason "Ready for the next
                # digest" started counting episodes that were already finished.
                def _relist(doc: Doc) -> None:
                    doc["summary_after_listing"] = True

                await transition(self._store, episode_id, EpisodeStatus.PUBLISHED, mutate=_relist)
                status = EpisodeStatus.PUBLISHED

            result_doc = await self._store.get(episode_id) or final_doc
            tier1 = result_doc.get("tier1") or {}
            summary = {
                "episode_id": episode_id,
                "status": status.value,
                "allow_asr": allow_asr,
                "summary_basis": tier1.get("summary_basis"),
                "relevance_score": tier1.get("relevance_score"),
                "transcript_source": result_doc.get("transcript_source"),
                "wall_ms": int((time.perf_counter() - started) * 1000),
            }
            log.info("episode.summarized_on_demand", **summary)
            return summary
        finally:
            self._episodes_in_flight.discard(episode_id)
            clear_run_context()

    def is_episode_in_flight(self, episode_id: str) -> bool:
        return episode_id in self._episodes_in_flight

    def _remember(self, job: str, summary: dict[str, Any]) -> None:
        """Record a completed job, in memory and on disk.

        In memory alone was not enough: "when did this last run?" is exactly the
        question asked after a restart, and a restart is what emptied the
        answer. The document is the durable record; the dict stays because every
        existing reader of `last_runs` expects it and it costs nothing.
        """
        entry = {"at": iso_now(), **summary}
        self.last_runs[job] = entry
        self._spawn_run_record(job, entry)

    def _spawn_run_record(self, job: str, entry: dict[str, Any]) -> None:
        """Persist without making a job's success depend on the write.

        A run that did its work and then failed to write its own history should
        still count as a run, so this is fire-and-forget and logs on failure.
        """

        async def _write() -> None:
            try:
                await self._store.create(
                    {
                        "_id": run_doc_id(new_run_id()),
                        "type": "run",
                        "job": job,
                        "at": entry["at"],
                        # The whole summary, minus anything unserialisable. It is
                        # what the console shows when a row is expanded.
                        "summary": {
                            k: v for k, v in entry.items() if k != "at" and isinstance(v, _JSONABLE)
                        },
                    }
                )
            except Exception as exc:
                log.warning("run.record_failed", job=job, error=str(exc))

        task = asyncio.create_task(_write())
        # Held so the loop cannot garbage-collect the task mid-write.
        self._run_records.add(task)
        task.add_done_callback(self._run_records.discard)

    # --- stages -------------------------------------------------------------

    async def _queue(self, status: EpisodeStatus, limit: int) -> list[Doc]:
        """Newest-first work queue for a status.

        Recency is the priority. A digest is a weekly artefact, so an episode
        published this morning is worth more than one from ten days ago, and
        when capacity is short the fresh one should win. Every stage orders the
        same way, so an episode does not lose its place between stages.

        Archive material is excluded: backfill has its own runner, its own
        thresholds and its own output, and must never consume the routine
        pipeline's budget or reach the weekly digest (roadmap A1).
        """
        return await self._store.find(
            {
                "type": "episode",
                "status": status.value,
                **ROUTINE_ONLY,
            },
            sort=typed_sort("published_at", "desc"),
            limit=limit,
        )

    async def _stage_triage(self, stats: PipelineStats) -> None:
        episodes = await self._queue(EpisodeStatus.NEW, self._settings.pipeline.max_triage_per_run)
        if not episodes:
            return
        log.info("stage.triage_start", queued=len(episodes))
        for episode in episodes:
            try:
                await self._tier0.triage(episode)
                stats.triaged += 1
            except LLMUnavailable as exc:
                # Tier-0 chain down: stop the stage, keep the queue intact.
                stats.stages_deferred.append("triage")
                log.error("stage.triage_deferred", error=str(exc), remaining=len(episodes))
                return
            except Exception as exc:
                stats.errors += 1
                log.warning(
                    "stage.triage_episode_failed",
                    episode_id=episode["_id"],
                    error=str(exc),
                    exc_info=True,
                )
                await mark_error(self._store, episode["_id"], "tier0", exc)

    async def _stage_dispatch(self, stats: PipelineStats) -> None:
        episodes = await self._queue(EpisodeStatus.TRIAGED, 500)
        for episode in episodes:
            try:
                target = await self._tier0.dispatch(episode)
                stats.dispatched += 1
                match target:
                    case EpisodeStatus.DROPPED:
                        stats.dropped += 1
                    case EpisodeStatus.AWAITING_TRANSCRIPT:
                        stats.escalated += 1
                    case EpisodeStatus.DIGEST_DIRECT:
                        stats.digest_direct += 1
                    case _:
                        pass
            except Exception as exc:
                stats.errors += 1
                log.warning("stage.dispatch_failed", episode_id=episode["_id"], error=str(exc))
                await mark_error(self._store, episode["_id"], "dispatch", exc)

    async def _stage_transcripts(self, stats: PipelineStats) -> None:
        episodes = await self._queue(
            EpisodeStatus.AWAITING_TRANSCRIPT, self._settings.pipeline.max_transcripts_per_run
        )
        if not episodes:
            return
        log.info("stage.transcripts_start", queued=len(episodes))
        for episode in episodes:
            try:
                # Per-show switch: ASR is the expensive path, so only shows
                # explicitly marked for it may transcribe.
                allow_asr = self._registry.allows_asr(episode["podcast_slug"])
                outcome = await self._transcripts.process(episode, allow_asr=allow_asr)
                match outcome:
                    case EpisodeStatus.TRANSCRIBED:
                        stats.transcripts_ok += 1
                    case EpisodeStatus.TRANSCRIPT_FAILED:
                        stats.transcripts_failed += 1
                    case _:
                        stats.transcripts_retrying += 1
            except ASRUnavailable as exc:
                stats.stages_deferred.append("transcripts")
                log.error("stage.transcripts_deferred", error=str(exc))
                return
            except Exception as exc:
                stats.errors += 1
                log.warning(
                    "stage.transcript_episode_failed",
                    episode_id=episode["_id"],
                    error=str(exc),
                    exc_info=True,
                )
                await mark_error(self._store, episode["_id"], "transcript", exc)

    async def _stage_summarize(self, stats: PipelineStats) -> None:
        budget = self._settings.pipeline.max_summaries_per_run
        # Episodes with a transcript first; description-only fallbacks after.
        episodes = await self._queue(EpisodeStatus.TRANSCRIBED, budget)
        if len(episodes) < budget:
            episodes += await self._queue(EpisodeStatus.TRANSCRIPT_FAILED, budget - len(episodes))
        if not episodes:
            return
        log.info("stage.summarize_start", queued=len(episodes))
        for episode in episodes:
            try:
                outcome = await self._tier1.summarize(episode)
                if outcome is EpisodeStatus.READY_FOR_DIGEST:
                    stats.summarized += 1
                else:
                    stats.scored_low += 1
            except LLMUnavailable as exc:
                stats.stages_deferred.append("summarize")
                log.error("stage.summarize_deferred", error=str(exc), remaining=len(episodes))
                return
            except Exception as exc:
                stats.errors += 1
                log.warning(
                    "stage.summarize_episode_failed",
                    episode_id=episode["_id"],
                    error=str(exc),
                    exc_info=True,
                )
                await mark_error(self._store, episode["_id"], "tier1", exc)
