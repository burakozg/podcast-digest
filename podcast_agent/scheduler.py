"""APScheduler wiring (§11).

Every job is ``max_instances=1`` + ``coalesce=True`` so a slow run can never
overlap the next firing; missed fires collapse into one catch-up run.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import Settings
from .logging_setup import get_logger
from .pipeline.runner import JobBusy, PipelineRunner
from .retention import RetentionJob
from .search import SearchIndex

log = get_logger(__name__)

#: True once shutdown has begun; see mark_shutting_down.
_shutting_down = False

#: Grace period for a fire that arrives late (e.g. the loop was busy).
MISFIRE_GRACE_S = 600

#: Tasks for jobs currently running, so shutdown can wait for their cleanup.
#: Weak, so a finished job's task is not kept alive by being remembered.
_running: weakref.WeakSet[asyncio.Task[None]] = weakref.WeakSet()


def build_scheduler(
    settings: Settings,
    runner: PipelineRunner,
    retention: RetentionJob,
    search: SearchIndex | None = None,
    signals: Callable[[], Awaitable[Any]] | None = None,
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(
        timezone=settings.scheduler.timezone,
        job_defaults={
            "max_instances": 1,
            "coalesce": True,
            "misfire_grace_time": MISFIRE_GRACE_S,
        },
    )

    jobs: list[tuple[str, str, Any]] = [
        ("ingest", settings.scheduler.ingest_cron, runner.run_ingest),
        ("pipeline", settings.scheduler.pipeline_cron, runner.run_pipeline),
        ("digest_weekly", settings.scheduler.digest_cron, runner.run_digest),
        ("retention_cleanup", settings.scheduler.retention_cron, retention.run),
        # No-ops while paused, which is the default state.
        ("backfill", settings.scheduler.backfill_cron, runner.run_backfill_scheduled),
    ]
    if signals is not None:
        jobs.append(("signals_export", settings.scheduler.signals_cron, signals))
    if search is not None:
        # Without this the index is only ever as current as the last manual
        # rebuild, which is the failure mode nobody notices: search quietly
        # stops finding this week's episodes and looks like it works.
        jobs.append(("search_sync", settings.scheduler.search_cron, search.sync))

    for job_id, cron, func in jobs:
        scheduler.add_job(
            _guarded(job_id, func),
            trigger=CronTrigger.from_crontab(cron, timezone=settings.scheduler.timezone),
            id=job_id,
            name=job_id,
            replace_existing=True,
        )
        log.info("scheduler.job_registered", job=job_id, cron=cron, tz=settings.scheduler.timezone)

    return scheduler


def mark_shutting_down() -> None:
    """Tell the job wrapper that cancellation from here on is expected.

    Set before the scheduler is stopped, so a job torn down mid-flight is
    reported as what it is rather than as a failure.
    """
    global _shutting_down
    _shutting_down = True


async def drain_jobs(grace_s: float) -> int:
    """Wait for cancelled scheduler jobs to finish their own cleanup.

    ``scheduler.shutdown(wait=False)`` cancels a running job and returns
    immediately, which is right — a backfill runs for hours and shutdown cannot
    wait for it to finish. But cancellation is not instant: the job's ``finally``
    still has to run, and for a job holding a database lease that cleanup is a
    *write*. Without this it landed after the store had closed:

        joblock.release_failed  Cannot send a request, as the client has been closed

    APScheduler owns those tasks, so tracking them here is the only way to know
    what to wait for. Returns how many were still running.
    """
    outstanding = [task for task in _running if not task.done()]
    if not outstanding:
        return 0
    with suppress(TimeoutError):
        async with asyncio.timeout(grace_s):
            await asyncio.gather(*outstanding, return_exceptions=True)
    still = sum(1 for task in outstanding if not task.done())
    if still:
        log.warning(
            "scheduler.jobs_still_running_at_shutdown",
            count=still,
            timeout_s=grace_s,
            detail="their cleanup may not have completed",
        )
    return len(outstanding)


def _guarded(job_id: str, func: Any) -> Any:
    """Swallow JobBusy so a manual trigger overlapping a scheduled fire is a
    logged no-op rather than a scheduler error."""

    async def _run() -> None:
        # Registered so shutdown can wait for this job's cleanup. APScheduler
        # keeps no handle we can await, and the set is weak so a finished job
        # does not pin its own task.
        task = asyncio.current_task()
        if task is not None:
            _running.add(task)
        try:
            await func()
        except JobBusy:
            log.info("scheduler.job_skipped_busy", job=job_id)
        except asyncio.CancelledError:
            # CancelledError is a BaseException, so the clause below never saw
            # it: every restart during a transcription or an LLM call escaped to
            # APScheduler and was logged as "raised an exception". Restarting is
            # not a failure, and dressing it as one buries the real ones — now
            # doubly so, since errors are kept in the database.
            if not _shutting_down:
                raise
            log.info("scheduler.job_cancelled_at_shutdown", job=job_id)
        except Exception as exc:
            # A job must never kill the scheduler.
            log.error("scheduler.job_failed", job=job_id, error=str(exc), exc_info=True)

    _run.__name__ = f"job_{job_id}"
    return _run
