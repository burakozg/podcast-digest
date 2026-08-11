"""Admin and health API (§9). LAN-only — see the deployment notes in the README.

All routes except ``/healthz`` require the admin key. Responses never include
transcripts, prompts or secrets.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from .. import logbuffer, logstore
from ..backfill.control import get_state as get_control_state
from ..backfill.control import rewind_cursors as rewind_backfill_cursors
from ..backfill.control import set_paused as set_control_paused
from ..backfill.ingest import floor_month
from ..config import WINDOW_CHOICES, Settings
from ..content import ContentSeedBuilder
from ..content import render as render_seeds
from ..content import select as select_seed_episodes
from ..content import write as write_seeds
from ..db import TRANSCRIPT_ATTACHMENT, Doc, Selector, Store, typed_sort
from ..digest.read import DigestUnreadable, read_digest
from ..entities import (
    DEFAULT_MIN_MENTIONS,
    aggregate,
    canonical,
    digest_weeks,
    rank,
    timeline,
    window_start,
    write_entity_notes,
)
from ..episodes import transition
from ..feedback import MAX_NOTE_CHARS, set_read, set_starred, set_verdict
from ..feedback import view as feedback_view
from ..insights import precision_report
from ..joblock import current_holders
from ..logging_setup import get_logger
from ..pipeline.runner import JobBusy, PipelineRunner, pending_routine_episodes
from ..podcasts import PodcastRegistry
from ..retention import RetentionJob
from ..sanitize import safe_url
from ..search import FIELDS as SEARCH_FIELDS
from ..search import SearchIndex, SearchUnavailable
from ..signals import export_new_marks
from ..state import (
    BACKFILL_ORIGIN,
    ROUTINE_ONLY,
    EpisodeStatus,
    IllegalTransition,
    retry_target,
)
from ..utils import digest_doc_id, iso, iso_now, podcast_doc_id, utcnow
from .auth import require_api_key

log = get_logger(__name__)

#: Jobs the scheduler owns, in the order the console lists them.
SCHEDULED_JOBS = (
    "ingest",
    "pipeline",
    "digest",
    "backfill",
    "rescore",
    "retention",
    "search_sync",
)

#: Documents examined when filtering by score. Comfortably above the whole
#: episode collection today; a cap rather than an expectation.
SCORE_FILTER_SCAN = 5000

#: What the pipeline will do next with an episode in each unfinished state,
#: in the words the console uses elsewhere.
NEXT_STEP = {
    EpisodeStatus.NEW.value: "waiting to be triaged",
    EpisodeStatus.TRIAGED.value: "waiting to be routed",
    EpisodeStatus.AWAITING_TRANSCRIPT.value: "fetching or transcribing",
    EpisodeStatus.TRANSCRIBED.value: "waiting to be summarised",
    EpisodeStatus.TRANSCRIPT_FAILED.value: (
        "no transcript — will be summarised from its description"
    ),
    EpisodeStatus.SUMMARIZED.value: "waiting to be scored",
}

health_router = APIRouter(tags=["health"])
api_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


def _settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _store(request: Request) -> Store:
    return request.app.state.store  # type: ignore[no-any-return]


def _registry(request: Request) -> PodcastRegistry:
    return request.app.state.registry  # type: ignore[no-any-return]


def _runner(request: Request) -> PipelineRunner:
    return request.app.state.runner  # type: ignore[no-any-return]


def _search(request: Request) -> SearchIndex:
    return request.app.state.search  # type: ignore[no-any-return]


# --- health -----------------------------------------------------------------


@health_router.get("/healthz", summary="Liveness and dependency check")
async def healthz(request: Request) -> dict[str, Any]:
    """No auth (§9). Reports process, CouchDB and scheduler state."""
    store: Store | None = getattr(request.app.state, "store", None)
    scheduler = getattr(request.app.state, "scheduler", None)

    couch_ok = False
    if store is not None:
        try:
            couch_ok = await store.ping()
        except Exception as exc:
            log.warning("healthz.couch_probe_failed", error=str(exc))

    scheduler_running = bool(scheduler and getattr(scheduler, "running", False))
    healthy = couch_ok and (scheduler_running or scheduler is None)
    payload = {
        "status": "ok" if healthy else "degraded",
        "couchdb": "ok" if couch_ok else "unreachable",
        "scheduler": "running" if scheduler_running else "stopped",
        "time": iso_now(),
    }
    if not healthy:
        # 503 so a container healthcheck actually fails.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=payload)
    return payload


# --- status -----------------------------------------------------------------


@api_router.get("/status", summary="Pipeline counts, feed health and last run summaries")
async def get_status(request: Request) -> dict[str, Any]:
    store = _store(request)
    settings = _settings(request)
    runner = _runner(request)

    counts: dict[str, int] = {}
    routine_counts: dict[str, int] = {}
    for episode_status in EpisodeStatus:
        counts[episode_status.value] = await store.count(
            {"type": "episode", "status": episode_status.value}
        )
        # Split out, because the two halves are moved by different jobs. A queue
        # of 82 that "Process queue" leaves untouched is not a stuck pipeline —
        # it is 82 archive episodes waiting on the archive walk.
        routine_counts[episode_status.value] = await store.count(
            {"type": "episode", "status": episode_status.value, **ROUTINE_ONLY}
        )

    # Through the registry, not `settings`. The file is only half the list:
    # a podcast added in the console lives in the database, so reading config
    # here reported feed health for 14 podcasts while 16 were being polled —
    # and a failing feed among the other seven would never have shown.
    registry = _registry(request)
    await registry.refresh(store)

    feeds: list[dict[str, Any]] = []
    for podcast in registry.enabled_podcasts():
        doc = await store.get(podcast_doc_id(podcast.slug)) or {}
        failures = int(doc.get("consecutive_failures") or 0)
        feeds.append(
            {
                "slug": podcast.slug,
                "name": podcast.name,
                "priority": podcast.priority.value,
                "always_escalate": podcast.always_escalate,
                "last_polled_at": doc.get("last_polled_at"),
                "consecutive_failures": failures,
                "circuit_open": failures >= 5,
                "last_error": doc.get("last_error"),
                # Where the archive walk has got to for this show (roadmap A1).
                # Without this the cursor is stored but invisible, and "where did
                # we leave off?" has no answer short of reading the database.
                "backfill": {
                    "mode": podcast.backfill_mode,
                    "cursor": doc.get("backfill_cursor"),
                    "complete": bool(doc.get("backfill_complete")),
                    "updated_at": doc.get("backfill_updated_at"),
                },
            }
        )

    return {
        "episode_counts": counts,
        "queue_depths": _queue_depths(counts, await _awaiting_digest(store)),
        # The same queue, restricted to episodes the routine pipeline will act
        # on. Everything else is archive material and moves only when the
        # archive walk runs.
        "queue_depths_routine": _queue_depths(
            routine_counts, await _awaiting_digest(store, ROUTINE_ONLY)
        ),
        "jobs_running": {
            job: runner.is_running(job)
            for job in ("ingest", "pipeline", "digest", "rescore", "backfill")
        },
        # Jobs held by *any* process against this database, this one included.
        # `jobs_running` above only ever knew about this server, so a run started
        # by a second instance or a CLI invocation looked like nothing at all.
        "jobs_held": [
            {
                "job": lock.get("job"),
                "host": lock.get("host"),
                "pid": lock.get("pid"),
                "acquired_at": lock.get("acquired_at"),
                "expires_at": lock.get("expires_at"),
                "this_process": lock.get("pid") == os.getpid(),
            }
            for lock in await current_holders(store)
        ],
        "backfill": await _backfill_progress(store, settings, registry),
        # Roadmap C2: how many scored episodes predate the current profile.
        "interest_profile": {
            "version": settings.interest_profile_version(),
            "stale_episodes": len(await runner.stale_episodes(limit=1000)),
        },
        "last_runs": runner.last_runs,
        "feeds": feeds,
        "config": {
            "digest_threshold": settings.pipeline.digest_threshold,
            "t_conf_high": settings.pipeline.t_conf_high,
            "t_rel_low": settings.pipeline.t_rel_low,
            "t_rel_high": settings.pipeline.t_rel_high,
            "timezone": settings.scheduler.timezone,
            "tiers": {
                tier: [e.litellm_model() for e in cfg.active_chain()]
                for tier, cfg in settings.llm.tiers.items()
            },
            "allow_cloud_fallback": {
                tier: cfg.allow_cloud_fallback for tier, cfg in settings.llm.tiers.items()
            },
            "asr_backend": settings.asr.backend,
            "episode_notes": settings.output.episode_notes,
        },
        "time": iso_now(),
    }


#: Statuses an episode sits in while it still expects to be written somewhere.
CLAIMABLE_STATUSES = (EpisodeStatus.READY_FOR_DIGEST, EpisodeStatus.DIGEST_DIRECT)


async def _awaiting_digest(store: Store, extra: Selector | None = None) -> int:
    """Episodes a digest could still claim.

    Status alone overstates it. An episode summarised on request *after* it was
    already listed keeps its claim, so nothing will ever pick it up again — it
    is finished, not waiting. Counting it as awaiting produces the same lie as
    labelling a published archive episode "queued": a number that implies work
    pending on a decision already taken, and which never drains.
    """
    total = 0
    for episode_status in CLAIMABLE_STATUSES:
        total += await store.count(
            {
                "type": "episode",
                "status": episode_status.value,
                "digest_id": None,
                **(extra or {}),
            }
        )
    return total


def _queue_depths(counts: dict[str, int], awaiting_digest: int) -> dict[str, int]:
    return {
        "triage": counts[EpisodeStatus.NEW.value],
        "dispatch": counts[EpisodeStatus.TRIAGED.value],
        "transcripts": counts[EpisodeStatus.AWAITING_TRANSCRIPT.value],
        "summarize": counts[EpisodeStatus.TRANSCRIBED.value]
        + counts[EpisodeStatus.TRANSCRIPT_FAILED.value],
        "awaiting_digest": awaiting_digest,
    }


async def _backfill_progress(
    store: Store, settings: Settings, registry: PodcastRegistry
) -> dict[str, Any]:
    """Overall archive progress: how far back each podcast has reached (roadmap A1).

    Read through the registry, not straight off `settings`. The walk itself uses
    the registry, so reading config here meant the page could report a mode or a
    window that the next run would not use — a console disagreeing with the
    thing it is a console for.
    """
    await registry.refresh(store)
    eligible = [p for p in registry.enabled_podcasts() if p.backfill_mode != "skip"]
    default_months = settings.backfill.months
    now = utcnow()
    # The headline floor is the furthest any podcast reaches, since that is what
    # bounds the walk overall.
    widest = max((p.backfill_months or default_months for p in eligible), default=default_months)
    floor = floor_month(now, widest)

    complete = 0
    cursors: dict[str, str | None] = {}
    podcasts: list[dict[str, Any]] = []
    for podcast in eligible:
        doc = await store.get(podcast_doc_id(podcast.slug)) or {}
        cursor = doc.get("backfill_cursor")
        cursors[podcast.slug] = cursor
        done = bool(doc.get("backfill_complete"))
        if done:
            complete += 1
        months = podcast.backfill_months or default_months
        # Carries the display name, so the console never has to show a slug to
        # someone who chose the podcast by its name.
        podcasts.append(
            {
                "slug": podcast.slug,
                "name": podcast.name,
                "mode": podcast.backfill_mode,
                "months": months,
                # False when it simply inherits the configured default, so the
                # console can show which podcasts were deliberately changed.
                "months_overridden": podcast.backfill_months is not None,
                "oldest_month_targeted": floor_month(now, months),
                "cursor": cursor,
                "complete": done,
                "updated_at": doc.get("backfill_updated_at"),
            }
        )

    # Why the archive walk is not moving, when it is not moving. Without this
    # a deferred backfill is indistinguishable from a broken one.
    waiting = await pending_routine_episodes(store, limit=200)

    return {
        "control": await get_control_state(store),
        "waiting_on_recent": {
            "pending": len(waiting),
            "blocked": bool(waiting),
            "episodes": [
                {
                    "episode_id": doc.get("_id"),
                    "podcast_slug": doc.get("podcast_slug"),
                    "title": doc.get("title"),
                    "status": doc.get("status"),
                    "published_at": doc.get("published_at"),
                    # A bare status reads as a fault when most of these are
                    # ordinary steps. TRANSCRIPT_FAILED in particular is the
                    # normal outcome for a podcast that publishes no transcript
                    # with local transcription off — the episode still gets a
                    # summary, from its description.
                    "next_step": NEXT_STEP.get(str(doc.get("status")), "waiting"),
                }
                for doc in waiting[:10]
            ],
        },
        "window_months_default": default_months,
        "window_choices": list(WINDOW_CHOICES),
        "oldest_month_targeted": floor,
        "digest_threshold": settings.backfill.digest_threshold,
        "shows_eligible": len(eligible),
        "shows_complete": complete,
        "shows_not_started": sum(1 for c in cursors.values() if c is None),
        "cursors": cursors,
        "podcasts": podcasts,
        "episodes_ingested": await store.count({"type": "episode", "origin": BACKFILL_ORIGIN}),
        "archive_files": await store.count({"type": "archive"}),
    }


# --- manual runs ------------------------------------------------------------


async def _launch(request: Request, job: str, coro_factory: Any, wait: bool) -> dict[str, Any]:
    runner = _runner(request)
    if runner.is_running(job):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"{job} is already running"
        )
    task = _spawn(request, job, coro_factory)
    if wait:
        try:
            # shield: if the caller hangs up, the run still finishes.
            result = await asyncio.shield(task)
        except JobBusy as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return {"job": job, "started": True, "waited": True, "result": result}

    # Fire-and-forget: ingest and pipeline runs routinely outlive an HTTP request.
    return {
        "job": job,
        "started": True,
        "waited": False,
        "detail": "running in background; poll /api/v1/status",
    }


def _spawn(request: Request, job: str, coro_factory: Any) -> asyncio.Task[Any]:
    """Start work as a task owned by the app, not by the request.

    Awaiting a coroutine inline means a client disconnect cancels it: closing a
    browser tab mid-run threw away minutes of local LLM time and left the episode
    back where it started. The task is registered on the app so it outlives the
    connection either way.
    """
    task = asyncio.create_task(coro_factory())
    tasks: set[asyncio.Task[Any]] = request.app.state.background_tasks
    tasks.add(task)

    def _done(finished: asyncio.Task[Any]) -> None:
        tasks.discard(finished)
        if not finished.cancelled() and finished.exception() is not None:
            log.error("api.job_failed", job=job, error=str(finished.exception()))

    task.add_done_callback(_done)
    return task


async def _background(job: str, coro_factory: Any) -> None:
    try:
        await coro_factory()
    except JobBusy:
        log.info("api.job_skipped_busy", job=job)
    except Exception as exc:
        log.error("api.job_failed", job=job, error=str(exc), exc_info=True)


@api_router.post("/runs/ingest", summary="Poll all feeds now")
async def run_ingest(
    request: Request,
    wait: bool = Query(default=False, description="Block until the run finishes"),
) -> dict[str, Any]:
    runner = _runner(request)
    return await _launch(request, "ingest", runner.run_ingest, wait)


@api_router.post("/runs/pipeline", summary="Process pending work now")
async def run_pipeline(
    request: Request,
    wait: bool = Query(default=False, description="Block until the run finishes"),
) -> dict[str, Any]:
    runner = _runner(request)
    return await _launch(request, "pipeline", runner.run_pipeline, wait)


@api_router.post("/runs/digest", summary="Generate a digest now")
async def run_digest(
    request: Request,
    since: datetime | None = Query(
        default=None,
        description="Start of the period (ISO-8601); defaults to the last digest's end",
    ),
    until: datetime | None = Query(default=None, description="End of the period (ISO-8601)"),
    dry_run: bool = Query(
        default=False, description="Render and report without writing or marking"
    ),
) -> dict[str, Any]:
    runner = _runner(request)

    # Digest generation is DB reads plus rendering — fast enough to run inline.
    async def _go() -> dict[str, Any]:
        result = await runner.run_digest(
            since=_as_utc(since), until=_as_utc(until), dry_run=dry_run
        )
        return result.as_dict()

    try:
        return {"job": "digest", "started": True, "waited": True, "result": await _go()}
    except JobBusy as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@api_router.get("/backfill/control", summary="Is the archive walk running or paused?")
async def get_backfill_control(request: Request) -> dict[str, Any]:
    return await get_control_state(_store(request))


@api_router.post("/backfill/control", summary="Start or pause the archive walk")
async def set_backfill_control(
    request: Request,
    paused: bool = Query(description="true to pause, false to start/resume"),
    note: str = Query(default="", max_length=200),
) -> dict[str, Any]:
    """Roadmap B2. Pausing takes effect at the next episode boundary — an
    in-flight episode finishes rather than being wasted."""
    return await set_control_paused(_store(request), paused, note=note)


@api_router.post("/backfill/rewind", summary="Walk the archive window again")
async def rewind_backfill(
    request: Request,
    podcast: str = Query(default="", description="One slug, or empty for every podcast"),
    confirm: bool = Query(default=False, description="Required: this queues hours of work"),
) -> dict[str, Any]:
    """Clear archive cursors so months already walked are re-read.

    Needed because the walk only moves backwards: a month it has passed cannot
    be revisited, including one passed under a policy that discarded most of
    what it saw. Nothing is deleted — ingestion is create-if-absent, so a second
    pass adds only what was missing and leaves existing episodes, summaries and
    digest claims untouched.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="refusing to rewind without confirm=true; this re-walks the whole window",
        )
    return await rewind_backfill_cursors(_store(request), slug=podcast or None)


@api_router.post("/runs/backfill", summary="Walk the archive backwards (dry run by default)")
async def run_backfill(
    request: Request,
    dry_run: bool = Query(default=True, description="Estimate only; write nothing"),
    confirm: bool = Query(
        default=False, description="Required to actually spend compute (dry_run=false)"
    ),
    force: bool = Query(
        default=False,
        description="Run even while recent episodes are still in the pipeline",
    ),
) -> dict[str, Any]:
    """Roadmap A1. Defaults to a dry run that reports what it would cost."""
    runner = _runner(request)
    try:
        result = await runner.run_backfill(dry_run=dry_run, confirm=confirm, force=force)
    except ValueError as exc:
        # Missing confirmation is a client mistake, not a server fault.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobBusy as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return {"job": "backfill", "result": result}


@api_router.post("/runs/rescore", summary="Re-score episodes against the current interest profile")
async def run_rescore(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    force: bool = Query(
        default=False, description="Re-score even episodes already on the current profile"
    ),
) -> dict[str, Any]:
    """Roadmap C2. Uses stored transcripts, so this costs Tier-1 tokens only."""
    runner = _runner(request)
    try:
        return {"job": "rescore", "result": await runner.run_rescore(limit=limit, force=force)}
    except JobBusy as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@api_router.post("/runs/retention", summary="Run retention cleanup now")
async def run_retention(request: Request) -> dict[str, Any]:
    retention: RetentionJob = request.app.state.retention
    return {"job": "retention", "result": await retention.run()}


# --- episodes ---------------------------------------------------------------

#: Fields returned by the episode endpoints. Excludes transcripts by construction.
_EPISODE_FIELDS = (
    "_id",
    "podcast_slug",
    "podcast_name",
    "title",
    "link",
    "published_at",
    "duration_s",
    "status",
    "transcript_source",
    "transcript_chars",
    "digest_id",
    "indexed_only",
    "summary_after_listing",
    "attempts",
    "created_at",
    "updated_at",
)


def _effective_score(doc: dict[str, Any]) -> tuple[int | None, bool]:
    """The score to judge an episode by, and whether it is still provisional.

    A summarised episode has a final score. A grey-zone one was judged from its
    description alone and only ever gets the triage guess — which is the number
    worth filtering on for it, so "show me everything above 7" does not silently
    exclude every episode that was never summarised.

    Expressed here rather than in Mango because "tier1 if present, else tier0"
    is not a comparison a selector can make: an `$or` across both fields would
    match an episode whose triage guessed 9 and whose summary then scored 3.
    """
    tier1 = doc.get("tier1") or {}
    if (final := tier1.get("relevance_score")) is not None:
        return int(final), False
    guess = (doc.get("tier0") or {}).get("relevance_guess")
    return (int(guess), True) if guess is not None else (None, False)


async def _require_episode(request: Request, episode_id: str) -> dict[str, Any]:
    """The episode, or a 404. Shared by every per-episode action."""
    doc = await _store(request).get(_normalize_episode_id(episode_id))
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such episode")
    return doc


def _episode_view(doc: dict[str, Any], *, verbose: bool = False) -> dict[str, Any]:
    view: dict[str, Any] = {key: doc.get(key) for key in _EPISODE_FIELDS}
    tier0 = doc.get("tier0") or {}
    tier1 = doc.get("tier1") or {}
    view["tier0"] = (
        {
            "relevance_guess": tier0.get("relevance_guess"),
            "confidence": tier0.get("confidence"),
            "route": tier0.get("route"),
            "rule": tier0.get("rule"),
            "matched_interests": tier0.get("matched_interests"),
        }
        if tier0
        else None
    )
    view["tier1"] = (
        {
            "relevance_score": tier1.get("relevance_score"),
            "summary_basis": tier1.get("summary_basis"),
            "matched_interests": tier1.get("matched_interests"),
            "listen_anyway": tier1.get("listen_anyway"),
        }
        if tier1
        else None
    )
    # The questions a browser actually asks: is there a summary, and is there a
    # transcript behind it? Both are otherwise buried in nested blocks.
    view["has_summary"] = bool(tier1.get("summary_md"))
    view["has_transcript"] = TRANSCRIPT_ATTACHMENT in (doc.get("_attachments") or {})
    # One definition of "the score", shared by the table, the filter and anyone
    # reading the API — the page used to re-derive the fallback in JavaScript.
    view["score"], view["score_provisional"] = _effective_score(doc)
    view["origin"] = doc.get("origin") or "feed"
    # Reader signals, on the list view as well as the detail one: a browsing
    # surface has to be able to show what is starred without a request per row.
    view.update(feedback_view(doc))
    view["archive_month"] = doc.get("archive_month")
    if error := doc.get("last_error"):
        view["last_error"] = (
            error if verbose else {k: v for k, v in error.items() if k != "traceback"}
        )
    if verbose:
        view["tier0_full"] = tier0 or None
        # Includes summary_md, key_takeaways and entities — the actual reading
        # material, which the list view deliberately omits for size.
        view["tier1_full"] = tier1 or None
    return view


@api_router.get("/episodes", summary="List episodes (paged, filterable)")
async def list_episodes(
    request: Request,
    episode_status: str | None = Query(default=None, alias="status"),
    podcast: str | None = Query(default=None, description="Podcast slug"),
    min_score: int = Query(default=0, ge=0, le=10, description="Effective score floor"),
    max_score: int = Query(default=10, ge=0, le=10, description="Effective score ceiling"),
    starred: bool | None = Query(default=None, description="Only starred, or only unstarred"),
    unread: bool | None = Query(default=None, description="True for unread only"),
    flagged: bool | None = Query(default=None, description="True for episodes with a verdict"),
    summarised: bool | None = Query(
        default=None, description="True for episodes carrying a summary, False for those without"
    ),
    limit: int = Query(default=50, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    store = _store(request)
    selector: dict[str, Any] = {"type": "episode"}
    if episode_status:
        if episode_status not in {s.value for s in EpisodeStatus}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown status {episode_status!r}",
            )
        selector["status"] = episode_status
    if podcast:
        selector["podcast_slug"] = podcast

    # Signal filters are applied in Python for the same reason as the score:
    # `starred: false` and an absent `starred` are the same thing to a reader
    # and different things to Mango, and a selector on absence cannot use an
    # index anyway (see db.base.resolve_index).
    # `summarised` belongs with them rather than in the selector: it asks about
    # a nested field that is absent far more often than it is present, and Mango
    # cannot index absence. Asking it there would scan the whole archive to
    # answer "which of these have summaries" — the exact question the reader is
    # asking, on the exact documents where the answer is usually no.
    signal_filter = (
        starred is not None or unread is not None or flagged is not None or summarised is not None
    )

    def _matches_signals(doc: dict[str, Any]) -> bool:
        if starred is not None and bool(doc.get("starred")) is not starred:
            return False
        if unread is not None and (doc.get("read_at") is None) is not unread:
            return False
        if summarised is not None:
            # The same definition the table renders and the drawer protects:
            # a summary is summary text, not a tier1 block that exists.
            has_summary = bool((doc.get("tier1") or {}).get("summary_md"))
            if has_summary is not summarised:
                return False
        return not (flagged is not None and bool(doc.get("feedback")) is not flagged)

    if min_score or max_score < 10 or signal_filter:
        # Paged in Python, because the score cannot be selected on: the database
        # would have to know which of two fields applies per document. Bounded
        # by SCORE_FILTER_SCAN so this stays a scan of a page-sized problem
        # rather than of the whole archive.
        matching = await store.find(
            selector, sort=typed_sort("published_at", "desc"), limit=SCORE_FILTER_SCAN
        )
        # An unscored episode is excluded from either end. It has not been
        # judged, which is a different thing from having been judged badly —
        # "below 4" means triage rejected it, not that nothing looked yet.
        kept = []
        for doc in matching:
            if not _matches_signals(doc):
                continue
            score = _effective_score(doc)[0]
            unscored_ok = not (min_score or max_score < 10)
            if unscored_ok or (score is not None and min_score <= score <= max_score):
                kept.append(doc)
        window = kept[skip : skip + limit]
        return {
            "count": len(window),
            "total": len(kept),
            "scanned": len(matching),
            # True when the scan cap was hit, so the console can say the totals
            # are a floor rather than quietly under-reporting.
            "truncated": len(matching) >= SCORE_FILTER_SCAN,
            "skip": skip,
            "limit": limit,
            "episodes": [_episode_view(d) for d in window],
        }

    docs = await store.find(
        selector, sort=typed_sort("published_at", "desc"), limit=limit, skip=skip
    )
    return {
        "count": len(docs),
        # Total matching the filter, so a pager can show "1-50 of 237" and know
        # whether there is a next page without over-fetching.
        "total": await store.count(selector),
        "skip": skip,
        "limit": limit,
        "episodes": [_episode_view(d) for d in docs],
    }


@api_router.get("/episodes/{episode_id}", summary="One episode in full (no transcript)")
async def get_episode(request: Request, episode_id: str) -> dict[str, Any]:
    return _episode_view(await _require_episode(request, episode_id), verbose=True)


class FeedbackIn(BaseModel):
    """The reader's view of what the pipeline decided (roadmap B1)."""

    model_config = ConfigDict(extra="forbid")

    #: `over` = summarised but not worth it. `under` = dropped or downgraded and
    #: should not have been. Null clears the verdict.
    verdict: Literal["over", "under"] | None = None
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


@api_router.post("/episodes/{episode_id}/star", summary="Star or unstar an episode")
async def star_episode(
    request: Request, episode_id: str, starred: bool = Query(default=True)
) -> dict[str, Any]:
    doc = await _require_episode(request, episode_id)
    updated = await set_starred(_store(request), doc["_id"], starred)
    return {"episode_id": doc["_id"], **feedback_view(updated)}


@api_router.post("/episodes/{episode_id}/read", summary="Mark an episode read or unread")
async def read_episode(
    request: Request, episode_id: str, read: bool = Query(default=True)
) -> dict[str, Any]:
    doc = await _require_episode(request, episode_id)
    updated = await set_read(_store(request), doc["_id"], read)
    return {"episode_id": doc["_id"], **feedback_view(updated)}


@api_router.post("/episodes/{episode_id}/feedback", summary="Say the call was wrong")
async def episode_feedback(
    request: Request, episode_id: str, body: Annotated[FeedbackIn, Body()]
) -> dict[str, Any]:
    """Record that triage got this episode wrong, in which direction, and why.

    Nothing acts on it automatically. It is captured so a monthly precision
    report — and later a few-shot prompt — has demonstrated taste to work from
    rather than the static profile alone. Demoting a show because three episodes
    went unstarred is how a digest quietly stops showing things.
    """
    doc = await _require_episode(request, episode_id)
    updated = await set_verdict(_store(request), doc["_id"], body.verdict, body.note)
    return {"episode_id": doc["_id"], **feedback_view(updated)}


@api_router.post("/episodes/{episode_id}/retry", summary="Reset a failed episode")
async def retry_episode(request: Request, episode_id: str) -> dict[str, Any]:
    store = _store(request)
    doc_id = _normalize_episode_id(episode_id)
    doc = await store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such episode")

    current = EpisodeStatus(doc["status"])
    # Prefer the status the episode actually held when it failed: a Tier-0 error
    # must go back to NEW for triage, not forward to AWAITING_TRANSCRIPT, which
    # would skip triage entirely and leave the episode with no tier0 verdict.
    target = _resume_point(doc) or retry_target(current)
    if target == current and current is not EpisodeStatus.AWAITING_TRANSCRIPT:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"status {current.value} is not retryable",
        )

    def _apply(d: dict[str, Any]) -> None:
        # Both counters, or this silently does nothing for the episodes most
        # likely to need it. `transcript_crash` retires an episode that was in
        # flight when the process died — usually the container's memory limit
        # rather than anything about the episode — and once that is addressed,
        # this is the button an operator reaches for. Leaving it set means the
        # stage gives up again before touching the audio, while the API still
        # answers 200.
        d["attempts"] = {
            **(d.get("attempts") or {}),
            "transcript": 0,
            "transcript_crash": 0,
        }
        d["last_error"] = None

    try:
        await transition(store, doc_id, target, mutate=_apply)
    except IllegalTransition as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    log.info("api.episode_retried", episode_id=doc_id, from_status=current.value, to=target.value)
    return {"episode_id": doc_id, "from": current.value, "to": target.value}


@api_router.post("/episodes/{episode_id}/summarize", summary="Summarise one episode now")
async def summarize_episode(
    request: Request,
    episode_id: str,
    allow_asr: bool = Query(
        default=True,
        description="False restricts acquisition to a published transcript (fast, free)",
    ),
    wait: bool = Query(
        default=False, description="Block until finished; ASR runs can take a long time"
    ),
) -> dict[str, Any]:
    """Owner override: acquire a transcript and summarise this episode.

    Always reaches a verdict — if no transcript can be obtained the summary is
    made from the description and labelled honestly, rather than leaving the
    episode queued with nothing to show.
    """
    runner = _runner(request)
    doc_id = _normalize_episode_id(episode_id)
    if await _store(request).get(doc_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such episode")
    if runner.is_episode_in_flight(doc_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="already being summarised")

    async def _go() -> dict[str, Any]:
        return await runner.summarize_episode(doc_id, allow_asr=allow_asr)

    # Started as a task either way: a summarisation that outlives the request is
    # normal, and losing it to a closed tab would waste real LLM time.
    task = _spawn(request, f"summarize:{doc_id}", _go)
    if wait:
        try:
            return {"episode_id": doc_id, "waited": True, "result": await asyncio.shield(task)}
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except JobBusy as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    # Transcription can take far longer than a browser will wait, so the default
    # is to return now and let the caller watch the episode's status change.
    return {
        "episode_id": doc_id,
        "waited": False,
        "allow_asr": allow_asr,
        "detail": "started; poll the episode for its status",
    }


@api_router.post("/episodes/{episode_id}/escalate", summary="Force full Tier-1 treatment")
async def escalate_episode(request: Request, episode_id: str) -> dict[str, Any]:
    """Owner override for a dropped or low-scored episode (§9)."""
    store = _store(request)
    doc_id = _normalize_episode_id(episode_id)
    doc = await store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such episode")

    current = EpisodeStatus(doc["status"])

    def _apply(d: dict[str, Any]) -> None:
        # Both counters, or this silently does nothing for the episodes most
        # likely to need it. `transcript_crash` retires an episode that was in
        # flight when the process died — usually the container's memory limit
        # rather than anything about the episode — and once that is addressed,
        # this is the button an operator reaches for. Leaving it set means the
        # stage gives up again before touching the audio, while the API still
        # answers 200.
        d["attempts"] = {
            **(d.get("attempts") or {}),
            "transcript": 0,
            "transcript_crash": 0,
        }
        d["last_error"] = None
        d["forced_escalation"] = {"from": current.value, "at": iso_now()}
        # Clear the digest claim so the re-summarised episode can appear again.
        d["digest_id"] = None

    try:
        await transition(store, doc_id, EpisodeStatus.AWAITING_TRANSCRIPT, mutate=_apply)
    except IllegalTransition as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"cannot escalate from {current.value}: {exc}",
        ) from exc

    log.info("api.episode_escalated", episode_id=doc_id, from_status=current.value)
    return {
        "episode_id": doc_id,
        "from": current.value,
        "to": EpisodeStatus.AWAITING_TRANSCRIPT.value,
    }


def _resume_point(doc: dict[str, Any]) -> EpisodeStatus | None:
    """The status this episode held when it failed, if it is safe to resume there."""
    recorded = (doc.get("last_error") or {}).get("status_when_failed")
    if not recorded:
        return None
    try:
        resumed = EpisodeStatus(recorded)
    except ValueError:
        return None
    # ERROR itself is not a resume point, and neither is a terminal status.
    if resumed in (EpisodeStatus.ERROR, EpisodeStatus.PUBLISHED):
        return None
    return resumed


def _normalize_episode_id(episode_id: str) -> str:
    """Accept both the bare hash and the full ``episode:<hash>`` id."""
    return episode_id if episode_id.startswith("episode:") else f"episode:{episode_id}"


# --- glance (roadmap E2) ----------------------------------------------------


@api_router.get("/content/seeds", summary="Episodes that qualify as writing material")
async def preview_seeds(request: Request) -> dict[str, Any]:
    """What a generate would consider, without spending a call on it.

    Useful on its own: an empty list means the filters are too narrow, and that
    is worth knowing before blaming the model.
    """
    settings = _settings(request)
    episodes = await select_seed_episodes(_store(request), settings)
    return {
        "enabled": settings.content.enabled,
        "window_days": settings.content.window_days,
        "min_score": settings.content.min_score,
        "interests": settings.content.interests,
        "count": len(episodes),
        "episodes": [
            {
                "episode_id": e["_id"],
                "podcast_name": e.get("podcast_name") or e.get("podcast_slug"),
                "title": e.get("title"),
                "published_at": e.get("published_at"),
                "score": (e.get("tier1") or {}).get("relevance_score"),
                "matched_interests": (e.get("tier1") or {}).get("matched_interests") or [],
            }
            for e in episodes
        ],
    }


@api_router.post("/content/seeds", summary="Find openings worth writing about")
async def build_seeds(request: Request) -> dict[str, Any]:
    """Roadmap E3. One call over the qualifying summaries, written to the vault."""
    settings = _settings(request)
    if not settings.content.enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "content seeds are switched off; set content.enabled and choose the "
                "interests you write about"
            ),
        )
    builder = ContentSeedBuilder(settings, _store(request), request.app.state.llm)
    seeds, episodes = await builder.build()
    if seeds is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "no candidate episodes in the window"
                if not episodes
                else "the model was unavailable; nothing was written"
            ),
        )
    path = write_seeds(settings, render_seeds(seeds, episodes, settings))
    return {
        "episodes_considered": len(episodes),
        "seeds": len(seeds.seeds),
        "threads": len(seeds.threads),
        "file_path": str(path.relative_to(settings.output.digest_dir)),
    }


@api_router.post("/signals/export", summary="Write this period's reader marks to the vault")
async def export_signals(request: Request) -> dict[str, Any]:
    """Runs weekly on its own; this forces it early.

    Writes only what has been marked since the last export and moves the cursor,
    so calling it twice does not repeat a mark — the second call finds nothing
    new and writes nothing.
    """
    result: dict[str, Any] = await export_new_marks(_store(request), _settings(request))
    return result


@api_router.get("/insights/precision", summary="Is the interest profile still right?")
async def get_precision(
    request: Request,
    days: int = Query(default=90, ge=7, le=3650),
) -> dict[str, Any]:
    """Roadmap C1 phase 1. Suggestions only — nothing here is ever applied.

    A system that demotes a show because three of its episodes went unstarred is
    one that quietly stops showing you things, and you never see what you were
    no longer shown.
    """
    return await precision_report(_store(request), _settings(request), days=days)


@api_router.get("/entities", summary="Named things across the corpus, most discussed first")
async def list_entities(
    request: Request,
    days: int = Query(default=0, ge=0, le=3650, description="0 for the whole corpus"),
    min_mentions: int = Query(default=DEFAULT_MIN_MENTIONS, ge=1, le=100),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """Roadmap D2. Aggregated on demand — nothing here is stored or cached.

    Tier-1 has been extracting these all along and nothing read them back.
    """
    found = await aggregate(_store(request), since=window_start(days))
    ranked = rank(found, min_mentions=min_mentions)
    return {
        "days": days or None,
        "min_mentions": min_mentions,
        "total": len(ranked),
        "entities": [e.as_dict() for e in ranked[:limit]],
    }


@api_router.get("/entities/{key}", summary="One entity, with its episodes and monthly shape")
async def get_entity(
    request: Request, key: str, days: int = Query(default=0, ge=0)
) -> dict[str, Any]:
    found = await aggregate(_store(request), since=window_start(days))
    entity = found.get(canonical(key))
    if entity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such entity")
    return {**entity.as_dict(with_episodes=True), "timeline": timeline(entity)}


@api_router.post("/entities/notes", summary="Write one Obsidian note per entity")
async def write_entities(
    request: Request,
    days: int = Query(default=0, ge=0, le=3650),
    min_mentions: int = Query(default=DEFAULT_MIN_MENTIONS, ge=1, le=100),
) -> dict[str, Any]:
    """Rewrites `entities/` in the digest directory.

    Wholesale rather than incrementally: a note is a view of the corpus, and a
    stale line in one is worse than a rebuilt file because the reader cannot
    tell which lines are current.
    """
    store = _store(request)
    found = await aggregate(store, since=window_start(days))
    ranked = rank(found, min_mentions=min_mentions)
    paths = write_entity_notes(_settings(request), ranked, week_of=await digest_weeks(store))
    return {"written": len(paths), "paths": paths[:50]}


@api_router.get("/search", summary="Full-text search over summaries and transcripts")
async def search(
    request: Request,
    q: str = Query(min_length=1, max_length=200, description="Words to look for"),
    field: str | None = Query(
        default=None, description=f"Restrict to one of {sorted(SEARCH_FIELDS)}"
    ),
    limit: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    index = _search(request)
    try:
        results = await index.search(q, field=field, limit=limit)
    except SearchUnavailable as exc:
        # 409, not 500: the index is a cache that may simply not exist yet, and
        # the fix is a rebuild rather than a bug report.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return {"query": q, "field": field, "count": len(results), "results": results}


@api_router.get("/search/status", summary="Whether the search index exists, and how big")
async def search_status(request: Request) -> dict[str, Any]:
    return await _search(request).stats()


@api_router.post("/search/sync", summary="Index what has changed since the last sync")
async def search_sync(request: Request) -> dict[str, Any]:
    """The routine operation, and what the scheduler runs.

    Only episodes whose content actually changed are re-indexed, so this is
    cheap enough to run often — it is what stops the index being as current as
    the last manual rebuild and no more.
    """
    return await _search(request).sync()


@api_router.post("/search/rebuild", summary="Rebuild the search index from scratch")
async def search_rebuild(request: Request) -> dict[str, Any]:
    """The repair operation. The index is derived, so this is always safe."""
    return await _search(request).rebuild()


@api_router.get("/glance", summary="One-line status for a small display")
async def glance(request: Request) -> dict[str, Any]:
    """Compact payload for the e-ink family calendar to poll.

    Deliberately a teaser, not a reading surface: a headline count plus the
    single best item. The display pipeline owns rendering; this owns the words.
    """
    store = _store(request)
    settings = _settings(request)

    pending = await store.count({"type": "episode", "status": EpisodeStatus.READY_FOR_DIGEST.value})
    latest = await store.find({"type": "digest"}, sort=typed_sort("generated_at", "desc"), limit=1)
    top = await store.find(
        {"type": "episode", "status": EpisodeStatus.READY_FOR_DIGEST.value},
        sort=typed_sort("published_at", "desc"),
        limit=50,
    )
    best = max(
        top,
        key=lambda e: int((e.get("tier1") or {}).get("relevance_score") or 0),
        default=None,
    )

    headline = f"Podcast digest: {pending} new summar{'y' if pending == 1 else 'ies'}"
    if best is not None:
        score = int((best.get("tier1") or {}).get("relevance_score") or 0)
        show = str(best.get("podcast_name") or best.get("podcast_slug") or "")
        headline += f", top: {show} {score}/10"

    return {
        "headline": headline[:120],
        "pending_summaries": pending,
        "top_pick": (
            {
                "podcast": best.get("podcast_name"),
                "title": best.get("title"),
                "score": int((best.get("tier1") or {}).get("relevance_score") or 0),
                "link": safe_url(best.get("link")),
            }
            if best is not None
            else None
        ),
        "last_digest": (
            {
                "period": latest[0]["_id"].split(":", 1)[1],
                "generated_at": latest[0].get("generated_at"),
            }
            if latest
            else None
        ),
        "generated_at": iso_now(),
        "timezone": settings.scheduler.timezone,
    }


# --- telemetry --------------------------------------------------------------


@api_router.get("/telemetry/costs", summary="LLM cost and latency by provider/model/tier/day")
async def telemetry_costs(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
) -> dict[str, Any]:
    store = _store(request)
    cutoff = iso(utcnow() - timedelta(days=days))

    totals = {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "fallbacks": 0}
    by_provider: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)
    by_model: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)
    by_tier: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)
    by_day: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)

    skip = 0
    page = 500
    while True:
        batch = await store.find(
            {"type": "llm_call", "ts": {"$gte": cutoff}}, limit=page, skip=skip
        )
        if not batch:
            break
        for call in batch:
            cost = float(call.get("cost_usd") or 0.0)
            latency = int(call.get("latency_ms") or 0)
            tokens_in = int(call.get("input_tokens") or 0)
            tokens_out = int(call.get("output_tokens") or 0)
            totals["calls"] += 1
            totals["cost_usd"] += cost
            totals["input_tokens"] += tokens_in
            totals["output_tokens"] += tokens_out
            totals["fallbacks"] += 1 if call.get("fallback_used") else 0
            for bucket, key in (
                (by_provider, str(call.get("provider"))),
                (by_model, str(call.get("model"))),
                (by_tier, str(call.get("tier"))),
                (by_day, str(call.get("ts") or "")[:10]),
            ):
                entry = bucket[key]
                entry["calls"] += 1
                entry["cost_usd"] += cost
                entry["input_tokens"] += tokens_in
                entry["output_tokens"] += tokens_out
                entry["latency_ms_total"] += latency
        if len(batch) < page:
            break
        skip += page

    return {
        "window_days": days,
        "since": cutoff,
        "totals": {**totals, "cost_usd": round(totals["cost_usd"], 6)},
        # Transcription, kept apart rather than folded in: an ASR run has no
        # tokens and no price, and averaging its minutes against a triage call's
        # seconds would ruin both numbers.
        "asr": await _asr_telemetry(_store(request), cutoff),
        "by_provider": _finalize(by_provider),
        "by_model": _finalize(by_model),
        "by_tier": _finalize(by_tier),
        "by_day": _finalize(by_day),
    }


async def _asr_telemetry(store: Store, cutoff: str) -> dict[str, Any]:
    """What local transcription spent, in the units that suit it.

    Time, not money: these models run here and cost nothing to call. The figure
    worth watching is the realtime factor — seconds of audio per second of
    compute — which says whether the chosen model and machine are a sane pairing.
    """
    runs = await store.find({"type": "asr_run", "ts": {"$gte": cutoff}}, limit=10_000)
    by_model: dict[str, dict[str, Any]] = {}
    by_podcast: dict[str, dict[str, Any]] = {}

    def _add(bucket: dict[str, dict[str, Any]], key: str, run: Doc) -> None:
        entry = bucket.setdefault(key, {"runs": 0, "audio_s": 0, "elapsed_s": 0.0})
        entry["runs"] += 1
        entry["audio_s"] += int(run.get("audio_duration_s") or 0)
        entry["elapsed_s"] += float(run.get("elapsed_s") or 0.0)

    for run in runs:
        _add(by_model, f"{run.get('model')} on {run.get('device')}", run)
        _add(by_podcast, str(run.get("podcast_slug") or "?"), run)

    def _finish(bucket: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "runs": e["runs"],
                "audio_hours": round(e["audio_s"] / 3600, 2),
                "compute_hours": round(e["elapsed_s"] / 3600, 2),
                "realtime_factor": round(e["audio_s"] / e["elapsed_s"], 2)
                if e["elapsed_s"]
                else None,
            }
            for key, e in sorted(bucket.items())
        }

    audio = sum(int(r.get("audio_duration_s") or 0) for r in runs)
    elapsed = sum(float(r.get("elapsed_s") or 0.0) for r in runs)
    return {
        "runs": len(runs),
        "audio_hours": round(audio / 3600, 2),
        "compute_hours": round(elapsed / 3600, 2),
        "realtime_factor": round(audio / elapsed, 2) if elapsed else None,
        "by_model": _finish(by_model),
        "by_podcast": _finish(by_podcast),
    }


def _empty_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms_total": 0,
    }


def _finalize(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, entry in sorted(buckets.items()):
        calls = entry["calls"] or 1
        out[key] = {
            "calls": entry["calls"],
            "cost_usd": round(entry["cost_usd"], 6),
            "input_tokens": entry["input_tokens"],
            "output_tokens": entry["output_tokens"],
            "avg_latency_ms": round(entry["latency_ms_total"] / calls),
        }
    return out


# --- digests ----------------------------------------------------------------


def _digest_period_key(doc: Doc) -> str:
    """``digest:2026-W31`` → ``2026-W31``."""
    return str(doc.get("_id", "")).split(":", 1)[-1]


def _digest_runs(doc: Doc) -> list[dict[str, Any]]:
    """Every generation for this week, oldest first.

    Documents written before runs were recorded have none, so the top-level
    fields stand in for the single run they describe.
    """
    runs = list(doc.get("runs") or [])
    if runs:
        return runs
    return [
        {
            "file_path": doc.get("file_path"),
            "period": doc.get("period") or {},
            "episode_ids": doc.get("episode_ids") or [],
            "stats": doc.get("stats") or {},
            "generated_at": doc.get("generated_at"),
        }
    ]


def _digest_summary(doc: Doc) -> dict[str, Any]:
    period = doc.get("period") or {}
    return {
        "digest_id": doc["_id"],
        "period_key": _digest_period_key(doc),
        "period": period,
        "from": period.get("from"),
        "to": period.get("to"),
        "file_path": doc.get("file_path"),
        "episodes": len(doc.get("episode_ids") or []),
        "stats": doc.get("stats"),
        # False means generation was interrupted between writing the file and
        # claiming its episodes: readable, but a rerun may still add to it.
        "marking_complete": doc.get("marking_complete"),
        "generated_at": doc.get("generated_at"),
        # A week can hold more than one: the generator never overwrites, so a
        # second run writes -r2 beside the first rather than replacing it.
        "runs": [
            {
                "run": index + 1,
                "file_path": run.get("file_path"),
                "from": (run.get("period") or {}).get("from"),
                "to": (run.get("period") or {}).get("to"),
                "episodes": len(run.get("episode_ids") or []),
                "generated_at": run.get("generated_at"),
            }
            for index, run in enumerate(_digest_runs(doc))
        ],
    }


@api_router.get("/digests", summary="List generated digests")
async def list_digests(
    request: Request, limit: int = Query(default=20, ge=1, le=200)
) -> dict[str, Any]:
    docs = await _store(request).find(
        {"type": "digest"}, sort=typed_sort("generated_at", "desc"), limit=limit
    )
    # Ordered by period rather than by generation time. Regenerating an old week
    # gives it the newest `generated_at`, which would otherwise shuffle it to the
    # top of a list whose whole purpose is to read as a calendar. ISO week keys
    # sort correctly as strings.
    ordered = sorted(docs, key=_digest_period_key, reverse=True)
    return {
        "count": len(ordered),
        "digests": [_digest_summary(d) for d in ordered],
    }


@api_router.get("/digests/{period_key}", summary="One digest, rendered for reading")
async def get_digest(
    request: Request,
    period_key: str,
    run: int = Query(default=0, ge=0, description="1-based run; 0 means the most recent"),
) -> dict[str, Any]:
    """Return a digest's metadata plus its Markdown and rendered HTML.

    Read from the file rather than rebuilt from the database, so the console
    cannot disagree with what actually landed in the digest directory — and a
    digest edited by hand shows as edited.
    """
    doc = await _store(request).get(digest_doc_id(period_key))
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no digest for {period_key}"
        )
    runs = _digest_runs(doc)
    if run > len(runs):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{period_key} has {len(runs)} run(s), not {run}",
        )
    # A week holds one entry per generation, because the generator writes a new
    # file rather than overwriting. Run 0 means the most recent.
    chosen = runs[run - 1] if run else runs[-1]
    summary = {
        **_digest_summary(doc),
        "run": run or len(runs),
        # The document's top-level file_path names the most recent run, so
        # reporting it alongside an older run's content would describe a file
        # that is not the one returned.
        "file_path": chosen.get("file_path"),
    }
    if run:
        period = chosen.get("period") or {}
        summary |= {
            "from": period.get("from"),
            "to": period.get("to"),
            "episodes": len(chosen.get("episode_ids") or []),
            "generated_at": chosen.get("generated_at"),
            "stats": chosen.get("stats") or {},
        }

    relative = chosen.get("file_path")
    if not relative:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{period_key} has no file recorded",
        )
    settings: Settings = request.app.state.settings
    try:
        content = read_digest(settings.output.digest_dir, str(relative))
    except DigestUnreadable as exc:
        # The digest directory belongs to the user, who may prune, move or sync
        # it. A missing file is a fact to report, not a server fault.
        log.warning("digest.unreadable", period_key=period_key, error=str(exc))
        raise HTTPException(status_code=status.HTTP_410_GONE, detail=str(exc)) from exc
    return {**summary, **content}


# --- activity: logs, job runs, model calls -----------------------------------


@api_router.get("/logs", summary="Recent log events held in memory")
async def get_logs(
    request: Request,
    level: str = Query(default="", description="Minimum severity: debug/info/warning/error"),
    contains: str = Query(default="", description="Substring match across every field"),
    logger: str = Query(default="", description="Restrict to one logger name"),
    limit: int = Query(default=200, ge=1, le=1000),
    since_seq: int = Query(default=0, ge=0, description="Only events newer than this seq"),
) -> dict[str, Any]:
    """The tail of the process log, newest first.

    In memory and bounded, so it is empty after a restart and holds roughly the
    last hour of a busy pipeline. The durable record of what actually happened
    is `/runs` and `/telemetry/costs`.
    """
    events = logbuffer.buffer.tail(
        limit=limit,
        level=level or None,
        contains=contains or None,
        logger=logger or None,
        since_seq=since_seq or None,
    )
    return {
        "count": len(events),
        "held": len(logbuffer.buffer),
        "capacity": logbuffer.buffer.capacity,
        "levels": logbuffer.buffer.levels(),
        "events": events,
    }


@api_router.get("/logs/stored", summary="Warnings and errors kept in the database")
async def get_stored_logs(
    request: Request,
    level: str = Query(default="", description="Exact level: warning, error, critical"),
    contains: str = Query(default="", description="Substring match on the event name"),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """The durable half of the log: what survived a restart.

    Only warning and above is kept, and identical events written in the same
    moment are one row with an occurrence count. Episode and feed failures are
    not duplicated here — they live on the episode and podcast documents, where
    they carry more structure than a log line could.
    """
    selector: dict[str, Any] = {"type": "log"}
    if level:
        selector["level"] = level
    docs = await _store(request).find(selector, sort=typed_sort("at", "desc"), limit=limit)
    if contains:
        needle = contains.lower()
        docs = [d for d in docs if needle in str(d.get("event", "")).lower()]
    return {
        "count": len(docs),
        "retention_days": _settings(request).retention.log_days,
        "queued": len(logstore.store),
        "dropped": logstore.store.dropped,
        "events": [
            {k: v for k, v in doc.items() if not k.startswith("_") and k != "type"} for doc in docs
        ],
    }


@api_router.get("/runs", summary="Job runs recorded, newest first")
async def list_runs(
    request: Request,
    job: str = Query(default="", description="Restrict to one job"),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    selector: dict[str, Any] = {"type": "run"}
    if job:
        selector["job"] = job
    docs = await _store(request).find(selector, sort=typed_sort("at", "desc"), limit=limit)
    return {
        "count": len(docs),
        "runs": [
            {
                "run_id": str(d.get("_id", "")).split(":", 1)[-1],
                "job": d.get("job"),
                "at": d.get("at"),
                "summary": d.get("summary") or {},
            }
            for d in docs
        ],
    }


@api_router.get("/runs/last", summary="When each job last ran")
async def last_runs(request: Request) -> dict[str, Any]:
    """One row per job, from the durable record rather than process memory.

    `last_runs` on the runner is emptied by a restart, which is precisely when
    someone asks the question.
    """
    runner = _runner(request)
    store = _store(request)
    out: dict[str, Any] = {}
    for job in SCHEDULED_JOBS:
        docs = await store.find({"type": "run", "job": job}, sort=typed_sort("at", "desc"), limit=1)
        latest = docs[0] if docs else None
        out[job] = {
            "at": (latest or {}).get("at"),
            "summary": (latest or {}).get("summary") or {},
            "running": runner.is_running(job),
            # Set while the process has run it since booting, which distinguishes
            # "never run" from "not run since the last restart".
            "this_process": job in runner.last_runs,
        }
    return {"jobs": out, "time": iso_now()}


__all__ = ["api_router", "health_router"]
