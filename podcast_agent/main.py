"""Application assembly and entrypoint (§10.5).

Wires configuration, storage, the LLM layer, pipeline stages, the scheduler and
the API together. Invalid configuration crashes here with a readable message
rather than starting half-configured (§8).
"""

from __future__ import annotations

import asyncio
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from . import __version__, logstore
from .api.auth import require_api_key
from .api.pages import page
from .api.podcasts import router as podcasts_router
from .api.routes import api_router, health_router
from .api.settings import router as settings_router
from .backfill.ingest import BackfillIngestor
from .backfill.process import BackfillProcessor
from .config import Settings, load_settings
from .db import CouchStore, Store, StoreError
from .digest.archive import ArchiveDigestGenerator
from .digest.generate import DigestGenerator
from .ingest.feeds import Ingestor
from .joblock import reclaim_local_leases
from .llm import build_llm_client
from .logging_setup import configure_logging, get_logger
from .migrate import run_all as run_migrations
from .net import UrlGuard, build_client
from .notify import Notifier
from .pipeline.runner import PipelineRunner
from .podcasts import PodcastRegistry
from .retention import RetentionJob
from .scheduler import build_scheduler, drain_jobs, mark_shutting_down
from .search import SearchIndex
from .settings_store import allowed_api_base_hosts, check_api_bases, get_overrides, mark_applied
from .signals import export_new_marks
from .summarize.tier1 import Tier1Stage
from .transcripts.acquire import TranscriptAcquirer
from .transcripts.asr import build_asr_backend
from .transcripts.stage import TranscriptStage
from .triage.tier0 import Tier0Stage
from .utils import iso_now

log = get_logger(__name__)

#: How long shutdown waits for a cancelled job to run its own cleanup — a
#: lease release is a database write, so it needs the store still open.
#: A job that declines to stop must not hold the whole shutdown open.
SHUTDOWN_GRACE_S = 10


def _warn_if_asr_unavailable(settings: Settings, registry: PodcastRegistry) -> None:
    """Say so at startup when podcasts want ASR that this install cannot do.

    `faster-whisper` is an optional extra, and a plain `uv sync` removes extras —
    which is exactly how it once disappeared mid-session here. Nothing failed
    loudly: transcript acquisition simply deferred, so episodes queued up behind
    a capability that had quietly gone away. One line at startup is cheaper than
    working that out from a stalled queue.
    """
    from importlib.util import find_spec

    if settings.asr.backend != "local":
        return
    wanting = [p.slug for p in registry.enabled_podcasts() if p.asr_enabled]
    if not wanting:
        return
    if find_spec("faster_whisper") is not None:
        log.info("asr.ready", model=settings.asr.model, podcasts=len(wanting))
        return
    log.warning(
        "asr.unavailable",
        detail=(
            "faster-whisper is not installed, so local transcription cannot run. "
            "Install the extra with: uv sync --all-extras"
        ),
        podcasts_expecting_asr=sorted(wanting),
    )


def build_app(settings: Settings, *, store: Store | None = None, llm: Any = None) -> FastAPI:
    """Construct the FastAPI app.

    ``store`` and ``llm`` are injectable so tests can assemble the real app
    against fakes without touching CouchDB or litellm.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.background_tasks = set()
        # When *this* process booted. The Settings page shows it next to the
        # "waiting for a restart" banner: a restart that missed the agent — the
        # CouchDB container restarted instead, say — otherwise looks exactly
        # like one that worked, and the banner just appears stuck.
        app.state.started_at = iso_now()

        active_store = store or CouchStore(
            settings.couchdb,
            settings.couchdb_password.get_secret_value() if settings.couchdb_password else None,
        )
        app.state.store = active_store
        await active_store.ensure_setup()
        # Only now is there somewhere to write to; the sink has been queueing
        # since configure_logging, so startup warnings are not lost.
        logstore.store.start(active_store)
        # Before anything reads episodes. A selector that matches on `origin`
        # cannot see a document that lacks it, so migrating while the pipeline
        # ran would hide work rather than merely delay it.
        await run_migrations(active_store, settings.output.digest_dir, settings.backfill.months)
        # Before the scheduler exists, so a job cannot be refused by a lock its
        # own predecessor left behind when it died.
        await reclaim_local_leases(active_store)

        # Two phases, necessarily: the database connection comes from the file,
        # and console overrides come from the database. Rebuild the settings with
        # the overrides applied, then run on those.
        # A distinct name, not a rebinding: assigning to `settings` here would
        # make it local to this closure and shadow the enclosing parameter.
        active_settings = settings
        # The file-and-environment configuration, kept for the console: an
        # override is judged against this rather than against whatever is
        # already running, so a stored override cannot authorise itself.
        app.state.baseline_settings = settings
        stored_overrides = await get_overrides(active_store)
        if stored_overrides:
            try:
                # Built and checked before it is adopted: assigning first would
                # leave a rejected override in force, which is the opposite of
                # what the fallback below is for.
                candidate = load_settings(overrides=stored_overrides)
                # Checked again here, not only where it was saved: the document
                # is writable by anything with database access, and this is the
                # last point before the endpoints become real.
                check_api_bases(candidate, allowed_api_base_hosts(settings))
                active_settings = candidate
                log.info("app.overrides_applied", sections=sorted(stored_overrides))
            except Exception as exc:
                # A stored override that no longer validates must not brick the
                # service; fall back to the file and say so loudly.
                log.error(
                    "app.overrides_invalid",
                    error=str(exc),
                    detail="running on config.yaml alone until this is corrected",
                )
        app.state.settings = active_settings
        await mark_applied(active_store)

        http_client = build_client()
        app.state.http_client = http_client
        guard = UrlGuard(active_settings.security)

        active_llm = llm if llm is not None else build_llm_client(active_settings, active_store)
        app.state.llm = active_llm

        notifier = Notifier(
            active_settings.notifications,
            http_client,
            active_settings.ntfy_token.get_secret_value() if active_settings.ntfy_token else None,
        )
        app.state.notifier = notifier
        if notifier.enabled:
            log.info(
                "app.notifications_enabled",
                min_score=active_settings.notifications.min_score,
                topic=active_settings.notifications.topic,
            )

        # A derived cache beside the database, not part of it: nothing in the
        # pipeline reads it and no decision depends on it, so it going stale
        # degrades a search box and nothing else.
        app.state.search = SearchIndex(active_settings, active_store)

        asr_backend = build_asr_backend(active_settings.asr)
        app.state.asr_backend = asr_backend

        # Output directories are created eagerly so a permissions problem shows
        # up at boot rather than at 06:00 on a Friday.
        output = active_settings.output
        for directory in (output.digest_dir, output.work_dir / "audio"):
            directory.mkdir(parents=True, exist_ok=True)

        ingestor = Ingestor(active_settings, active_store, http_client, guard)
        # §6: podcast documents are seeded from config at startup. Doing it only
        # during an ingest run meant a freshly deployed instance had no documents
        # to attach console overrides to, so the show list could not be edited
        # until the first poll had happened.
        await ingestor.seed_podcast_docs()

        registry = PodcastRegistry(active_settings)
        await registry.refresh(active_store)
        ingestor.use_registry(registry)
        app.state.registry = registry
        _warn_if_asr_unavailable(active_settings, registry)

        tier0_stage = Tier0Stage(active_settings, active_store, active_llm, registry)
        transcript_stage = TranscriptStage(
            active_settings,
            active_store,
            TranscriptAcquirer(
                active_settings, active_store, http_client, guard, asr_backend, registry
            ),
        )
        tier1_stage = Tier1Stage(active_settings, active_store, active_llm, notifier)

        runner = PipelineRunner(
            active_settings,
            active_store,
            ingestor=ingestor,
            tier0=tier0_stage,
            transcripts=transcript_stage,
            tier1=tier1_stage,
            # The LLM is what makes the digest's opening section possible
            # (roadmap D1). Passing it does not make digest generation depend on
            # a model: the section is skipped when one is unavailable.
            digest=DigestGenerator(active_settings, active_store, active_llm),
            registry=registry,
            # Archive backfill reuses the same stages; the differences are in
            # its own config (no ASR, stricter threshold) and its own output.
            backfill_ingest=BackfillIngestor(
                active_settings, active_store, http_client, guard, registry
            ),
            backfill_process=BackfillProcessor(
                active_settings,
                active_store,
                tier0=tier0_stage,
                transcripts=transcript_stage,
                tier1=tier1_stage,
                # The shared registry, so a console change is seen on the next
                # run rather than at the next restart.
                registry=registry,
            ),
            archive=ArchiveDigestGenerator(active_settings, active_store, registry),
        )
        app.state.runner = runner
        retention = RetentionJob(active_settings, active_store)
        app.state.retention = retention

        async def export_signals() -> dict[str, Any]:
            """Weekly, half an hour after the digest: a period's reader marks
            written into the vault where anything else can read them."""
            return await export_new_marks(active_store, active_settings)

        scheduler = build_scheduler(
            active_settings, runner, retention, app.state.search, signals=export_signals
        )
        scheduler.start()
        app.state.scheduler = scheduler

        log.info(
            "app.started",
            version=__version__,
            podcasts=len(registry.enabled_podcasts()),
            interests=len(active_settings.interest_profile),
            digest_dir=str(active_settings.output.digest_dir),
            timezone=active_settings.scheduler.timezone,
            asr_backend=active_settings.asr.backend,
        )

        if active_settings.scheduler.run_on_startup:
            log.info("app.startup_run_triggered")
            task = asyncio.create_task(_startup_run(runner))
            app.state.background_tasks.add(task)
            task.add_done_callback(app.state.background_tasks.discard)

        try:
            yield
        finally:
            log.info("app.stopping")
            # Before the shutdown that causes the cancellation, so an in-flight
            # job is logged as stopped rather than as failed.
            mark_shutting_down()
            scheduler.shutdown(wait=False)
            # wait=False cancels a running job and returns; the job's own
            # cleanup — releasing its database lease, which is a write — still
            # has to run, and it needs the store open to do it.
            await drain_jobs(SHUTDOWN_GRACE_S)
            # Before the store closes, so anything queued during shutdown —
            # which is when the interesting failures happen — still lands.
            await logstore.store.stop()
            running = list(app.state.background_tasks)
            for task in running:
                task.cancel()
            # Awaited, and awaited *here*: a cancelled job runs its own cleanup,
            # and that cleanup needs the store still open — releasing the job's
            # database lease is a write. Cancelling without waiting let the
            # release land after active_store.close(), so every restart during a
            # backfill orphaned control:lock:backfill and locked the job out
            # until the lease expired on its own.
            #
            # Bounded, because a task that declines to stop must not be able to
            # hold the whole shutdown open.
            if running:
                with suppress(TimeoutError):
                    async with asyncio.timeout(SHUTDOWN_GRACE_S):
                        await asyncio.gather(*running, return_exceptions=True)
            await http_client.aclose()
            await asr_backend.close()
            closer = getattr(active_llm, "close", None)
            if closer is not None:
                await closer()
            if store is None:
                await active_store.close()
            log.info("app.stopped")

    app = FastAPI(
        title="Podcast Digest Agent",
        version=__version__,
        summary="Cybersecurity podcast triage, summarisation and Markdown digests",
        lifespan=lifespan,
        # Docs are served behind the admin key instead (§9).
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # CORS is deliberately not enabled: this is a LAN-only JSON API (§9).
    app.include_router(health_router)
    app.include_router(api_router)
    app.include_router(podcasts_router)
    app.include_router(settings_router)

    @app.exception_handler(StoreError)
    async def _store_unavailable(_request: Any, exc: StoreError) -> JSONResponse:
        """A database that is briefly unreachable is not a server fault.

        Left unhandled this became "Exception in ASGI application" with a full
        httpx traceback in the log and a 500 in the browser — which reads like a
        bug in the console rather than a moment of contention underneath it.
        """
        log.warning("api.store_unavailable", error=str(exc)[:300])
        return JSONResponse(
            status_code=503,
            content={"detail": f"database unavailable: {str(exc)[:200]}"},
        )

    @app.get("/openapi.json", include_in_schema=False, dependencies=[Depends(require_api_key)])
    async def openapi_json() -> JSONResponse:
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False, dependencies=[Depends(require_api_key)])
    async def docs() -> HTMLResponse:
        return get_swagger_ui_html(openapi_url="/openapi.json", title="Podcast Digest Agent")

    # Console pages. Served without the API key on purpose: each is inert HTML
    # and JS carrying no data. The key is entered in the browser and sent as a
    # header on every request, which is the only way a plain page navigation can
    # authenticate against a header-keyed API — all data still goes through the
    # authenticated endpoints.
    for route, filename in (
        ("/admin", "admin.html"),
        ("/admin/digests", "digests.html"),
        ("/admin/episodes", "episodes.html"),
        ("/admin/podcasts", "podcasts.html"),
        ("/admin/insights", "insights.html"),
        ("/admin/backfill", "backfill.html"),
        ("/admin/logs", "logs.html"),
        ("/admin/settings", "settings.html"),
    ):

        def _console(route: str = route, filename: str = filename) -> Any:
            async def handler() -> HTMLResponse:
                # These pages carry their own JS and change with every upgrade,
                # and nothing in the URL changes when they do. Without this a
                # browser serves yesterday's console after a deploy and the new
                # control is simply absent — which is a support question, not a
                # visible failure. They are a few KB on a LAN; caching them buys
                # nothing worth that.
                return HTMLResponse(
                    page(filename, route),
                    headers={"Cache-Control": "no-store, must-revalidate"},
                )

            return handler

        app.get(route, include_in_schema=False)(_console())

    app.state.settings = settings
    app.state.admin_api_key = (
        settings.admin_api_key.get_secret_value() if settings.admin_api_key else None
    )
    return app


async def _startup_run(runner: PipelineRunner) -> None:
    try:
        await runner.run_ingest()
        await runner.run_pipeline()
    except Exception as exc:
        log.error("app.startup_run_failed", error=str(exc), exc_info=True)


def _load_or_die() -> Settings:
    try:
        return load_settings()
    except ValidationError as exc:
        # Config errors must be readable without a stack trace (§8).
        print("FATAL: invalid configuration\n", file=sys.stderr)
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(root)"
            print(f"  {location}: {error['msg']}", file=sys.stderr)
        print(
            "\nCheck config.yaml and the PODAGENT_* environment variables (see .env.example).",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    except (OSError, ValueError) as exc:
        print(f"FATAL: could not load configuration: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def create_app() -> FastAPI:
    """ASGI factory used by uvicorn (``podcast_agent.main:create_app``)."""
    settings = _load_or_die()
    configure_logging(settings.logging)
    if not settings.admin_api_key:
        log.warning(
            "app.no_admin_key",
            detail="PODAGENT_ADMIN_API_KEY is unset — all /api/v1 endpoints will return 503",
        )
    return build_app(settings)


def require_free_port(host: str, port: int) -> None:
    """Refuse to start when something is already listening.

    uvicorn runs the application's startup hooks *before* it binds the socket.
    A second instance started by mistake therefore does the whole lifespan —
    connects to CouchDB, runs migrations, starts the scheduler, and records the
    stored settings as applied, which clears the console's "waiting for a
    restart" banner — and only then dies on the port. The first process carries
    on serving the old configuration, so the restart looks like it worked while
    nothing whatsoever changed. Checking first turns that into a plain refusal.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            print(
                f"FATAL: {host}:{port} is already in use — another instance is "
                f"probably running. Stop it first, then start this one.\n"
                f"  pgrep -fl '[b]in/podcast-agent'\n"
                f"  pkill -TERM -f '[b]in/podcast-agent'",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc


def cli() -> None:
    """Console entrypoint: run the ASGI server."""
    import uvicorn

    settings = _load_or_die()
    configure_logging(settings.logging)
    # Before load_settings' side effects and before anything touches CouchDB.
    require_free_port(settings.api.host, settings.api.port)
    uvicorn.run(
        "podcast_agent.main:create_app",
        factory=True,
        host=settings.api.host,
        port=settings.api.port,
        log_config=None,  # structlog owns stdout
        access_log=False,
    )


if __name__ == "__main__":
    cli()
