"""structlog configuration: JSON to stdout, captured by `docker logs` (§10.1).

Secrets are redacted defensively at the processor level so that even an
accidental ``log.info("...", api_key=...)`` cannot leak a credential.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from . import logbuffer, logstore
from .config import LoggingConfig

#: Keys whose values are replaced with a marker wherever they appear in a log
#: event. Matched case-insensitively as substrings.
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "authorization",
    "credential",
)

_REDACTED = "***redacted***"

#: Keys that match a sensitive substring but are ordinary telemetry. Without this
#: allowlist, "token" inside "input_tokens" redacted the per-call token counts —
#: silently destroying the cost data the whole telemetry design exists to collect.
_NEVER_REDACT = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_tokens",
        "max_input_tokens",
        "chunk_target_tokens",
        "tokens",
    }
)

#: Keys that carry bulk untrusted content. Never logged in full (§10.1).
_BULK_KEYS = ("transcript", "description_raw", "prompt", "response", "summary_md")

_BULK_PREVIEW_CHARS = 200


def _redact_secrets(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    for key in list(event_dict):
        lowered = key.lower()
        if lowered in _NEVER_REDACT:
            continue
        if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
            event_dict[key] = _REDACTED
    return event_dict


def _truncate_bulk(_logger: Any, _name: str, event_dict: EventDict) -> EventDict:
    for key in list(event_dict):
        if not any(part in key.lower() for part in _BULK_KEYS):
            continue
        value = event_dict[key]
        if isinstance(value, str) and len(value) > _BULK_PREVIEW_CHARS:
            event_dict[key] = f"{value[:_BULK_PREVIEW_CHARS]}…[{len(value)} chars total]"
    return event_dict


#: Third-party warnings that are true, harmless, and repeated on every call.
#:
#: litellm warns that a model is absent from its built-in cost map and that
#: *cache* cost fields will therefore default to zero. It names the model by the
#: opaque id the Router assigns each deployment, so it fires for every deployment
#: on every construction. It is irrelevant here twice over: this deployment runs
#: local models that cost nothing, and it does not use prompt caching at all.
#: Cost is recorded per call in `llm_call` documents regardless.
_KNOWN_NOISE = (
    "not in built-in cost map",
    "cache cost fields will default to 0",
)


def _drop_known_noise(record: logging.LogRecord) -> bool:
    """Filter for third-party loggers: False drops the record."""
    message = str(record.getMessage())
    return not any(phrase in message for phrase in _KNOWN_NOISE)


def _demote_overlap_skips(record: logging.LogRecord) -> bool:
    """Recast APScheduler's overlap notice as the routine event it is.

    Backfill polls every 20 minutes but a run takes hours, so ``max_instances=1``
    refuses nearly every fire — which is the design, not a fault. At warning
    level that was ~40 identical lines a night, and it is how a repeatedly
    truncated audio download and a truncated LLM response in the same log went
    unread for a week.
    """
    if "maximum number of running instances reached" not in str(record.getMessage()):
        return True
    record.levelno = logging.DEBUG
    record.levelname = "DEBUG"
    # The record has already passed this logger's own level check, so lowering
    # levelno alone would not stop it reaching the handler. Dropping it here is
    # what keeps it out of the log at INFO while leaving it visible at DEBUG.
    return logging.getLogger().getEffectiveLevel() <= logging.DEBUG


def tame_litellm_logging() -> None:
    """Strip litellm's own handler and filter its known-noise warning.

    Called twice on purpose. Once from :func:`configure_logging`, and again
    after litellm is imported — litellm attaches the handler at import time, and
    the app configures logging before it imports litellm, so doing it once would
    have no effect on the process that matters.

    Without it every litellm line appears twice: as JSON through this chain, and
    as its own coloured text straight to stderr.
    """
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        if _drop_known_noise not in logger.filters:
            logger.addFilter(_drop_known_noise)


def configure_logging(cfg: LoggingConfig) -> None:
    """Install the structlog pipeline. Idempotent — safe to call more than once.

    Everything, including third-party stdlib logging from uvicorn/httpx/litellm,
    is rendered by the same processor chain and lands on stdout in one format
    (§10.1). That matters practically: the documented `docker logs | jq` workflow
    breaks the moment a plain-text line is interleaved with the JSON.
    """
    level = logging.getLevelNamesMapping()[cfg.level]

    # Applied to structlog events and to foreign stdlib records alike, so a
    # uvicorn line carries the same timestamp format and redaction as ours.
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _redact_secrets,
        _truncate_bulk,
    ]

    renderer: Processor
    if cfg.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # A stdlib LoggerFactory (not PrintLoggerFactory) is required: add_logger_name
    # reads `logger.name`, which only a stdlib logger has.
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            # The console's log tail. Placed here rather than in `shared` so it
            # sees every event exactly once — structlog's own and uvicorn's
            # alike — after redaction and after the traceback has been rendered
            # into a string. It is a tap: stdout is unaffected.
            logbuffer.sink,
            # Warning and above also go to CouchDB, so a restart does not
            # take the answer to "what went wrong" with it. Queued here,
            # written by a background task — never on this thread.
            logstore.sink,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace rather than append, so repeated calls cannot duplicate every line.
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    quiet_level = max(logging.WARNING, level)
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "apscheduler.executors.default"):
        logging.getLogger(noisy).setLevel(quiet_level)

    scheduler_log = logging.getLogger("apscheduler.scheduler")
    if _demote_overlap_skips not in scheduler_log.filters:
        scheduler_log.addFilter(_demote_overlap_skips)

    tame_litellm_logging()
    # uvicorn installs its own handlers; clear them so lines are not emitted twice.
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(uvicorn_logger).handlers = []
        logging.getLogger(uvicorn_logger).propagate = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_run(run_id: str, **extra: Any) -> None:
    """Bind run-scoped context so every subsequent log line carries run_id (§10.1)."""
    structlog.contextvars.bind_contextvars(run_id=run_id, **extra)


def clear_run_context() -> None:
    structlog.contextvars.clear_contextvars()
