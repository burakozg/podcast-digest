"""Durable record of warnings and errors, for looking back later.

The in-memory tail in :mod:`podcast_agent.logbuffer` answers "what is happening
now" and is emptied by a restart — which is exactly when someone asks what went
wrong. Episode and feed failures are already persisted where they belong
(``episode.last_error``, ``podcast.last_error``, ``run`` summaries), so this is
deliberately *not* a general log table. It captures the events that belong to no
episode and no run: a scheduler job dying, the database becoming unreachable, a
backend missing at startup, an override refused.

Four constraints, each of which matters more than the feature:

**It cannot block anything.** Log calls are synchronous and happen inside the
pipeline; a database write must never sit in that path. Events go onto a bounded
queue and a background task drains it.

**It cannot recurse.** Reporting a failed write would log, which queues, which
fails again. The drain path never logs through the normal chain.

**It cannot grow without bound.** The queue is bounded, so a database outage
drops old events rather than consuming memory, and repeated identical events are
collapsed into one document with a count — the lesson from a single missing
index once writing thousands of copies of one warning.

**It cannot capture everything.** When CouchDB is unreachable, the thing that
would record the error is the thing that is broken. Those events exist only on
stdout, which is the argument for keeping the container's own logs somewhere —
not an argument against this.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from .utils import iso_now

if TYPE_CHECKING:  # pragma: no cover - import cycle
    # Imported for typing only. At runtime `logging_setup` imports this module,
    # and `db` imports `logging_setup`, so importing `db` here would close the
    # loop — and break for any entry point that reaches logging_setup first.
    from .db import Store

#: Severities kept. Info is the running commentary; this is the exceptions.
KEEP_LEVELS = frozenset({"warning", "warn", "error", "critical", "exception"})

#: Bounded so an unreachable database drops the oldest events rather than
#: growing until the process dies. Losing the tail of a long outage is a far
#: better failure than falling over while trying to record it.
QUEUE_LIMIT = 1000

#: How often the drain task writes what has accumulated.
DRAIN_INTERVAL_SECONDS = 5.0

#: Documents written per drain. Identical events within a batch collapse into
#: one row carrying an occurrence count.
DRAIN_BATCH = 100

#: Fields that are rendered noise rather than information, or already columns.
_DROP_KEYS = frozenset({"_record", "_from_structlog", "positional_args", "seq"})

#: Values are truncated before storage; a traceback is the usual offender.
MAX_VALUE_CHARS = 4000


def log_doc_id() -> str:
    return f"log:{uuid.uuid4()}"


class LogStore:
    """Queue events here; a background task writes them."""

    def __init__(self, capacity: int = QUEUE_LIMIT) -> None:
        self._queue: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._task: asyncio.Task[None] | None = None
        self._store: Store | None = None
        #: Events dropped because the queue was full while the writer was stuck.
        self.dropped = 0

    # --- the sync side, called from the logging pipeline --------------------

    def offer(self, event: dict[str, Any]) -> None:
        """Queue an event if it is worth keeping. Never raises, never blocks."""
        if str(event.get("level", "")).lower() not in KEEP_LEVELS:
            return
        with suppress(Exception):
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
            self._queue.append(
                {key: _trim(value) for key, value in event.items() if key not in _DROP_KEYS}
            )

    def __len__(self) -> int:
        return len(self._queue)

    # --- the async side, owned by the application lifespan ------------------

    def start(self, store: Store) -> None:
        if self._task is not None:
            return
        self._store = store
        self._task = asyncio.create_task(self._drain_forever())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await self.flush()

    async def _drain_forever(self) -> None:
        # Cancellation must leave the loop, not be swallowed and looped on —
        # otherwise stop() hangs forever waiting for a task that keeps going.
        while True:
            await asyncio.sleep(DRAIN_INTERVAL_SECONDS)
            await self.flush()

    async def flush(self) -> None:
        """Write what has accumulated. Swallows its own failures, silently.

        Silent by necessity, not by preference: logging a failure here would
        queue another event, which would fail the same way. A dropped record is
        the cost of not taking the logging pipeline down with the database.
        """
        if self._store is None or not self._queue:
            return
        batch = [self._queue.popleft() for _ in range(min(DRAIN_BATCH, len(self._queue)))]
        for doc in _collapse(batch):
            with suppress(Exception):
                await self._store.create({"_id": log_doc_id(), "type": "log", **doc})


#: How much of an over-long value is kept from the front. The rest comes from
#: the end, because a Python traceback puts the exception type and message
#: *last*: clipping the tail kept a stack of framework frames and threw away the
#: line that says what actually went wrong. These are the events kept so that a
#: restart does not take the answer with it, so the answer is the part to keep.
_HEAD_SHARE = 0.3


def _trim(value: Any) -> Any:
    if not isinstance(value, str) or len(value) <= MAX_VALUE_CHARS:
        return value
    head = int(MAX_VALUE_CHARS * _HEAD_SHARE)
    tail = MAX_VALUE_CHARS - head
    dropped = len(value) - MAX_VALUE_CHARS
    return f"{value[:head]}\n…[{dropped} chars omitted]…\n{value[-tail:]}"


def _collapse(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold identical events into one document with a count.

    One missing index once produced the same warning on every query. Storing
    each copy would turn a single actionable fact into thousands of rows.
    """
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in batch:
        key = (
            str(event.get("level", "")),
            str(event.get("event", "")),
            str(event.get("logger", "")),
        )
        if existing := grouped.get(key):
            existing["occurrences"] += 1
            existing["last_at"] = event.get("timestamp") or iso_now()
            continue
        grouped[key] = {
            **event,
            "at": event.get("timestamp") or iso_now(),
            "occurrences": 1,
        }
    return list(grouped.values())


#: Process-wide, like the logging pipeline it hangs off.
store = LogStore()


def sink(_logger: Any, _name: str, event_dict: Any) -> Any:
    """structlog processor: a tap, returning its input untouched."""
    store.offer(dict(event_dict))
    return event_dict
