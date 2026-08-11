"""A bounded in-memory tail of recent log events, for the console.

Logs go to stdout, which is right for a service and useless for someone looking
at a browser on the other side of the house. This keeps the last few thousand
events in memory so the console can show them.

**In memory, deliberately.** Writing every log line to CouchDB would mean the
act of logging generates documents that generate more logging, and a retention
job to clean up after it — a lot of machinery for something whose value is
almost entirely in the last few minutes. The durable record already exists
elsewhere and is what the console shows alongside this: `llm_call` documents for
every model invocation, and `run` documents for every job. What is lost on
restart is the debug chatter, which is the part nobody needs after the fact.

Events arrive here already redacted: the sink is installed after the redaction
processor, so a secret cannot reach the buffer even though the buffer is served
over the API.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

#: Roughly an hour of a busy pipeline, and about a megabyte of memory.
DEFAULT_CAPACITY = 2000

#: Keys carrying rendered noise rather than information. `event` and everything
#: else is kept, because a log line's value is in its bespoke fields.
_DROP_KEYS = frozenset({"_record", "_from_structlog", "stack", "positional_args"})

#: One field cannot be allowed to blow the buffer's memory budget; a traceback
#: is the usual culprit.
MAX_VALUE_CHARS = 2000


class LogBuffer:
    """Thread-safe ring of recent events.

    Logging happens on the event loop, on APScheduler's worker threads and on
    uvicorn's, so both ends take a lock. `deque.append` is atomic under the GIL,
    but building a filtered snapshot is not.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0

    @property
    def capacity(self) -> int:
        return self._events.maxlen or 0

    def add(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            self._events.append({"seq": self._seq, **event})

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        return len(self._events)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def tail(
        self,
        *,
        limit: int = 200,
        level: str | None = None,
        contains: str | None = None,
        logger: str | None = None,
        since_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        """Most recent events first, oldest-truncated to ``limit``.

        ``level`` filters at *or above* the given severity, which is what a
        reader means by "show me warnings" — not "hide the errors too".
        """
        floor = _LEVEL_ORDER.get((level or "").lower(), 0)
        needle = (contains or "").lower()

        selected: list[dict[str, Any]] = []
        for event in reversed(self.snapshot()):
            if since_seq is not None and event.get("seq", 0) <= since_seq:
                break
            if floor and _LEVEL_ORDER.get(str(event.get("level", "")).lower(), 0) < floor:
                continue
            if logger and logger not in str(event.get("logger", "")):
                continue
            if needle and needle not in _haystack(event):
                continue
            selected.append(event)
            if len(selected) >= limit:
                break
        return selected

    def levels(self) -> dict[str, int]:
        """How many of each severity are held, for the console's filter chips."""
        counts: dict[str, int] = {}
        for event in self.snapshot():
            key = str(event.get("level") or "info")
            counts[key] = counts.get(key, 0) + 1
        return counts


#: Ordering for "at or above this level". Unknown levels sort lowest so a filter
#: never silently swallows something it does not recognise.
_LEVEL_ORDER = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "warn": 30,
    "error": 40,
    "critical": 50,
    "exception": 40,
}


def _haystack(event: dict[str, Any]) -> str:
    return " ".join(str(v) for k, v in event.items() if k != "seq").lower()


def _trim(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
        return value[:MAX_VALUE_CHARS] + "…"
    return value


#: The process-wide buffer. A module-level instance rather than something passed
#: around, because the logging pipeline is itself process-wide and the sink has
#: to be installable from `configure_logging`, which has no application context.
buffer = LogBuffer()


def sink(_logger: Any, _name: str, event_dict: Any) -> Any:
    """structlog processor that copies each event into the buffer.

    Returns its input untouched: this is a tap, not a transform, and it must not
    change what lands on stdout. Installed *after* redaction, so what is kept is
    what was already safe to print.
    """
    # Silent by necessity: reporting a failure here would log, which re-enters
    # this processor, which fails again. A dropped console line is worth far
    # less than a working log pipeline, so the tap fails closed and quietly.
    with suppress(Exception):  # pragma: no cover - a broken tap must not break logging
        buffer.add(
            {key: _trim(value) for key, value in event_dict.items() if key not in _DROP_KEYS}
        )
    return event_dict


def iter_events() -> Iterator[dict[str, Any]]:  # pragma: no cover - convenience
    yield from buffer.snapshot()
