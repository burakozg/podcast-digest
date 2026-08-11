"""Run-state control for archive backfill (roadmap B2 support).

Backfill is the one job here that runs for hours, so it needs a switch that is
neither "kill the process" nor "wait for it to finish". This keeps a persisted
paused flag and a cooperative stop check: a pause takes effect at the next
episode boundary, so the in-flight episode completes rather than being wasted,
and nothing is left half-written.

The flag defaults to **paused**. A scheduled archive walk should never begin
because someone deployed a config file; starting it is an explicit act, matching
the ``confirm=true`` guard on manual runs.
"""

from __future__ import annotations

from typing import Any

from ..db import Doc, Store
from ..logging_setup import get_logger
from ..utils import iso_now

log = get_logger(__name__)

CONTROL_DOC_ID = "control:backfill"


async def get_state(store: Store) -> dict[str, Any]:
    """Current control state, with defaults when nothing has been set yet."""
    doc = await store.get(CONTROL_DOC_ID)
    if doc is None:
        return {"paused": True, "updated_at": None, "note": "never started"}
    return {
        "paused": bool(doc.get("paused", True)),
        "updated_at": doc.get("updated_at"),
        "note": doc.get("note") or "",
    }


async def _merge(store: Store, changes: dict[str, Any]) -> dict[str, Any]:
    """Write ``changes`` onto the control document, preserving the rest.

    Replacing the document wholesale is how pausing the walk would silently
    reset the window someone had chosen.
    """
    existing = await store.get(CONTROL_DOC_ID) or {}
    doc: Doc = {
        **{k: v for k, v in existing.items() if not k.startswith("_")},
        "_id": CONTROL_DOC_ID,
        "type": "control",
        "key": "backfill",
        **changes,
        "updated_at": iso_now(),
    }
    doc.setdefault("paused", True)
    if existing:
        doc["_rev"] = existing["_rev"]
    await store.put(doc)
    return await get_state(store)


async def is_paused(store: Store) -> bool:
    return bool((await get_state(store))["paused"])


async def set_paused(store: Store, paused: bool, *, note: str = "") -> dict[str, Any]:
    """Pause or resume the archive walk. Idempotent."""
    state = await _merge(store, {"paused": paused, "note": note})
    log.info("backfill.control_changed", paused=paused, note=note)
    return state


class PauseCheck:
    """Asks the store whether backfill has been paused, with a short cache.

    Called between episodes, so it must be cheap; a stale answer for a couple of
    seconds is harmless, since the point is to stop within moments rather than
    instantly.
    """

    def __init__(self, store: Store, *, cache_seconds: float = 3.0) -> None:
        self._store = store
        self._cache_seconds = cache_seconds
        self._last_checked: float = 0.0
        self._last_value = False

    async def should_stop(self) -> bool:
        import time

        now = time.monotonic()
        if now - self._last_checked < self._cache_seconds:
            return self._last_value
        try:
            self._last_value = await is_paused(self._store)
        except Exception as exc:
            # A storage blip must not silently stop a long run.
            log.warning("backfill.pause_check_failed", error=str(exc))
            self._last_value = False
        self._last_checked = now
        return self._last_value


async def rewind_cursors(store: Store, *, slug: str | None = None) -> dict[str, Any]:
    """Clear archive cursors so the window is walked again.

    The walk only ever moves backwards, so a month it has already passed is
    unreachable — including months it passed under a policy that discarded most
    of what it saw. Widening the window does not help: that extends the far end,
    while the missing episodes are in months already behind the cursor.

    Deletes nothing. Ingestion is keyed by ``sha256(slug + guid)`` and
    create-if-absent, so a second pass over the same month re-reads the feed and
    adds only what was not recorded before; existing episodes keep their status,
    their summaries and their digest claims.
    """
    selector: dict[str, Any] = {"type": "podcast"}
    if slug:
        selector["slug"] = slug
    docs = await store.find(selector, limit=1_000)

    rewound: list[str] = []
    for doc in docs:
        if doc.get("backfill_cursor") is None and not doc.get("backfill_complete"):
            continue
        doc["backfill_cursor"] = None
        doc["backfill_complete"] = False
        doc["backfill_updated_at"] = iso_now()
        await store.put(doc)
        rewound.append(str(doc.get("slug")))

    log.info("backfill.cursors_rewound", podcasts=len(rewound))
    return {"rewound": sorted(rewound), "count": len(rewound)}
