"""Episode document helpers: guarded status transitions and error recording.

Every status change goes through :func:`transition`, which validates against the
state machine (§6) and retries on MVCC conflict.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from .db import Doc, Store, update_doc
from .logging_setup import get_logger
from .state import EpisodeStatus, assert_transition
from .utils import iso_now

log = get_logger(__name__)

#: Truncation for stored tracebacks — enough to diagnose, not enough to bloat.
MAX_TRACEBACK_CHARS = 4000


async def transition(
    store: Store,
    episode_id: str,
    new_status: EpisodeStatus,
    *,
    mutate: Callable[[Doc], None] | None = None,
) -> Doc:
    """Move an episode to ``new_status``, applying ``mutate`` in the same write.

    Doing the payload update and the status change atomically means a crash can
    never leave a doc marked SUMMARIZED with no summary on it.
    """

    def _apply(doc: Doc) -> None:
        current = doc.get("status", EpisodeStatus.NEW.value)
        if current == new_status.value and new_status not in _SELF_TRANSITION_OK:
            # Idempotent replay of an already-applied step: apply payload only.
            if mutate:
                mutate(doc)
            doc["updated_at"] = iso_now()
            return
        assert_transition(current, new_status, episode_id)
        if mutate:
            mutate(doc)
        doc["status"] = new_status.value
        doc["updated_at"] = iso_now()

    await update_doc(store, episode_id, _apply)
    refreshed = await store.get(episode_id)
    assert refreshed is not None  # update_doc would have raised
    return refreshed


#: Statuses a stage may legitimately re-enter (a queued retry that made no
#: forward progress still needs its attempt counter written).
_SELF_TRANSITION_OK = frozenset({EpisodeStatus.AWAITING_TRANSCRIPT})


async def mark_error(store: Store, episode_id: str, stage: str, exc: BaseException) -> None:
    """Record a per-episode failure without stopping the run (§10.3 poison pill)."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    def _apply(doc: Doc) -> None:
        current = doc.get("status", EpisodeStatus.NEW.value)
        doc["last_error"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
            "traceback": tb[-MAX_TRACEBACK_CHARS:],
            "at": iso_now(),
            "status_when_failed": current,
        }
        doc["updated_at"] = iso_now()
        # ERROR is reachable from every non-terminal status; if it is not (e.g.
        # PUBLISHED), keep the status and just record the error.
        try:
            assert_transition(current, EpisodeStatus.ERROR, episode_id)
            doc["status"] = EpisodeStatus.ERROR.value
        except Exception:
            log.warning(
                "episode.error_status_not_applicable", episode_id=episode_id, status=current
            )

    try:
        await update_doc(store, episode_id, _apply)
    except Exception as write_exc:
        log.error(
            "episode.error_record_failed",
            episode_id=episode_id,
            stage=stage,
            error=str(write_exc),
        )


def bump_attempt(doc: Doc, key: str) -> int:
    """Increment and return an attempt counter on the doc (in place)."""
    attempts = doc.setdefault("attempts", {})
    attempts[key] = int(attempts.get(key) or 0) + 1
    return int(attempts[key])


def attempt_count(doc: Doc, key: str) -> int:
    return int((doc.get("attempts") or {}).get(key) or 0)


def filter_interest_keys(claimed: list[str] | None, valid: set[str]) -> list[str]:
    """Keep only interest keys that exist in the profile.

    The model is asked for exact keys but may invent or mangle them; an unknown
    key is dropped rather than retried, since it never affects routing.
    """
    if not claimed:
        return []
    seen: set[str] = set()
    kept: list[str] = []
    for key in claimed:
        normalized = str(key).strip().lower()
        if normalized in valid and normalized not in seen:
            seen.add(normalized)
            kept.append(normalized)
    return kept


def call_meta_dict(meta: Any) -> dict[str, Any]:
    """Flatten CallMeta for embedding in the episode's tier block (§6)."""
    return {
        "model": meta.model,
        "provider": meta.provider,
        "latency_ms": meta.latency_ms,
        "cost_usd": meta.cost_usd,
        "input_tokens": meta.input_tokens,
        "output_tokens": meta.output_tokens,
        "fallback_used": meta.fallback_used,
        "validation_retries": meta.validation_retries,
        "prompt_version": meta.prompt_version,
    }
