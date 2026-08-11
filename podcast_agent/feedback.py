"""Per-episode reader signals: starred, read, and "wrong call" (roadmap B1).

These are the inputs Theme C needs and cannot synthesise. The pipeline knows
what it *decided* about an episode; only the reader knows whether the decision
was right, and until that is recorded anywhere, a precision report or a few-shot
prompt has nothing to learn from.

Deliberately thin. No scoring, no inference, no automatic demotion of a show
because three of its episodes went unstarred — signals are recorded here and
interpreted later, by a person, in a report they can argue with. Guessing on
their behalf is how a digest quietly stops showing things.

Stored on the episode document rather than in a table of their own: they are
facts about that episode, they are read whenever it is, and they must survive
exactly as long as it does.
"""

from __future__ import annotations

from typing import Any, Literal

from .db import Doc, Store, update_doc
from .logging_setup import get_logger
from .sanitize import md_escape_inline
from .utils import iso_now

log = get_logger(__name__)

#: What the reader thought of the pipeline's decision.
#:
#: Two directions, because they mean opposite things and a single "wrong" flag
#: would conflate them: `over` is a false positive (summarised, not worth it),
#: `under` a false negative (dropped or downgraded, should not have been). The
#: second is the expensive kind — nobody sees what they were not shown — which
#: is precisely why it has to be reportable from a browsing view.
Verdict = Literal["over", "under"]

MAX_NOTE_CHARS = 500


def _feedback_block(verdict: Verdict | None, note: str | None) -> dict[str, Any] | None:
    if verdict is None:
        return None
    return {
        "verdict": verdict,
        # Free text from a browser, but it is rendered back into an HTML console
        # and may end up in a future prompt, so it is escaped on the way in.
        "note": md_escape_inline(note or "", max_chars=MAX_NOTE_CHARS) or None,
        "at": iso_now(),
    }


async def _update(store: Store, episode_id: str, apply: Any) -> Doc:
    """Apply ``apply`` and return the document that was written.

    `update_doc` returns CouchDB's put receipt — an id and a rev — which is the
    right answer for a caller that only needs to know the write landed, and the
    wrong one for these, whose callers render the new state straight back to the
    browser. Captured from the mutator rather than re-read: it is the same
    object that was stored, and on a conflict retry the last run is the one that
    won.
    """
    written: dict[str, Doc] = {}

    def _apply(doc: Doc) -> None:
        apply(doc)
        written["doc"] = doc

    await update_doc(store, episode_id, _apply)
    return written["doc"]


async def set_starred(store: Store, episode_id: str, starred: bool) -> Doc:
    def _apply(doc: Doc) -> None:
        doc["starred"] = starred
        doc["starred_at"] = iso_now() if starred else None

    doc = await _update(store, episode_id, _apply)
    log.info("feedback.starred", episode_id=episode_id, starred=starred)
    return doc


async def set_read(store: Store, episode_id: str, read: bool) -> Doc:
    def _apply(doc: Doc) -> None:
        # A timestamp rather than a boolean: "when did you read this" answers
        # time-to-read, which is the signal C1 wants, and "have you read this"
        # falls out of it for free.
        doc["read_at"] = iso_now() if read else None

    doc = await _update(store, episode_id, _apply)
    log.info("feedback.read", episode_id=episode_id, read=read)
    return doc


async def set_verdict(
    store: Store, episode_id: str, verdict: Verdict | None, note: str | None = None
) -> Doc:
    """Record — or clear — the reader's view of the pipeline's decision.

    The scored state at the time is captured alongside it. A verdict is only
    interpretable against what the pipeline actually decided, and a later
    re-score would otherwise silently rewrite the thing being judged.
    """

    def _apply(doc: Doc) -> None:
        block = _feedback_block(verdict, note)
        if block is not None:
            tier1 = doc.get("tier1") or {}
            tier0 = doc.get("tier0") or {}
            block["judged"] = {
                "status": doc.get("status"),
                "relevance_score": tier1.get("relevance_score"),
                "relevance_guess": tier0.get("relevance_guess"),
                "interest_profile_version": (
                    tier1.get("interest_profile_version") or tier0.get("interest_profile_version")
                ),
            }
        doc["feedback"] = block

    doc = await _update(store, episode_id, _apply)
    log.info("feedback.verdict", episode_id=episode_id, verdict=verdict)
    return doc


def view(doc: Doc) -> dict[str, Any]:
    """The signal fields, for the episode API."""
    feedback = doc.get("feedback") or None
    return {
        "starred": bool(doc.get("starred")),
        "starred_at": doc.get("starred_at"),
        "read_at": doc.get("read_at"),
        "feedback": feedback,
    }
