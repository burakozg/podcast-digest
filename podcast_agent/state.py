"""Episode status enum and the single source of truth for legal transitions (§6).

Every status change in the system goes through :func:`assert_transition`. An
illegal transition raises rather than silently corrupting pipeline state — a
crash here means a bug in a stage, not bad data.
"""

from __future__ import annotations

from enum import StrEnum


class EpisodeStatus(StrEnum):
    #: Ingested, not yet triaged.
    NEW = "NEW"
    #: Tier-0 result stored; the routing decision is recorded on the doc but the
    #: episode has not yet been dispatched to its next stage.
    TRIAGED = "TRIAGED"
    #: Queued for transcript acquisition (feed transcript / scrape / ASR).
    AWAITING_TRANSCRIPT = "AWAITING_TRANSCRIPT"
    #: Transcript text stored as an attachment.
    TRANSCRIBED = "TRANSCRIBED"
    #: All acquisition strategies exhausted; Tier-1 will run description-only.
    TRANSCRIPT_FAILED = "TRANSCRIPT_FAILED"
    #: Tier-1 result stored, not yet classified against digest_threshold.
    SUMMARIZED = "SUMMARIZED"
    #: Scored at or above digest_threshold — earns a full digest entry.
    READY_FOR_DIGEST = "READY_FOR_DIGEST"
    #: Grey-zone Tier-0 verdict: one-liner in the digest, never summarized.
    DIGEST_DIRECT = "DIGEST_DIRECT"
    #: Summarized but below digest_threshold; audit trail only.
    SCORED_LOW = "SCORED_LOW"
    #: Confidently irrelevant at Tier-0. Kept for audit.
    DROPPED = "DROPPED"
    #: Included in a generated digest.
    PUBLISHED = "PUBLISHED"
    #: Unexpected failure; traceback stored on the doc for inspection.
    ERROR = "ERROR"


#: The two values of `episode.origin`: routine polling, or the archive walk.
#: They live here, beside the status vocabulary, rather than in the backfill
#: package — both ingesters need them, and putting them under one of the two
#: made the other import it, which is a cycle.
BACKFILL_ORIGIN = "backfill"
ROUTINE_ORIGIN = "routine"

#: Mango clause selecting everything that is *not* archive material.
#:
#: An equality match, which Mango serves straight from an index. The obvious
#: spelling — `{"origin": {"$ne": BACKFILL_ORIGIN}}` — cannot be, and worse is
#: wrong: CouchDB holds no index entry for a document lacking the field, so every
#: comparison against a missing field fails, negative ones included. That matched
#: no routine episode at all until it was caught in production. The `$or` with
#: `$exists: false` that replaced it was correct but unindexable, so every
#: pipeline query scanned a range and filtered in memory. Writing `origin` on
#: every episode — see `migrate.backfill_origins` — buys the honest spelling.
ROUTINE_ONLY: dict[str, object] = {"origin": ROUTINE_ORIGIN}


S = EpisodeStatus

#: Statuses whose episodes still need pipeline work.
ACTIVE_STATUSES: frozenset[EpisodeStatus] = frozenset(
    {S.NEW, S.TRIAGED, S.AWAITING_TRANSCRIPT, S.TRANSCRIBED, S.TRANSCRIPT_FAILED, S.SUMMARIZED}
)

#: Statuses eligible for a full digest entry.
DIGESTABLE_STATUSES: frozenset[EpisodeStatus] = frozenset({S.READY_FOR_DIGEST, S.DIGEST_DIRECT})

#: Statuses that appear only in the digest's audit table.
AUDIT_ONLY_STATUSES: frozenset[EpisodeStatus] = frozenset(
    {S.SCORED_LOW, S.DROPPED, S.TRANSCRIPT_FAILED, S.ERROR}
)

#: Allowed transitions. Read as: from -> set of permitted next statuses.
ALLOWED_TRANSITIONS: dict[EpisodeStatus, frozenset[EpisodeStatus]] = {
    S.NEW: frozenset({S.TRIAGED, S.ERROR}),
    # Dispatch of the stored Tier-0 route.
    S.TRIAGED: frozenset({S.AWAITING_TRANSCRIPT, S.DIGEST_DIRECT, S.DROPPED, S.ERROR}),
    # Self-transition covers a queued retry that made no forward progress.
    S.AWAITING_TRANSCRIPT: frozenset(
        {S.AWAITING_TRANSCRIPT, S.TRANSCRIBED, S.TRANSCRIPT_FAILED, S.ERROR}
    ),
    S.TRANSCRIBED: frozenset({S.SUMMARIZED, S.ERROR}),
    # Description-only Tier-1 fallback.
    S.TRANSCRIPT_FAILED: frozenset({S.SUMMARIZED, S.AWAITING_TRANSCRIPT, S.ERROR}),
    S.SUMMARIZED: frozenset({S.READY_FOR_DIGEST, S.SCORED_LOW, S.ERROR}),
    # SCORED_LOW here is re-classification after a re-score against a changed
    # interest profile (C2). AWAITING_TRANSCRIPT is a manual re-summarisation:
    # the common case is an episode summarised description-only that the owner
    # now wants transcribed properly. Neither is reachable from normal flow.
    S.READY_FOR_DIGEST: frozenset({S.PUBLISHED, S.SCORED_LOW, S.AWAITING_TRANSCRIPT, S.ERROR}),
    S.DIGEST_DIRECT: frozenset({S.PUBLISHED, S.AWAITING_TRANSCRIPT, S.ERROR}),
    # AWAITING_TRANSCRIPT is the owner override via /episodes/{id}/escalate;
    # READY_FOR_DIGEST is a re-score promoting the episode (C2).
    S.SCORED_LOW: frozenset({S.AWAITING_TRANSCRIPT, S.READY_FOR_DIGEST, S.PUBLISHED, S.ERROR}),
    S.DROPPED: frozenset({S.AWAITING_TRANSCRIPT}),
    # Terminal to the pipeline: nothing automatic moves a published episode.
    # The single exception is the owner override at /episodes/{id}/escalate,
    # which is the opposite of silent — it is an explicit request, it records
    # `forced_escalation` on the document, and it clears the digest claim so the
    # episode can be summarised properly and claimed again.
    #
    # The file already written keeps what it said. Digests and archive files are
    # never overwritten, so a re-summarised episode appears in a new file rather
    # than rewriting history — which is what made this safe to allow.
    S.PUBLISHED: frozenset({S.AWAITING_TRANSCRIPT}),
    # Recovery via /episodes/{id}/retry.
    S.ERROR: frozenset({S.NEW, S.AWAITING_TRANSCRIPT, S.TRANSCRIBED, S.TRANSCRIPT_FAILED}),
}


class IllegalTransition(Exception):
    """Raised when a stage attempts a status change the state machine forbids."""

    def __init__(
        self, current: EpisodeStatus, requested: EpisodeStatus, episode_id: str | None = None
    ):
        self.current = current
        self.requested = requested
        self.episode_id = episode_id
        where = f" for {episode_id}" if episode_id else ""
        allowed = sorted(ALLOWED_TRANSITIONS.get(current, frozenset()))
        super().__init__(
            f"illegal transition{where}: {current} -> {requested} "
            f"(allowed from {current}: {', '.join(allowed) or 'none — terminal'})"
        )


def can_transition(current: EpisodeStatus, requested: EpisodeStatus) -> bool:
    return requested in ALLOWED_TRANSITIONS.get(current, frozenset())


def assert_transition(
    current: EpisodeStatus | str,
    requested: EpisodeStatus | str,
    episode_id: str | None = None,
) -> EpisodeStatus:
    """Validate a transition and return the new status.

    Accepts raw strings (as read from CouchDB) and normalises them.
    """
    try:
        cur = EpisodeStatus(current)
        req = EpisodeStatus(requested)
    except ValueError as exc:
        raise ValueError(
            f"unknown episode status in transition {current!r} -> {requested!r}: {exc}"
        ) from exc
    if not can_transition(cur, req):
        raise IllegalTransition(cur, req, episode_id)
    return req


def retry_target(current: EpisodeStatus) -> EpisodeStatus:
    """Where a failed episode should be reset to by /episodes/{id}/retry.

    Returns the last known-good state, based on how far the episode had got.
    """
    match current:
        case S.ERROR | S.TRANSCRIPT_FAILED:
            return S.AWAITING_TRANSCRIPT
        case S.AWAITING_TRANSCRIPT:
            return S.AWAITING_TRANSCRIPT
        case _:
            return current
