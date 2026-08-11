"""State machine tests (§10.5): the transition map is the pipeline's safety net."""

from __future__ import annotations

import itertools

import pytest

from podcast_agent.state import (
    ALLOWED_TRANSITIONS,
    EpisodeStatus,
    IllegalTransition,
    assert_transition,
    can_transition,
    retry_target,
)

S = EpisodeStatus


def test_every_status_has_a_transition_entry() -> None:
    """A status missing from the map would raise on any transition out of it."""
    assert set(ALLOWED_TRANSITIONS) == set(EpisodeStatus)


def test_happy_path_full_pipeline() -> None:
    path = [
        S.NEW,
        S.TRIAGED,
        S.AWAITING_TRANSCRIPT,
        S.TRANSCRIBED,
        S.SUMMARIZED,
        S.READY_FOR_DIGEST,
        S.PUBLISHED,
    ]
    for current, nxt in itertools.pairwise(path):
        assert can_transition(current, nxt), f"{current} -> {nxt} should be legal"


def test_description_only_path() -> None:
    """Transcript failure is not a dead end — Tier-1 still runs (§4)."""
    assert can_transition(S.AWAITING_TRANSCRIPT, S.TRANSCRIPT_FAILED)
    assert can_transition(S.TRANSCRIPT_FAILED, S.SUMMARIZED)
    assert can_transition(S.SUMMARIZED, S.SCORED_LOW)


def test_digest_direct_path() -> None:
    assert can_transition(S.TRIAGED, S.DIGEST_DIRECT)
    assert can_transition(S.DIGEST_DIRECT, S.PUBLISHED)


def test_published_moves_only_by_explicit_owner_override() -> None:
    """Terminal to the pipeline; re-openable by a person.

    Nothing automatic moves a published episode — the guard exists so it cannot
    be re-published silently. An explicit escalate is not silent: it records
    `forced_escalation` and clears the digest claim, and the file already
    written is never rewritten, so history stands.
    """
    assert ALLOWED_TRANSITIONS[S.PUBLISHED] == frozenset({S.AWAITING_TRANSCRIPT})
    assert can_transition(S.PUBLISHED, S.AWAITING_TRANSCRIPT)


@pytest.mark.parametrize(
    "target", [S.NEW, S.TRIAGED, S.TRANSCRIBED, S.SUMMARIZED, S.READY_FOR_DIGEST, S.DROPPED]
)
def test_published_goes_nowhere_else(target: S) -> None:
    """Only the one door, so a stage cannot resurrect a published episode."""
    with pytest.raises(IllegalTransition) as excinfo:
        assert_transition(S.PUBLISHED, target, "episode:abc")
    assert "episode:abc" in str(excinfo.value)


def test_dropped_can_only_be_escalated() -> None:
    """Owner override is the single way back from DROPPED (§9)."""
    assert ALLOWED_TRANSITIONS[S.DROPPED] == frozenset({S.AWAITING_TRANSCRIPT})


def test_cannot_skip_triage() -> None:
    with pytest.raises(IllegalTransition):
        assert_transition(S.NEW, S.SUMMARIZED)
    with pytest.raises(IllegalTransition):
        assert_transition(S.NEW, S.PUBLISHED)


def test_accepts_raw_strings_from_couchdb() -> None:
    assert assert_transition("NEW", "TRIAGED") is S.TRIAGED


def test_unknown_status_is_a_value_error_not_illegal_transition() -> None:
    with pytest.raises(ValueError, match="unknown episode status"):
        assert_transition("NEW", "BANANA")


def test_awaiting_transcript_self_transition_is_allowed() -> None:
    """A retry that made no progress still needs its attempt counter written."""
    assert can_transition(S.AWAITING_TRANSCRIPT, S.AWAITING_TRANSCRIPT)


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (S.ERROR, S.AWAITING_TRANSCRIPT),
        (S.TRANSCRIPT_FAILED, S.AWAITING_TRANSCRIPT),
        (S.AWAITING_TRANSCRIPT, S.AWAITING_TRANSCRIPT),
        (S.PUBLISHED, S.PUBLISHED),
    ],
)
def test_retry_target(current: EpisodeStatus, expected: EpisodeStatus) -> None:
    assert retry_target(current) is expected


def test_error_is_reachable_from_every_active_status() -> None:
    """A poison-pill episode must always be markable ERROR (§10.3)."""
    for status in (
        S.NEW,
        S.TRIAGED,
        S.AWAITING_TRANSCRIPT,
        S.TRANSCRIBED,
        S.TRANSCRIPT_FAILED,
        S.SUMMARIZED,
    ):
        assert can_transition(status, S.ERROR), f"{status} must reach ERROR"
