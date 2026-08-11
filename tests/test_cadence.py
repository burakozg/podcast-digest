"""Publication-rhythm inference.

The console states a cadence as fact, so the failure that matters is not a
missing label but a confident wrong one — a podcast that dropped a whole season
at once reading as "~daily", or one hiatus turning a weekly podcast into a
monthly one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from podcast_agent.cadence import MIN_EPISODES, cadence_from_dates, describe_cadence

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


def every(days: float, count: int = 12) -> list[datetime]:
    return [NOW - timedelta(days=days * i) for i in range(count)]


@pytest.mark.parametrize(
    ("gap_days", "expected"),
    [
        (1.0, "~daily"),
        (2.0, "~3x/week"),
        (3.5, "~2x/week"),
        (7.0, "~weekly"),
        (14.0, "~fortnightly"),
        (30.0, "~monthly"),
        (91.0, "~quarterly"),
        (365.0, "occasional"),
    ],
)
def test_regular_rhythms_are_named(gap_days: float, expected: str) -> None:
    label, detail = cadence_from_dates(every(gap_days))
    assert label == expected
    assert detail is not None and "median" in detail


def test_a_weekly_podcast_that_slips_a_day_still_reads_as_weekly() -> None:
    """Bounds sit between the rhythms, not on them; feeds are not metronomes."""
    for gap in (6.0, 6.8, 7.4, 8.5):
        assert cadence_from_dates(every(gap))[0] == "~weekly"


def test_one_hiatus_does_not_change_the_reported_rhythm() -> None:
    """The reason the measure is a median: a mean would report ~fortnightly."""
    dates = every(7.0, count=11)
    dates.append(dates[-1] - timedelta(days=120))
    assert cadence_from_dates(dates)[0] == "~weekly"


def test_a_burst_of_bonus_episodes_does_not_claim_a_daily_podcast() -> None:
    weekly = every(7.0, count=9)
    burst = [NOW - timedelta(hours=h) for h in (1, 2, 3)]
    assert cadence_from_dates(weekly + burst)[0] == "~weekly"


def test_the_recent_window_wins_over_ancient_history() -> None:
    """A podcast that was daily for years and is now weekly is now weekly."""
    recent_weekly = every(7.0, count=12)
    old_daily = [recent_weekly[-1] - timedelta(days=i) for i in range(1, 200)]
    assert cadence_from_dates(recent_weekly + old_daily)[0] == "~weekly"


def test_too_little_history_says_nothing_rather_than_guessing() -> None:
    for count in range(MIN_EPISODES):
        assert cadence_from_dates(every(7.0, count=count)) == (None, None)


def test_a_season_dropped_at_once_is_not_a_cadence() -> None:
    """Identical timestamps give zero-length gaps, which would claim ~daily."""
    assert cadence_from_dates([NOW] * 10) == (None, None)


def test_naive_and_aware_timestamps_do_not_raise() -> None:
    """Feeds mix the two, and subtracting one from the other is a TypeError."""
    mixed = [NOW, (NOW - timedelta(days=7)).replace(tzinfo=None), NOW - timedelta(days=14)]
    assert cadence_from_dates(mixed)[0] == "~weekly"


def test_the_callers_list_is_not_reordered() -> None:
    dates = every(7.0, count=5)
    before = list(dates)
    cadence_from_dates(dates)
    assert dates == before


class TestFromStoredStrings:
    def test_iso_timestamps_are_read(self) -> None:
        assert describe_cadence([d.isoformat() for d in every(7.0)])[0] == "~weekly"

    def test_undated_and_unparseable_episodes_are_dropped_not_guessed(self) -> None:
        stored: list[str | None] = [d.isoformat() for d in every(7.0, count=4)]
        stored += [None, "", "not a date", "0000-00-00"]
        assert describe_cadence(stored)[0] == "~weekly"

    def test_all_undated_says_nothing(self) -> None:
        assert describe_cadence([None, None, None, ""]) == (None, None)
