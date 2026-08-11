"""How often a podcast publishes, inferred from the episodes we hold.

Feeds do declare a cadence — ``sy:updatePeriod``, ``itunes:type`` — but rarely,
and when they do it is frequently the value the show had when the feed was first
set up. Measuring the gaps between real episodes is self-correcting: a weekly
show that goes fortnightly says so within a couple of months, and a podcast that
stops publishing drifts to "occasional" on its own.

The measure is the **median** gap over recent episodes, not the mean. A single
two-month hiatus, or three bonus episodes dropped on one day, would drag a mean
badly; the median ignores both and keeps reporting the normal rhythm.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise

from .utils import parse_iso

#: Gaps are measured over this many recent episodes. Enough to be robust to one
#: irregular week, short enough that a cadence change shows up in a month or two
#: rather than being averaged away against years of history.
WINDOW = 12

#: Two dated episodes give one gap, which is an anecdote. Three give two gaps and
#: a median that at least survives a single outlier.
MIN_EPISODES = 3

#: (upper bound in days, label). First bucket whose bound the median falls under
#: wins. Bounds sit between the natural rhythms rather than on them, so a show
#: that publishes every 6.8 days still reads as weekly.
_BUCKETS: tuple[tuple[float, str], ...] = (
    (1.4, "~daily"),
    (2.5, "~3x/week"),
    (4.5, "~2x/week"),
    (9.0, "~weekly"),
    (20.0, "~fortnightly"),
    (45.0, "~monthly"),
    (100.0, "~quarterly"),
)

#: Beyond the last bucket there is no rhythm worth naming.
_IRREGULAR = "occasional"


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def describe_cadence(published_at: list[str | None]) -> tuple[str | None, str | None]:
    """Return ``(label, detail)`` from stored ISO publication timestamps.

    Undated and unparseable entries are dropped rather than guessed at.
    """
    dates = [d for d in (parse_iso(v) for v in published_at) if d is not None]
    return cadence_from_dates(dates)


def cadence_from_dates(dates: list[datetime]) -> tuple[str | None, str | None]:
    """Return ``(label, detail)`` for a podcast's publication rhythm.

    ``dates`` is every publication time known for the podcast, in any order.
    Returns ``(None, None)`` when there is not enough history to say anything
    honest, which the console renders as "not enough history" rather than
    inventing a number.

    ``detail`` states the measurement behind the label, so a surprising value is
    checkable instead of merely disagreeable.
    """
    if len(dates) < MIN_EPISODES:
        return None, None

    # Newest first, so the window measures the current rhythm rather than the
    # rhythm this podcast had when it launched. Sorted into a new list: the
    # caller's ordering is not ours to change.
    #
    # Feeds mix offset-aware and naive timestamps, and comparing the two raises.
    # Naive values are read as UTC, matching how the rest of the pipeline stores
    # them.
    normalised = [d if d.tzinfo else d.replace(tzinfo=UTC) for d in dates]
    recent = sorted(normalised, reverse=True)[:WINDOW]

    gaps = [(earlier - later).total_seconds() / 86_400.0 for earlier, later in pairwise(recent)]
    # Simultaneous publication (a season dropped at once) yields zero gaps, which
    # would claim a cadence far tighter than the podcast really has.
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return None, None

    median = _median(gaps)
    label = next((name for bound, name in _BUCKETS if median < bound), _IRREGULAR)
    detail = f"median {median:.1f} days between the last {len(gaps) + 1} episodes held"
    return label, detail
