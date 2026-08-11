"""Precision report over the reader's own signals (roadmap C1, phase 1).

The pipeline knows what it decided. Only the reader knows whether the decision
was right, and B1 is where that gets recorded. This turns those signals into the
question worth asking every month or so: *is the interest profile still describing
what you actually want?*

Three deliberate limits, all of them the point rather than a shortcut.

**Nothing is applied.** Every finding is a suggestion with the numbers behind it
and the exact configuration change, for a person to accept or reject. A system
that demotes a show because three of its episodes went unstarred is a system
that quietly stops showing you things, and the failure is invisible: you never
see what you were no longer shown.

**Nothing is suggested below a sample threshold.** "You starred 0 of 2" is not
evidence of anything, and a report that says it anyway trains its reader to
ignore it.

**Absence of a star is weak evidence.** Not starring is what happens when you
read something useful and move on, so an unstarred episode means very little on
its own. A *flag* is deliberate, so it counts for much more. The report says so
rather than pretending its proxies are measurements.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .config import Settings
from .db import Doc, Store, typed_sort
from .logging_setup import get_logger
from .podcasts import PodcastRegistry
from .state import EpisodeStatus
from .utils import iso, utcnow

log = get_logger(__name__)

BATCH = 500

#: Surfaced episodes before a show or interest is worth an opinion about.
#: Below this, a run of zero stars is ordinary noise.
MIN_SAMPLE = 8

#: Starred share at or below which a suggestion is offered, given the sample.
LOW_PRECISION = 0.1

#: Statuses meaning "this was put in front of you". Everything else was either
#: never judged or judged and set aside, and neither is something you declined
#: to star — you were never shown it.
SURFACED = frozenset({EpisodeStatus.READY_FOR_DIGEST.value, EpisodeStatus.PUBLISHED.value})

#: What a show can be demoted to, in order.
_DEMOTE = {"high": "med", "med": "low"}
_PROMOTE = {"low": "med", "med": "high"}


@dataclass(slots=True)
class Tally:
    label: str
    surfaced: int = 0
    starred: int = 0
    read: int = 0
    flagged_over: int = 0
    flagged_under: int = 0
    #: Populated for shows only, so a suggestion can name the current value.
    priority: str | None = None

    @property
    def precision(self) -> float | None:
        """Starred share of what was surfaced. None below the sample floor.

        A proxy, not a measurement — see the module docstring.
        """
        if self.surfaced < MIN_SAMPLE:
            return None
        return round(self.starred / self.surfaced, 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "surfaced": self.surfaced,
            "starred": self.starred,
            "read": self.read,
            "flagged_over": self.flagged_over,
            "flagged_under": self.flagged_under,
            "precision": self.precision,
            "enough_to_judge": self.surfaced >= MIN_SAMPLE,
            **({"priority": self.priority} if self.priority else {}),
        }


def _suggestion(
    kind: str, target: str, why: str, change: str, *, confidence: str
) -> dict[str, str]:
    return {
        "kind": kind,
        "target": target,
        "why": why,
        "change": change,
        # Stated rather than implied. A reader who cannot tell a strong finding
        # from a weak one ends up trusting neither.
        "confidence": confidence,
    }


def _show_suggestions(tally: Tally) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    priority = tally.priority or "med"

    if tally.flagged_under:
        out.append(
            _suggestion(
                "promote_show",
                tally.label,
                f"you marked {tally.flagged_under} episode(s) as ranked too low",
                (
                    f"priority: {_PROMOTE.get(priority, priority)}"
                    if priority in _PROMOTE
                    else "already at the highest priority — consider always_escalate: true"
                ),
                # A flag is a deliberate act, unlike the absence of a star.
                confidence="strong — you said so explicitly",
            )
        )

    if tally.surfaced < MIN_SAMPLE:
        return out

    if tally.flagged_over >= 2:
        out.append(
            _suggestion(
                "demote_show",
                tally.label,
                (
                    f"you marked {tally.flagged_over} of {tally.surfaced} surfaced "
                    "episodes as not worth the space"
                ),
                (
                    f"priority: {_DEMOTE[priority]}"
                    if priority in _DEMOTE
                    else "already at the lowest priority — consider enabled: false"
                ),
                confidence="strong — you said so explicitly",
            )
        )
    elif tally.starred == 0:
        out.append(
            _suggestion(
                "review_show",
                tally.label,
                f"you starred none of {tally.surfaced} surfaced episodes",
                (
                    f"priority: {_DEMOTE[priority]}"
                    if priority in _DEMOTE
                    else "already at the lowest priority — consider enabled: false"
                ),
                # Not starring is also what happens when you read something
                # useful and move on.
                confidence="weak — absence of a star is not a complaint",
            )
        )
    return out


def _interest_suggestions(tally: Tally, weight: int) -> list[dict[str, str]]:
    if tally.surfaced < MIN_SAMPLE:
        return []
    if tally.starred == 0 and weight > 1:
        return [
            _suggestion(
                "lower_weight",
                tally.label,
                f"you starred none of {tally.surfaced} episodes matched to this interest",
                f"weight: {max(1, weight - 2)}  (currently {weight})",
                confidence="weak — absence of a star is not a complaint",
            )
        ]
    precision = tally.precision or 0.0
    if precision >= 0.5 and weight < 10:
        return [
            _suggestion(
                "raise_weight",
                tally.label,
                (
                    f"you starred {tally.starred} of {tally.surfaced} episodes "
                    "matched to this interest"
                ),
                f"weight: {min(10, weight + 1)}  (currently {weight})",
                confidence="moderate — a consistent pattern, not a stated one",
            )
        ]
    return []


async def precision_report(store: Store, settings: Settings, *, days: int = 90) -> dict[str, Any]:
    """Per-show and per-interest signal tallies, with suggestions to apply by hand."""
    since = iso(utcnow() - timedelta(days=days))
    # Through the registry, not `settings`: the file is only half the list. A
    # podcast added in the console lives in the database, so reading config here
    # left five shows listed by slug with no priority — and a suggestion cannot
    # name the next priority down for a show whose current one it cannot see.
    # Disabled shows are included, because their past episodes still carry the
    # marks this report is counting.
    registry = PodcastRegistry(settings)
    await registry.refresh(store)
    records = registry.all_podcasts()
    priorities = {p.slug: p.priority.value for p in records}
    names = {p.slug: p.name for p in records}
    weights = {i.key: i.weight for i in settings.interest_profile}
    labels = {i.key: i.label for i in settings.interest_profile}

    shows: dict[str, Tally] = {}
    interests: dict[str, Tally] = {}
    totals = Tally(label="all")
    per_month: dict[str, dict[str, int]] = defaultdict(lambda: {"surfaced": 0, "starred": 0})

    # Reader signals that fall outside the window. Counted, because the report
    # is otherwise silent about them: four starred episodes and a report showing
    # one is indistinguishable from a report that is broken, and the reader has
    # no way to learn that the window is what hid the rest.
    outside = Tally(label="outside the window")

    skip = 0
    while True:
        page = await store.find(
            {"type": "episode", "published_at": {"$gte": since}},
            sort=typed_sort("published_at", "desc"),
            limit=BATCH,
            skip=skip,
        )
        if not page:
            break
        for episode in page:
            if episode.get("status") not in SURFACED:
                continue
            _accumulate(episode, shows, interests, totals, per_month, priorities, names, labels)
        skip += len(page)
        if len(page) < BATCH:
            break

    # A second pass for the marks the window excludes. Only the two signals a
    # person deliberately makes: an unstarred episode outside the window says
    # nothing, and counting those would just restate the size of the archive.
    for signal, selector in (
        ("starred", {"type": "episode", "starred": True}),
        ("flagged", {"type": "episode", "feedback": {"$exists": True}}),
    ):
        for episode in await store.find(selector, limit=BATCH):
            if (episode.get("published_at") or "") >= since:
                continue
            if episode.get("status") not in SURFACED:
                continue
            if signal == "starred":
                outside.starred += 1
            else:
                outside.flagged_over += 1

    suggestions: list[dict[str, str]] = []
    for tally in sorted(shows.values(), key=lambda t: -t.surfaced):
        suggestions.extend(_show_suggestions(tally))
    for key, tally in sorted(interests.items(), key=lambda kv: -kv[1].surfaced):
        suggestions.extend(_interest_suggestions(tally, weights.get(key, 5)))

    log.info(
        "insights.precision_report",
        days=days,
        surfaced=totals.surfaced,
        starred=totals.starred,
        suggestions=len(suggestions),
    )
    return {
        "days": days,
        "min_sample": MIN_SAMPLE,
        # What a wider window would add, so the page can say so rather than
        # leaving the difference to be discovered.
        "outside_window": {
            "starred": outside.starred,
            "flagged": outside.flagged_over + outside.flagged_under,
        },
        "totals": totals.as_dict(),
        "shows": [t.as_dict() for t in sorted(shows.values(), key=lambda t: -t.surfaced)],
        "interests": [
            t.as_dict() | {"weight": weights.get(key, 5)}
            for key, t in sorted(interests.items(), key=lambda kv: -kv[1].surfaced)
        ],
        "by_month": [{"month": m, **counts} for m, counts in sorted(per_month.items())],
        "suggestions": suggestions,
        # Said plainly, because a report nobody can calibrate is a report nobody
        # should act on.
        "caveat": (
            "Starring is optional, so an unstarred episode is weak evidence — it is "
            f"also what reading something useful looks like. Nothing below {MIN_SAMPLE} "
            "surfaced episodes gets a suggestion, and nothing here is ever applied for you."
        ),
    }


def _accumulate(
    episode: Doc,
    shows: dict[str, Tally],
    interests: dict[str, Tally],
    totals: Tally,
    per_month: dict[str, dict[str, int]],
    priorities: dict[str, str],
    names: dict[str, str],
    labels: dict[str, str],
) -> None:
    slug = str(episode.get("podcast_slug") or "")
    show = shows.setdefault(
        slug,
        Tally(label=names.get(slug, slug) or slug, priority=priorities.get(slug)),
    )
    starred = bool(episode.get("starred"))
    read = bool(episode.get("read_at"))
    verdict = (episode.get("feedback") or {}).get("verdict")

    for tally in (show, totals):
        tally.surfaced += 1
        tally.starred += starred
        tally.read += read
        tally.flagged_over += verdict == "over"
        tally.flagged_under += verdict == "under"

    month = (episode.get("published_at") or "")[:7]
    if month:
        per_month[month]["surfaced"] += 1
        per_month[month]["starred"] += starred

    for key in (episode.get("tier1") or {}).get("matched_interests") or []:
        key = str(key)
        interest = interests.setdefault(key, Tally(label=labels.get(key, key)))
        interest.surfaced += 1
        interest.starred += starred
        interest.read += read
        interest.flagged_over += verdict == "over"
        interest.flagged_under += verdict == "under"
