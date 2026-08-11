"""Precision report over reader signals (roadmap C1, phase 1).

The three limits in the module docstring are the design, so most of these tests
are about what the report refuses to say: nothing below the sample floor,
nothing applied, and nothing that presents a missing star as a complaint.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.db import MemoryStore
from podcast_agent.insights import MIN_SAMPLE, precision_report
from podcast_agent.main import build_app
from podcast_agent.state import EpisodeStatus
from podcast_agent.utils import podcast_doc_id

S = EpisodeStatus
KEY = {"X-API-Key": "test-admin-key"}
RECENT = datetime.now(UTC) - timedelta(days=5)


def surfaced(
    guid: str,
    *,
    slug: str = "test-show",
    starred: bool = False,
    read: bool = False,
    verdict: str | None = None,
    interests: list[str] | None = None,
    status: EpisodeStatus = S.READY_FOR_DIGEST,
) -> dict[str, Any]:
    doc = make_episode(
        guid=guid,
        podcast_slug=slug,
        status=status,
        published_at=RECENT,
        tier1={
            "relevance_score": 8,
            "summary_md": "s",
            "matched_interests": interests or [],
        },
    )
    doc["starred"] = starred
    if read:
        doc["read_at"] = "2026-08-01T00:00:00+00:00"
    if verdict:
        doc["feedback"] = {"verdict": verdict, "at": "2026-08-01T00:00:00+00:00"}
    return doc


def kinds(report: dict[str, Any]) -> set[str]:
    return {s["kind"] for s in report["suggestions"]}


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


class TestWhatCounts:
    async def test_only_surfaced_episodes_are_counted(self, store: MemoryStore, settings) -> None:
        """A dropped episode is not something you declined to star.

        You were never shown it, so counting it as an unstarred miss would
        punish the pipeline for filtering correctly.
        """
        store.seed(
            surfaced("shown"),
            make_episode(guid="dropped", status=S.DROPPED, published_at=RECENT),
            make_episode(guid="low", status=S.SCORED_LOW, published_at=RECENT),
        )
        report = await precision_report(store, settings)
        assert report["totals"]["surfaced"] == 1

    async def test_published_episodes_still_count(self, store: MemoryStore, settings) -> None:
        """They are the ones actually written into a digest and read."""
        store.seed(surfaced("a", status=S.PUBLISHED))
        assert (await precision_report(store, settings))["totals"]["surfaced"] == 1

    async def test_signals_are_tallied_per_show(self, store: MemoryStore, settings) -> None:
        store.seed(
            surfaced("a", starred=True),
            surfaced("b", read=True),
            surfaced("c", verdict="over"),
            surfaced("d", verdict="under"),
        )
        show = (await precision_report(store, settings))["shows"][0]
        assert (show["surfaced"], show["starred"], show["read"]) == (4, 1, 1)
        assert (show["flagged_over"], show["flagged_under"]) == (1, 1)

    async def test_interests_are_tallied_separately(self, store: MemoryStore, settings) -> None:
        store.seed(
            surfaced("a", starred=True, interests=["ot_ics"]),
            surfaced("b", interests=["ot_ics", "ai_agent_security"]),
        )
        by_label = {i["label"]: i for i in (await precision_report(store, settings))["interests"]}
        assert by_label["OT/ICS security"]["surfaced"] == 2
        assert by_label["OT/ICS security"]["starred"] == 1

    async def test_the_window_is_respected(self, store: MemoryStore, settings) -> None:
        old = make_episode(
            guid="old",
            status=S.READY_FOR_DIGEST,
            published_at=datetime.now(UTC) - timedelta(days=400),
            tier1={"relevance_score": 8, "summary_md": "s"},
        )
        store.seed(surfaced("new"), old)
        assert (await precision_report(store, settings, days=90))["totals"]["surfaced"] == 1


class TestItRefusesToOverclaim:
    async def test_nothing_is_suggested_below_the_sample_floor(
        self, store: MemoryStore, settings
    ) -> None:
        """ "You starred 0 of 2" is not evidence, and saying it trains the
        reader to ignore the report."""
        store.seed(*[surfaced(f"e{i}") for i in range(MIN_SAMPLE - 1)])
        report = await precision_report(store, settings)
        assert report["suggestions"] == []
        assert report["shows"][0]["enough_to_judge"] is False
        assert report["shows"][0]["precision"] is None

    async def test_a_zero_star_run_is_offered_only_as_weak_evidence(
        self, store: MemoryStore, settings
    ) -> None:
        """Not starring is also what reading something useful looks like."""
        store.seed(*[surfaced(f"e{i}") for i in range(MIN_SAMPLE)])
        report = await precision_report(store, settings)
        review = next(s for s in report["suggestions"] if s["kind"] == "review_show")
        assert review["confidence"].startswith("weak")

    async def test_an_explicit_flag_is_strong_evidence(self, store: MemoryStore, settings) -> None:
        store.seed(
            *[surfaced(f"e{i}", verdict="over") for i in range(2)],
            *[surfaced(f"f{i}") for i in range(MIN_SAMPLE)],
        )
        report = await precision_report(store, settings)
        demote = next(s for s in report["suggestions"] if s["kind"] == "demote_show")
        assert demote["confidence"].startswith("strong")

    async def test_a_single_under_flag_is_reported_without_a_sample(
        self, store: MemoryStore, settings
    ) -> None:
        """A false negative is the expensive kind: nobody sees what they were
        not shown, so one deliberate report of it is worth acting on."""
        store.seed(surfaced("a", verdict="under"))
        assert "promote_show" in kinds(await precision_report(store, settings))

    async def test_the_caveat_is_part_of_the_report(self, store: MemoryStore, settings) -> None:
        report = await precision_report(store, settings)
        assert "weak evidence" in report["caveat"]
        assert "nothing here is ever applied for you" in report["caveat"]


class TestConsoleAddedPodcastsAreNamed:
    """The file is only half the list.

    A podcast added in the console lives in the database, so reading
    `settings.podcasts` here listed five shows by their slug with no priority —
    and a demotion suggestion cannot name the next priority down for a show
    whose current one it cannot see.
    """

    async def _report(self, store: MemoryStore, settings) -> dict[str, Any]:
        return await precision_report(store, settings)

    def _console_podcast(self, slug: str, name: str, priority: str) -> dict[str, Any]:
        """Shaped the way `podcasts.add_podcast` actually writes one: the
        settings live under `overrides`, not at the top level."""
        return {
            "_id": podcast_doc_id(slug),
            "type": "podcast",
            "slug": slug,
            "source": "console",
            "name": name,
            "feed_url": "https://example.com/added.xml",
            "overrides": {
                "name": name,
                "feed_url": "https://example.com/added.xml",
                "enabled": True,
                "priority": priority,
            },
        }

    async def test_a_console_added_podcast_shows_its_name(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(
            self._console_podcast("added-show", "Added Show", "high"),
            surfaced("a", slug="added-show"),
        )
        labels = [s["label"] for s in (await self._report(store, settings))["shows"]]
        assert "Added Show" in labels, f"listed by slug instead: {labels}"

    async def test_a_console_added_podcast_shows_its_priority(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(
            self._console_podcast("added-show", "Added Show", "high"),
            surfaced("a", slug="added-show"),
        )
        row = next(
            s for s in (await self._report(store, settings))["shows"] if s["label"] == "Added Show"
        )
        assert row["priority"] == "high"

    async def test_a_console_override_wins_over_the_file(
        self, store: MemoryStore, settings
    ) -> None:
        """The registry merges them, so the report must agree with the Podcasts
        page rather than with what config.yaml said originally."""
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "source": "config",
                "overrides": {"priority": "low"},
            },
            surfaced("a"),
        )
        row = next(
            s for s in (await self._report(store, settings))["shows"] if s["label"] == "Test Show"
        )
        assert row["priority"] == "low"


class TestMarksOutsideTheWindow:
    """Starring four episodes and seeing one counted looks like a broken report.

    The window is doing it, and the page said nothing — which is the same
    silence as a queue that never drains: a number that omits without saying so.
    """

    def _old(self, guid: str, *, days: int, starred: bool = False, verdict: str | None = None):
        doc = make_episode(
            guid=guid,
            status=S.PUBLISHED,
            published_at=datetime.now(UTC) - timedelta(days=days),
            tier1={"relevance_score": 8, "summary_md": "s"},
        )
        doc["starred"] = starred
        if verdict:
            doc["feedback"] = {"verdict": verdict, "at": "2026-08-01T00:00:00+00:00"}
        return doc

    async def test_a_star_beyond_the_window_is_counted_separately(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(
            self._old("recent", days=5, starred=True), self._old("old", days=200, starred=True)
        )
        report = await precision_report(store, settings, days=90)
        assert report["totals"]["starred"] == 1
        assert report["outside_window"]["starred"] == 1

    async def test_widening_the_window_moves_it_inside(self, store: MemoryStore, settings) -> None:
        store.seed(self._old("old", days=200, starred=True))
        report = await precision_report(store, settings, days=365)
        assert report["totals"]["starred"] == 1
        assert report["outside_window"]["starred"] == 0

    async def test_a_flag_beyond_the_window_is_counted_too(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(self._old("old", days=200, verdict="over"))
        report = await precision_report(store, settings, days=90)
        assert report["outside_window"]["flagged"] == 1

    async def test_an_unstarred_old_episode_is_not_counted(
        self, store: MemoryStore, settings
    ) -> None:
        """Only deliberate marks. Counting old unstarred episodes would just
        restate the size of the archive."""
        store.seed(self._old("old", days=200))
        report = await precision_report(store, settings, days=90)
        assert report["outside_window"] == {"starred": 0, "flagged": 0}


class TestSuggestions:
    async def test_a_demotion_names_the_next_priority_down(
        self, tmp_path, store: MemoryStore
    ) -> None:
        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": "https://example.com/feed.xml",
                    "priority": "high",
                }
            ],
        )
        store.seed(*[surfaced(f"e{i}") for i in range(MIN_SAMPLE)])
        review = next(
            s
            for s in (await precision_report(store, settings))["suggestions"]
            if s["kind"] == "review_show"
        )
        assert review["change"] == "priority: med"

    async def test_a_show_already_lowest_is_offered_disabling_not_deleting(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Nothing is ever deleted; the archive stays. Only intake stops."""
        settings = make_settings(
            tmp_path,
            podcasts=[
                {
                    "slug": "test-show",
                    "name": "Test Show",
                    "feed_url": "https://example.com/feed.xml",
                    "priority": "low",
                }
            ],
        )
        store.seed(*[surfaced(f"e{i}") for i in range(MIN_SAMPLE)])
        review = next(
            s
            for s in (await precision_report(store, settings))["suggestions"]
            if s["kind"] == "review_show"
        )
        assert "enabled: false" in review["change"]
        assert "delete" not in review["change"].lower()

    async def test_a_well_starred_interest_is_offered_more_weight(
        self, store: MemoryStore, settings
    ) -> None:
        # ai_agent_security ships at 9, so there is room above it.
        store.seed(
            *[
                surfaced(f"e{i}", starred=i % 2 == 0, interests=["ai_agent_security"])
                for i in range(MIN_SAMPLE + 2)
            ]
        )
        raise_ = next(
            s
            for s in (await precision_report(store, settings))["suggestions"]
            if s["kind"] == "raise_weight"
        )
        assert raise_["change"] == "weight: 10  (currently 9)"
        assert raise_["confidence"].startswith("moderate")

    async def test_an_interest_already_at_ten_is_not_pushed_higher(
        self, store: MemoryStore, settings
    ) -> None:
        """ot_ics ships at the maximum; there is nowhere above it to suggest."""
        store.seed(
            *[surfaced(f"e{i}", starred=True, interests=["ot_ics"]) for i in range(MIN_SAMPLE)]
        )
        report = await precision_report(store, settings)
        assert not [s for s in report["suggestions"] if s["kind"] == "raise_weight"]

    async def test_every_suggestion_carries_its_numbers(self, store: MemoryStore, settings) -> None:
        """A suggestion without its evidence is an instruction."""
        store.seed(*[surfaced(f"e{i}", verdict="over") for i in range(MIN_SAMPLE)])
        for suggestion in (await precision_report(store, settings))["suggestions"]:
            assert suggestion["why"]
            assert suggestion["change"]
            assert suggestion["confidence"]


class TestNothingIsApplied:
    async def test_the_report_does_not_touch_the_configuration(
        self, tmp_path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path)
        before = [(p.slug, p.priority) for p in settings.podcasts]
        weights = [(i.key, i.weight) for i in settings.interest_profile]
        store.seed(*[surfaced(f"e{i}", verdict="over") for i in range(MIN_SAMPLE)])
        await precision_report(store, settings)
        assert [(p.slug, p.priority) for p in settings.podcasts] == before
        assert [(i.key, i.weight) for i in settings.interest_profile] == weights

    async def test_the_report_does_not_touch_an_episode(self, store: MemoryStore, settings) -> None:
        store.seed(*[surfaced(f"e{i}") for i in range(MIN_SAMPLE)])
        before = store.docs_of_type("episode")
        await precision_report(store, settings)
        assert store.docs_of_type("episode") == before


class TestApi:
    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_it_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/insights/precision").status_code == 401

    def test_it_reports(self, tmp_path, store: MemoryStore) -> None:
        store.seed(surfaced("a", starred=True))
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/insights/precision", headers=KEY).json()
        assert body["totals"]["starred"] == 1
        assert body["min_sample"] == MIN_SAMPLE

    def test_the_window_is_a_parameter(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert (
                client.get("/api/v1/insights/precision?days=30", headers=KEY).json()["days"] == 30
            )

    def test_an_absurd_window_is_refused(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/insights/precision?days=1", headers=KEY).status_code == 422
