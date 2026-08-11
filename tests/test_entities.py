"""Entity and trend tracking (roadmap D2).

Tier-1 has been extracting named things all along and nothing read them back.
One episode saying "Volt Typhoon" is a detail; six episodes across four shows
saying it over five months is a story, and no per-episode artefact shows that.

The tests that carry the most weight are the canonicalisation ones. Over-merging
is the dangerous error: it fuses two unrelated things into one timeline that
reads as evidence, and nothing downstream can tell.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.db import MemoryStore
from podcast_agent.entities import (
    aggregate,
    canonical,
    display_name,
    rank,
    timeline,
    write_entity_notes,
)
from podcast_agent.main import build_app
from podcast_agent.state import EpisodeStatus

S = EpisodeStatus
KEY = {"X-API-Key": "test-admin-key"}


def episode(guid: str, entities: list[str], *, month: int = 6, show: str = "Test Show"):
    return make_episode(
        guid=guid,
        podcast_name=show,
        status=S.READY_FOR_DIGEST,
        published_at=datetime(2026, month, 15, tzinfo=UTC),
        tier1={"relevance_score": 8, "summary_md": "s", "entities": entities},
    )


class TestCanonicalisation:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Volt Typhoon", "volt typhoon"),
            ("Volt  Typhoon", "Volt Typhoon"),
            ("Mandiant", "Mandiant Inc."),
            ("The Shadow Brokers", "Shadow Brokers"),
            ("Modbus.", "Modbus"),
        ],
    )
    def test_spellings_of_one_thing_agree(self, a: str, b: str) -> None:
        assert canonical(a) == canonical(b)

    @pytest.mark.parametrize(
        "written",
        ["CVE-2026-1234", "cve 2026 1234", "CVE_2026_1234", "cve-2026-1234"],
    )
    def test_cve_identifiers_have_one_form(self, written: str) -> None:
        assert canonical(written) == "cve-2026-1234"

    def test_leading_zeros_are_insignificant(self) -> None:
        """CVE-2026-01234 and CVE-2026-1234 are the same advisory."""
        assert canonical("CVE-2026-01234") == canonical("CVE-2026-1234")

    def test_a_four_digit_id_keeps_its_zeros(self) -> None:
        """CVE-2026-0001 is written that way and must round-trip."""
        assert canonical("CVE-2026-0001") == "cve-2026-0001"

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Volt Typhoon", "Salt Typhoon"),
            ("CVE-2026-1234", "CVE-2026-1235"),
            ("Purview", "Purview DSPM"),
            ("Sentinel", "Sentinel One"),
        ],
    )
    def test_different_things_stay_different(self, a: str, b: str) -> None:
        """Over-merging fuses two timelines into one that reads as evidence."""
        assert canonical(a) != canonical(b)

    def test_empty_input_has_no_key(self) -> None:
        assert canonical("  ") == ""
        assert canonical(" . , ") == ""

    def test_the_shown_spelling_is_the_common_one(self) -> None:
        assert display_name({"Volt Typhoon": 5, "volt typhoon": 1}) == "Volt Typhoon"

    def test_ties_prefer_the_more_informative_form(self) -> None:
        assert display_name({"Volt": 2, "Volt Typhoon": 2}) == "Volt Typhoon"


class TestAggregation:
    async def test_mentions_are_counted_across_episodes(self, store: MemoryStore) -> None:
        store.seed(
            episode("a", ["Volt Typhoon", "Modbus"]),
            episode("b", ["volt typhoon"]),
        )
        found = await aggregate(store)
        assert found[canonical("Volt Typhoon")].mentions == 2
        assert found[canonical("Modbus")].mentions == 1

    async def test_one_episode_counts_once_however_it_spells_it(self, store: MemoryStore) -> None:
        """Otherwise a model listing both spellings doubles its own evidence."""
        store.seed(episode("a", ["Volt Typhoon", "volt typhoon", "VOLT TYPHOON"]))
        assert (await aggregate(store))[canonical("Volt Typhoon")].mentions == 1

    async def test_shows_are_tracked_separately_from_mentions(self, store: MemoryStore) -> None:
        """Agreement across independent shows is the stronger signal."""
        store.seed(
            episode("a", ["Volt Typhoon"], show="Risky Business"),
            episode("b", ["Volt Typhoon"], show="Risky Business"),
            episode("c", ["Volt Typhoon"], show="Darknet Diaries"),
        )
        entity = (await aggregate(store))[canonical("Volt Typhoon")]
        assert entity.mentions == 3
        assert entity.shows == {"Risky Business", "Darknet Diaries"}

    async def test_first_and_last_seen_bound_the_story(self, store: MemoryStore) -> None:
        store.seed(
            episode("a", ["Volt Typhoon"], month=3),
            episode("b", ["Volt Typhoon"], month=7),
        )
        entity = (await aggregate(store))[canonical("Volt Typhoon")]
        assert entity.first_seen[:7] == "2026-03"
        assert entity.last_seen[:7] == "2026-07"

    async def test_episodes_without_a_summary_contribute_nothing(self, store: MemoryStore) -> None:
        """Entities come from the Tier-1 pass; a dropped episode never had one."""
        store.seed(make_episode(guid="dropped", status=S.DROPPED, tier0={"relevance_guess": 1}))
        assert await aggregate(store) == {}

    async def test_a_sentence_fragment_is_ignored(self, store: MemoryStore) -> None:
        """It would never match anything else, and poisons the index."""
        store.seed(episode("a", ["x" * 200, "Modbus"]))
        found = await aggregate(store)
        assert set(found) == {canonical("Modbus")}

    async def test_a_window_can_be_applied(self, store: MemoryStore) -> None:
        store.seed(
            episode("old", ["Volt Typhoon"], month=1),
            episode("new", ["Volt Typhoon"], month=7),
        )
        found = await aggregate(store, since="2026-06-01T00:00:00+00:00")
        assert found[canonical("Volt Typhoon")].mentions == 1


class TestRanking:
    def _entities(self, store: MemoryStore) -> None:
        store.seed(
            episode("a", ["Volt Typhoon", "Modbus"], show="A"),
            episode("b", ["Volt Typhoon"], show="B"),
            episode("c", ["Volt Typhoon"], show="C"),
            episode("d", ["Kerberos"], show="A"),
        )

    async def test_most_discussed_first(self, store: MemoryStore) -> None:
        self._entities(store)
        ranked = rank(await aggregate(store), min_mentions=1)
        assert ranked[0].name == "Volt Typhoon"

    async def test_single_mentions_are_excluded_by_default(self, store: MemoryStore) -> None:
        """A vault of four thousand single-use notes is a worse graph than none."""
        self._entities(store)
        assert {e.name for e in rank(await aggregate(store))} == {"Volt Typhoon"}

    async def test_the_threshold_is_adjustable(self, store: MemoryStore) -> None:
        self._entities(store)
        assert len(rank(await aggregate(store), min_mentions=1)) == 3

    async def test_the_monthly_shape_is_reported(self, store: MemoryStore) -> None:
        store.seed(
            episode("a", ["Volt Typhoon"], month=3),
            episode("b", ["Volt Typhoon"], month=3),
            episode("c", ["Volt Typhoon"], month=5),
        )
        entity = (await aggregate(store))[canonical("Volt Typhoon")]
        assert timeline(entity) == [
            {"month": "2026-03", "mentions": 2},
            {"month": "2026-05", "mentions": 1},
        ]


class TestObsidianNotes:
    async def test_a_note_is_written_per_entity(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(tmp_path)
        store.seed(episode("a", ["Volt Typhoon"]), episode("b", ["Volt Typhoon"]))
        ranked = rank(await aggregate(store))
        written = write_entity_notes(settings, ranked)

        assert written == ["entities/volt-typhoon.md"]
        text = (settings.output.digest_dir / written[0]).read_text()
        assert "# Volt Typhoon" in text
        assert "type: podcast-entity" in text
        assert "2 episodes across 1 show" in text

    async def test_the_note_links_the_week_an_episode_appeared_in(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Wikilinks are what give the graph view edges to draw."""
        settings = make_settings(tmp_path)
        doc = episode("a", ["Volt Typhoon"])
        doc["digest_id"] = "digest:2026-W31"
        store.seed(doc, episode("b", ["Volt Typhoon"]))
        ranked = rank(await aggregate(store))
        written = write_entity_notes(settings, ranked, week_of={"digest:2026-W31": "2026-W31"})
        assert (
            "[[podcast-digest-2026-W31]]" in (settings.output.digest_dir / written[0]).read_text()
        )

    async def test_notes_are_rewritten_not_appended(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """A stale line is worse than a rebuilt file: the reader cannot tell."""
        settings = make_settings(tmp_path)
        store.seed(episode("a", ["Volt Typhoon"]), episode("b", ["Volt Typhoon"]))
        ranked = rank(await aggregate(store))
        write_entity_notes(settings, ranked)
        write_entity_notes(settings, ranked)
        text = (settings.output.digest_dir / "entities/volt-typhoon.md").read_text()
        assert text.count("# Volt Typhoon") == 1

    async def test_notes_live_beside_the_digests_not_in_work_dir(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """They are output the vault syncs, not derived scratch."""
        settings = make_settings(tmp_path)
        store.seed(episode("a", ["Volt Typhoon"]), episode("b", ["Volt Typhoon"]))
        write_entity_notes(settings, rank(await aggregate(store)))
        assert (settings.output.digest_dir / "entities").is_dir()

    async def test_a_hostile_entity_name_cannot_escape_its_line(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Entity strings are model output over an untrusted transcript (§10.2)."""
        settings = make_settings(tmp_path)
        nasty = "Evil](http://x) [click"
        store.seed(episode("a", [nasty]), episode("b", [nasty]))
        written = write_entity_notes(settings, rank(await aggregate(store)))
        text = (settings.output.digest_dir / written[0]).read_text()
        heading = next(line for line in text.splitlines() if line.startswith("# "))
        # Every bracket that could close a link is escaped, so the heading
        # renders as the literal text rather than as a link someone can click.
        assert heading.count("](") == heading.count("\\](")
        assert "\\[" in heading
        # And the frontmatter stays parseable YAML rather than raw brackets.
        entity_line = next(line for line in text.splitlines() if line.startswith("entity:"))
        assert entity_line.startswith('entity: "')


class TestApi:
    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed(self, store: MemoryStore) -> None:
        store.seed(
            episode("a", ["Volt Typhoon", "Modbus"], show="A"),
            episode("b", ["volt typhoon"], show="B"),
        )

    def test_it_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/entities").status_code == 401

    def test_listing_ranks_by_mentions(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/entities", headers=KEY).json()
        assert body["entities"][0]["name"] == "Volt Typhoon"
        assert body["entities"][0]["show_count"] == 2

    def test_one_entity_carries_its_episodes_and_timeline(
        self, tmp_path, store: MemoryStore
    ) -> None:
        self._seed(store)
        with self._client(tmp_path, store) as client:
            body = client.get("/api/v1/entities/Volt%20Typhoon", headers=KEY).json()
        assert len(body["episodes"]) == 2
        assert body["timeline"]

    def test_an_entity_can_be_looked_up_by_any_spelling(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/entities/volt%20typhoon", headers=KEY).status_code == 200

    def test_an_unknown_entity_is_404(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/entities/nothing", headers=KEY).status_code == 404

    def test_notes_can_be_written_from_the_api(self, tmp_path, store: MemoryStore) -> None:
        self._seed(store)
        with self._client(tmp_path, store) as client:
            body = client.post("/api/v1/entities/notes", headers=KEY).json()
        assert body["written"] == 1


class TestEpisodeNotesLinkBack:
    def test_the_template_uses_wikilinks(self) -> None:
        """Edges from both ends, or the graph view has nothing to draw."""
        template = (
            Path(__file__).parent.parent / "podcast_agent/digest/templates/episode.md.j2"
        ).read_text()
        assert "e.entity_links" in template

    def test_the_weekly_digest_keeps_plain_text(self) -> None:
        """It is read top to bottom, not navigated."""
        template = (
            Path(__file__).parent.parent / "podcast_agent/digest/templates/digest.md.j2"
        ).read_text()
        assert "entity_links" not in template

    def test_the_view_builds_them(self, tmp_path: Path) -> None:
        from podcast_agent.digest.generate import BASIS_LABELS, _episode_views

        view = _episode_views(
            make_settings(tmp_path),
            episode("a", ["Volt Typhoon"]),
            BASIS_LABELS,
        )
        assert view["entity_links"] == ["[[volt-typhoon|Volt Typhoon]]"]


class TestItReadsRatherThanWrites:
    async def test_aggregating_does_not_touch_an_episode(self, store: MemoryStore) -> None:
        """Nothing here calls a model or mutates the corpus it reads."""
        store.seed(episode("a", ["Volt Typhoon"]))
        before = store.docs_of_type("episode")
        await aggregate(store)
        assert store.docs_of_type("episode") == before
