"""One note per summarised episode, so a topic note's lines lead somewhere.

The acceptance test for the whole feature is the link rate: before this existed,
2,210 of 2,702 lines in the vault's topic notes pointed at nothing, because only
episodes that reached a *weekly* digest had anything to link to.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from helpers import make_episode, make_settings

from podcast_agent.db import MemoryStore
from podcast_agent.digest.episode_notes import (
    EPISODES_DIR,
    note_name,
    show_folder,
    summarised_episodes,
    write_episode_notes,
)
from podcast_agent.entities import EPISODE_NAMES_DOC_ID, aggregate, rank, write_entity_notes
from podcast_agent.state import EpisodeStatus as S

PUBLISHED = datetime(2026, 7, 28, tzinfo=UTC)
ARCHIVE = datetime(2024, 3, 4, tzinfo=UTC)


def _episode(guid: str, *, when: datetime, status: S, digest_id: str | None, **over: Any):
    doc = make_episode(
        guid=guid,
        status=status,
        published_at=when,
        tier1={
            "relevance_score": 8,
            "summary_md": f"What {guid} was about.",
            "why_it_matters": "It matters.",
            "entities": ["Volt Typhoon"],
            "key_takeaways": ["A point"],
            "summary_basis": "transcript",
        },
        **over,
    )
    doc["digest_id"] = digest_id
    return doc


class TestWhichEpisodesGetANote:
    async def test_every_episode_with_a_summary(self, store: MemoryStore) -> None:
        store.seed(
            _episode("weekly", when=PUBLISHED, status=S.PUBLISHED, digest_id="digest:2026-W31"),
            # The archive: published under `archive:<slug>:<month>`, which is why
            # it never had a week to link back to.
            _episode(
                "old", when=ARCHIVE, status=S.PUBLISHED, digest_id="archive:test-show:2024-03"
            ),
            # Summarised but below the digest threshold. It still contributes
            # entities to topic notes, so a line for it exists either way — and a
            # second class of dead link is indistinguishable from the first.
            _episode("low", when=PUBLISHED, status=S.SCORED_LOW, digest_id=None),
        )
        found = await summarised_episodes(store)
        assert {e["guid"] for e in found} == {"weekly", "old", "low"}

    async def test_an_episode_with_no_summary_gets_none(self, store: MemoryStore) -> None:
        store.seed(make_episode(guid="dropped", status=S.DROPPED, published_at=PUBLISHED))
        assert await summarised_episodes(store) == []


class TestTheNotesThemselves:
    async def test_one_file_per_episode_named_by_date_and_title(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(
            _episode(
                "a",
                when=PUBLISHED,
                status=S.PUBLISHED,
                digest_id="digest:2026-W31",
                title="A Thing",
            )
        )
        names = await write_episode_notes(store, settings, week_of={"digest:2026-W31": "2026-W31"})

        episodes = settings.output.digest_dir / EPISODES_DIR
        notes = sorted(episodes.rglob("*.md"))
        # Filed under the show, so the folder is browsable rather than a flat
        # list of several hundred files.
        assert [p.relative_to(episodes).as_posix() for p in notes] == [
            "Test Show/2026-07-28-a-thing.md"
        ]
        # What a link points at is the filename alone: Obsidian resolves
        # `[[wikilinks]]` by filename, so grouping changes no link anywhere.
        assert list(names.values()) == ["2026-07-28-a-thing"]

    async def test_an_episode_that_reached_a_digest_links_back_to_its_week(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(_episode("a", when=PUBLISHED, status=S.PUBLISHED, digest_id="digest:2026-W31"))
        await write_episode_notes(store, settings, week_of={"digest:2026-W31": "2026-W31"})
        text = next((settings.output.digest_dir / EPISODES_DIR).rglob("*.md")).read_text()
        assert "type: podcast-episode" in text
        assert "[[podcast-digest-2026-W31]]" in text

    async def test_an_archive_episode_links_to_no_week_at_all(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        # The bug this feature exists to fix, in its smallest form: an archive
        # episode has no week, and must not claim one that was never written.
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(
            _episode("old", when=ARCHIVE, status=S.PUBLISHED, digest_id="archive:test-show:2024-03")
        )
        await write_episode_notes(store, settings, week_of={})
        text = next((settings.output.digest_dir / EPISODES_DIR).rglob("*.md")).read_text()
        assert "podcast-digest-None" not in text
        assert "[[podcast-digest" not in text
        assert "What old was about." in text  # the summary is still there

    async def test_a_rerun_rewrites_in_place(self, tmp_path: Path, store: MemoryStore) -> None:
        # Unlike a digest, these are a view of the corpus and are meant to be
        # refreshed — no `-r2` beside the first.
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(_episode("a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None))
        await write_episode_notes(store, settings)
        await write_episode_notes(store, settings)
        assert len(list((settings.output.digest_dir / EPISODES_DIR).rglob("*.md"))) == 1


class TestFilenamesArePinned:
    """The same rule topic notes now follow: publishers edit titles, and a name
    recomputed from one moves the note out from under every link to it."""

    async def test_a_retitled_episode_keeps_its_note(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(
            _episode(
                "a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None, title="Original Title"
            )
        )
        first = await write_episode_notes(store, settings)
        assert list(first.values()) == ["2026-07-28-original-title"]

        doc = await store.get(next(iter(first)))
        assert doc is not None
        doc["title"] = "Publisher Renamed This"
        await store.put(doc)

        second = await write_episode_notes(store, settings)
        assert list(second.values()) == ["2026-07-28-original-title"], "the note was renamed"
        notes = sorted(p.name for p in (settings.output.digest_dir / EPISODES_DIR).rglob("*.md"))
        assert notes == ["2026-07-28-original-title.md"], f"orphan appeared: {notes}"

    async def test_the_heading_still_follows_the_new_title(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(_episode("a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None, title="Old"))
        await write_episode_notes(store, settings)
        doc = await store.get((await store.find({"type": "episode"}, limit=1))[0]["_id"])
        assert doc is not None
        doc["title"] = "New Title"
        await store.put(doc)
        await write_episode_notes(store, settings)
        text = next((settings.output.digest_dir / EPISODES_DIR).rglob("*.md")).read_text()
        assert "# New Title" in text

    async def test_the_pin_is_recorded(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(_episode("a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None))
        names = await write_episode_notes(store, settings)
        doc = await store.get(EPISODE_NAMES_DOC_ID)
        assert doc is not None
        assert doc["names"] == names

    def test_an_untitled_episode_does_not_collide_with_every_other(self) -> None:
        a = note_name({"_id": "episode:x", "published_at": "2026-07-28T00:00:00Z", "title": ""})
        b = note_name({"_id": "episode:y", "published_at": "2026-07-28T00:00:00Z", "title": ""})
        assert a != b


class TestTopicNotesLinkTheEpisode:
    """The point of all of it."""

    async def test_a_line_links_the_episode_and_still_names_its_week(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(
            _episode(
                "a",
                when=PUBLISHED,
                status=S.PUBLISHED,
                digest_id="digest:2026-W31",
                title="A Thing",
            )
        )
        weeks = {"digest:2026-W31": "2026-W31"}
        note_of = await write_episode_notes(store, settings, week_of=weeks)
        await write_entity_notes(
            store,
            settings,
            rank(await aggregate(store), min_mentions=1),
            week_of=weeks,
            note_of=note_of,
        )
        text = (settings.output.digest_dir / "entities/volt-typhoon.md").read_text()
        assert "[[2026-07-28-a-thing|A Thing]]" in text
        assert "[[podcast-digest-2026-W31]]" in text

    async def test_an_archive_line_links_the_episode_though_it_has_no_week(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(
            _episode(
                "old",
                when=ARCHIVE,
                status=S.PUBLISHED,
                digest_id="archive:test-show:2024-03",
                title="Old Thing",
            )
        )
        note_of = await write_episode_notes(store, settings, week_of={})
        await write_entity_notes(
            store, settings, rank(await aggregate(store), min_mentions=1), note_of=note_of
        )
        text = (settings.output.digest_dir / "entities/volt-typhoon.md").read_text()
        assert "[[2024-03-04-old-thing|Old Thing]]" in text
        # The ownership marker `<!-- begin:podcast-digest -->` is in every topic
        # note; what must be absent is a *link* to a week that does not exist.
        assert "[[podcast-digest-" not in text

    async def test_without_a_note_the_line_is_plain_text_not_a_dangling_link(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path)
        store.seed(
            _episode("a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None, title="A Thing")
        )
        await write_entity_notes(
            store, settings, rank(await aggregate(store), min_mentions=1), note_of={}
        )
        text = (settings.output.digest_dir / "entities/volt-typhoon.md").read_text()
        assert "A Thing" in text
        assert "[[" not in text.split("## From podcasts")[1]


class TestProjection:
    def test_episode_notes_are_routed_to_their_own_vault_folder(self) -> None:
        from podcast_agent.config import VaultConfig
        from podcast_agent.vault import build_vault

        v = build_vault(VaultConfig(enabled=True, couchdb_url="http://x:5984"), None)
        assert v.vault_path(Path("episodes/Test Show/2026-07-28-a-thing.md")) == (
            "11 podcasts/episodes/Test Show/2026-07-28-a-thing.md"
        )
        assert v.vault_path(Path("entities/x.md")) == "99 topics/x.md"
        assert v.vault_path(Path("2026/w34.md")) == "11 podcasts/digests/2026/w34.md"

    def test_they_are_in_what_gets_projected(self, tmp_path: Path) -> None:
        from podcast_agent.vault import projectable

        (tmp_path / EPISODES_DIR / "Test Show").mkdir(parents=True)
        (tmp_path / EPISODES_DIR / "Test Show" / "a.md").write_text("x", encoding="utf-8")
        assert [p.name for p in projectable(tmp_path)] == ["a.md"]

    def test_a_note_left_at_the_old_flat_path_is_not_projected_alongside_it(
        self, tmp_path: Path
    ) -> None:
        # Both would arrive in the vault under the same filename, and a
        # `[[wikilink]]` with two targets resolves to one of them silently.
        from podcast_agent.vault import projectable

        (tmp_path / EPISODES_DIR / "Test Show").mkdir(parents=True)
        (tmp_path / EPISODES_DIR / "Test Show" / "a.md").write_text("x", encoding="utf-8")
        (tmp_path / EPISODES_DIR / "a.md").write_text("x", encoding="utf-8")
        assert [p.parent.name for p in projectable(tmp_path)] == ["Test Show"]


@pytest.mark.parametrize(
    ("published", "title", "expected"),
    [
        ("2026-07-28T10:00:00Z", "A Thing", "2026-07-28-a-thing"),
        ("", "A Thing", "undated-a-thing"),
        # slugify caps the length itself; note_name adds no second cap.
        ("2026-07-28T10:00:00Z", "x" * 200, "2026-07-28-" + "x" * 60),
    ],
)
def test_note_name_shapes(published: str, title: str, expected: str) -> None:
    assert note_name({"_id": "episode:1", "published_at": published, "title": title}) == expected


class TestTitlesThatWouldBreakTheLink:
    """Podcast titles really do contain pipes and brackets — 18 episodes lost
    their link to one before the alias was sanitised rather than the link
    dropped."""

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            (r"VCISO Tradecraft \| Carlota Sage", "VCISO Tradecraft - Carlota Sage"),
            (r"A little help. \[Research Saturday\]", "A little help. (Research Saturday)"),
            ("Plain title", "Plain title"),
        ],
    )
    def test_the_alias_is_made_safe(self, title: str, expected: str) -> None:
        from podcast_agent.entities import _alias

        assert _alias(title) == expected

    async def test_such_an_episode_still_gets_linked(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(
            _episode(
                "a",
                when=PUBLISHED,
                status=S.PUBLISHED,
                digest_id=None,
                title="Tradecraft | Guest [Special]",
            )
        )
        note_of = await write_episode_notes(store, settings)
        await write_entity_notes(
            store, settings, rank(await aggregate(store), min_mentions=1), note_of=note_of
        )
        body = (settings.output.digest_dir / "entities/volt-typhoon.md").read_text()
        line = next(row for row in body.splitlines() if row.startswith("- 20"))
        assert line.count("[[") == 1 and line.count("]]") == 1
        assert "Tradecraft - Guest (Special)" in line


class TestTheFolderAShowGetsFiledUnder:
    def test_the_show_name_as_written(self) -> None:
        assert show_folder({"podcast_name": "Risky Business"}) == "Risky Business"

    def test_a_name_no_filesystem_would_accept_is_made_safe(self) -> None:
        # A real one: the vault has to open on Windows and iOS too, and neither
        # takes `:` in a path. Replaced with a space rather than dropped, so the
        # words do not run together.
        assert show_folder(
            {"podcast_name": "No Priors: Artificial Intelligence | Technology | Startups"}
        ) == ("No Priors Artificial Intelligence Technology Startups")

    def test_a_trailing_dot_does_not_survive(self) -> None:
        # Legal to create on Linux, impossible to open on Windows.
        assert show_folder({"podcast_name": "The Daily."}) == "The Daily"

    def test_a_show_with_no_name_falls_back_to_its_slug(self) -> None:
        assert show_folder({"podcast_name": "", "podcast_slug": "test-show"}) == "test-show"


class TestNamesAreUniqueAcrossShows:
    """Filing per show is what makes this necessary: two shows publishing "Week
    in review" on one day would previously have shared a file, and now share a
    filename — which is worse, because a `[[wikilink]]` with two targets
    resolves to one of them and says nothing."""

    async def test_a_second_show_with_the_same_title_gets_its_own_name(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(
            _episode(
                "a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None, title="Week in review"
            ),
            _episode(
                "b",
                when=PUBLISHED,
                status=S.PUBLISHED,
                digest_id=None,
                title="Week in review",
                slug="other-show",
                podcast_name="Other Show",
            ),
        )
        names = await write_episode_notes(store, settings)
        assert len(set(names.values())) == 2, names
        assert "2026-07-28-week-in-review" in names.values()
        # The disambiguator names the show rather than counting, so the reader
        # can tell which note is which without opening both.
        assert any(n.endswith("-other-show") or n.endswith("-test-show") for n in names.values())

    async def test_the_second_run_does_not_rename_either_of_them(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        store.seed(
            _episode(
                "a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None, title="Week in review"
            ),
            _episode(
                "b",
                when=PUBLISHED,
                status=S.PUBLISHED,
                digest_id=None,
                title="Week in review",
                slug="other-show",
                podcast_name="Other Show",
            ),
        )
        first = await write_episode_notes(store, settings)
        assert await write_episode_notes(store, settings) == first


class TestNotesLeftAtAPathWeStoppedUsing:
    """A note that moves has to *finish* moving. Two files with one name make
    every link to that name ambiguous, and Obsidian resolves it silently."""

    async def test_the_flat_layout_is_cleared_away(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        episodes = settings.output.digest_dir / EPISODES_DIR
        episodes.mkdir(parents=True)
        stale = episodes / "2026-07-28-a-thing.md"
        stale.write_text("written before notes were grouped by show", encoding="utf-8")

        store.seed(
            _episode("a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None, title="A Thing")
        )
        await write_episode_notes(store, settings)

        assert not stale.exists()
        assert [p.relative_to(episodes).as_posix() for p in episodes.rglob("*.md")] == [
            "Test Show/2026-07-28-a-thing.md"
        ]

    async def test_a_show_that_renames_itself_takes_its_notes_with_it(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        episodes = settings.output.digest_dir / EPISODES_DIR
        store.seed(
            _episode("a", when=PUBLISHED, status=S.PUBLISHED, digest_id=None, title="A Thing")
        )
        await write_episode_notes(store, settings)

        doc = (await store.find({"type": "episode"}, limit=1))[0]
        doc["podcast_name"] = "Test Show Reborn"
        await store.put(doc)
        await write_episode_notes(store, settings)

        assert [p.relative_to(episodes).as_posix() for p in episodes.rglob("*.md")] == [
            "Test Show Reborn/2026-07-28-a-thing.md"
        ]
        # And the folder it left, so the vault does not show an empty show.
        assert not (episodes / "Test Show").exists()
