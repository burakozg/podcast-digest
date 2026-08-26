"""One topic note, many writers.

These are the tests that decide whether a person's own writing survives. The
agent rewrites every topic note weekly; if the merge is wrong, the failure is
silent and looks like Obsidian lost your work.
"""

from __future__ import annotations

from podcast_agent.notes import (
    OWNER,
    merge_frontmatter,
    merge_owned_section,
    split_frontmatter,
    wrap,
)

OURS = (
    '---\ntype: topic\ntitle: "Anthropic"\ntags: [topic]\n'
    "podcasts_mentions: 97\npodcasts_shows: 16\n---\n\n"
    "# Anthropic\n\n"
    + wrap("## From podcasts\n\n*97 episodes across 16 shows*\n\n- 2026-08-20 · **Caveat** — New")
    + "\n"
)


class TestCreatingANoteThatDoesNotExistYet:
    def test_ours_becomes_the_file(self) -> None:
        assert merge_owned_section(None, OURS) == OURS
        assert merge_owned_section("", OURS) == OURS
        assert merge_owned_section("   \n", OURS) == OURS


class TestWhatBelongsToSomebodyElseIsNotTouched:
    EXISTING = (
        "---\n"
        "type: topic\n"
        'title: "Anthropic"\n'
        "tags: [topic, ai]\n"
        "clippings_count: 12\n"
        "podcasts_mentions: 4\n"
        "---\n\n"
        "# Anthropic\n\n"
        "My own view: they ship faster than they explain.\n\n"
        "<!-- begin:podcast-digest -->\n## From podcasts\n\n- STALE\n<!-- end:podcast-digest -->\n\n"
        "<!-- begin:notes-app -->\n## From clippings\n\n- An article I clipped\n<!-- end:notes-app -->\n"
    )

    def test_the_readers_own_prose_survives(self) -> None:
        out = merge_owned_section(self.EXISTING, OURS)
        assert "My own view: they ship faster than they explain." in out

    def test_another_applications_section_survives(self) -> None:
        out = merge_owned_section(self.EXISTING, OURS)
        assert "<!-- begin:notes-app -->" in out
        assert "- An article I clipped" in out

    def test_our_section_is_replaced_in_place(self) -> None:
        out = merge_owned_section(self.EXISTING, OURS)
        assert "- STALE" not in out
        assert "- 2026-08-20 · **Caveat** — New" in out
        # In place: the reader's prose is still above ours, the other app below.
        assert out.index("My own view") < out.index("## From podcasts")
        assert out.index("## From podcasts") < out.index("## From clippings")

    def test_a_tag_the_reader_added_is_not_dropped(self) -> None:
        # The subtle one: YAML takes the last duplicate key, so appending our own
        # `tags: [topic]` would silently lose `ai` with nothing to show for it.
        out = merge_owned_section(self.EXISTING, OURS)
        assert "tags: [topic, ai]" in out
        assert out.count("tags:") == 1

    def test_another_applications_frontmatter_key_survives(self) -> None:
        assert "clippings_count: 12" in merge_owned_section(self.EXISTING, OURS)

    def test_our_own_keys_are_brought_up_to_date(self) -> None:
        out = merge_owned_section(self.EXISTING, OURS)
        assert "podcasts_mentions: 97" in out
        assert "podcasts_mentions: 4" not in out

    def test_merging_twice_changes_nothing(self) -> None:
        once = merge_owned_section(self.EXISTING, OURS)
        assert merge_owned_section(once, OURS) == once


class TestAdoptingTheNotesAlreadyInTheVault:
    """137 notes were written before ownership markers existed. They must be
    taken over, not duplicated beside a second copy of the same list."""

    LEGACY = (
        "---\n"
        "type: podcast-entity\n"
        'entity: "Anthropic"\n'
        "mentions: 97\n"
        "shows: 16\n"
        "first_seen: 2025-07-02\n"
        "last_seen: 2026-08-20\n"
        "tags: [podcast-entity, cybersecurity]\n"
        "---\n\n"
        "# Anthropic\n\n"
        "*97 episodes across 16 shows · 2025-07-02 → 2026-08-20*\n\n"
        "## Mentioned in\n\n"
        "- 2026-08-20 · **Caveat** — Old line `8/10`\n"
    )

    def test_the_old_list_is_removed_rather_than_left_to_go_stale(self) -> None:
        out = merge_owned_section(self.LEGACY, OURS)
        assert "## Mentioned in" not in out
        assert "Old line" not in out

    def test_the_new_section_is_there_exactly_once(self) -> None:
        out = merge_owned_section(self.LEGACY, OURS)
        assert out.count("## From podcasts") == 1
        assert out.count(f"<!-- begin:{OWNER} -->") == 1

    def test_the_old_unnamespaced_counts_are_retired(self) -> None:
        out = merge_owned_section(self.LEGACY, OURS)
        front, _ = split_frontmatter(out)
        assert not any(
            line.startswith(("mentions:", "shows:", "first_seen:", "entity:")) for line in front
        )
        assert "podcasts_mentions: 97" in out

    def test_it_stops_claiming_to_be_the_only_writer(self) -> None:
        assert "type: podcast-entity" not in merge_owned_section(self.LEGACY, OURS)

    def test_migrating_twice_changes_nothing(self) -> None:
        once = merge_owned_section(self.LEGACY, OURS)
        assert merge_owned_section(once, OURS) == once


class TestANoteAPersonWroteThemselves:
    def test_our_section_is_appended_without_disturbing_theirs(self) -> None:
        theirs = "# Anthropic\n\nEverything I think about them.\n"
        out = merge_owned_section(theirs, OURS)
        assert "Everything I think about them." in out
        assert "## From podcasts" in out
        assert out.index("Everything I think") < out.index("## From podcasts")

    def test_they_keep_their_own_heading(self) -> None:
        theirs = "# Anthropic, the company\n\nMine.\n"
        out = merge_owned_section(theirs, OURS)
        assert "# Anthropic, the company" in out
        assert out.count("# Anthropic") == 1


class TestFrontmatterRules:
    def test_a_shared_key_is_supplied_only_when_absent(self) -> None:
        assert merge_frontmatter(["tags: [mine]"], ["tags: [topic]"]) == ["tags: [mine]"]
        assert merge_frontmatter(["other: 1"], ["tags: [topic]"]) == ["other: 1", "tags: [topic]"]

    def test_our_prefixed_keys_always_win(self) -> None:
        assert merge_frontmatter(["podcasts_mentions: 4"], ["podcasts_mentions: 97"]) == [
            "podcasts_mentions: 97"
        ]

    def test_one_of_ours_that_stopped_being_produced_is_dropped(self) -> None:
        assert merge_frontmatter(["podcasts_gone: 1"], ["podcasts_mentions: 2"]) == [
            "podcasts_mentions: 2"
        ]

    def test_a_note_without_frontmatter_is_read_whole(self) -> None:
        assert split_frontmatter("# Just a note\n") == ([], "# Just a note\n")
