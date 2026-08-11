"""Digest generation tests (§5, §10.3).

Covers the Obsidian output contract, atomic/no-overwrite writes, the
publish-after-write ordering, and reconciliation of an interrupted run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from helpers import make_episode, make_settings

from podcast_agent.backfill.ingest import BACKFILL_ORIGIN
from podcast_agent.db import MemoryStore
from podcast_agent.digest import generate as generate_module
from podcast_agent.digest.generate import DigestGenerator, _atomic_write, _iso_week_key
from podcast_agent.state import EpisodeStatus

S = EpisodeStatus

FIXED_NOW = datetime(2026, 7, 31, 6, 0, tzinfo=UTC)
PERIOD_FROM = datetime(2026, 7, 24, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _freeze_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Digest frontmatter embeds the generation time; freeze it for determinism."""
    monkeypatch.setattr(generate_module, "utcnow", lambda: FIXED_NOW)


def tier1_block(score: int, **over: Any) -> dict[str, Any]:
    block = {
        "relevance_score": score,
        "matched_interests": ["ot_ics"],
        "why_it_matters": "Directly relevant to your OT/ICS remit.",
        "summary_md": "The hosts trace a **PLC** compromise at a water utility.",
        "key_takeaways": ["Segment OT networks", "Patch the HMI"],
        "entities": ["Modbus", "CVE-2026-1234"],
        "listen_anyway": False,
        "summary_basis": "transcript",
        "cost_usd": 0.01,
    }
    block.update(over)
    return block


def seed_corpus(store: MemoryStore) -> None:
    published = datetime(2026, 7, 28, 9, 0, tzinfo=UTC)
    store.seed(
        make_episode(
            guid="top",
            title="Deep dive: PLC malware",
            status=S.READY_FOR_DIGEST,
            published_at=published,
            tier1=tier1_block(9),
            tier0={"relevance_guess": 8, "confidence": 9, "cost_usd": 0.001},
        ),
        make_episode(
            guid="also",
            title="Briefly relevant episode",
            status=S.READY_FOR_DIGEST,
            published_at=published,
            tier1=tier1_block(6, summary_basis="description_only"),
        ),
        make_episode(
            guid="maybe",
            title="Might be interesting",
            status=S.DIGEST_DIRECT,
            published_at=published,
            tier0={
                "relevance_guess": 5,
                "confidence": 8,
                "reasoning": "Possibly touches OT topics.",
                "route": "DIGEST_DIRECT",
            },
        ),
        make_episode(
            guid="low",
            title="Scored low after summarising",
            status=S.SCORED_LOW,
            published_at=published,
            tier1=tier1_block(2),
        ),
        make_episode(
            guid="dropped",
            title="Dropped at triage",
            status=S.DROPPED,
            published_at=published,
            tier0={"relevance_guess": 1, "confidence": 9, "route": "DROP"},
        ),
    )


async def run(settings, store: MemoryStore, **kw: Any):
    return await DigestGenerator(settings, store).generate(since=kw.pop("since", PERIOD_FROM), **kw)


class TestDigestContent:
    async def test_writes_expected_path(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        result = await run(settings, store)
        assert result.file_path is not None
        assert result.file_path.name == "podcast-digest-2026-W31.md"
        assert result.file_path.parent.name == "2026"
        assert result.file_path.exists()

    async def test_frontmatter_is_obsidian_friendly(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        result = await run(settings, store)
        text = result.file_path.read_text()  # type: ignore[union-attr]
        assert text.startswith("---\n")
        head = text.split("---", 2)[1]
        assert "type: podcast-digest" in head
        assert "week: 2026-W31" in head
        assert "episodes_scanned: 5" in head
        assert "episodes_summarized: 2" in head
        assert "tags: [podcast-digest, cybersecurity]" in head

    async def test_all_four_sections_render(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        text = (await run(settings, store)).file_path.read_text()  # type: ignore[union-attr]
        assert "## Top picks" in text
        assert "## Also relevant" in text
        assert "## Maybe interesting (not summarised)" in text
        assert "## Everything else scanned" in text

    async def test_score_buckets_land_correctly(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        text = (await run(settings, store)).file_path.read_text()  # type: ignore[union-attr]
        top = text.split("## Top picks")[1].split("## Also relevant")[0]
        also = text.split("## Also relevant")[1].split("## Maybe interesting")[0]
        assert "Deep dive: PLC malware" in top
        assert "Briefly relevant episode" in also
        assert "Deep dive" not in also

    async def test_summary_basis_is_labelled_honestly(self, settings, store: MemoryStore) -> None:
        """§5: a description-only entry must say so."""
        seed_corpus(store)
        text = (await run(settings, store)).file_path.read_text()  # type: ignore[union-attr]
        assert "basis: local transcription" in text
        assert "basis: description only (no transcript available)" in text

    async def test_audit_table_lists_dropped_and_low_scored(
        self, settings, store: MemoryStore
    ) -> None:
        seed_corpus(store)
        text = (await run(settings, store)).file_path.read_text()  # type: ignore[union-attr]
        audit = text.split("## Everything else scanned")[1]
        assert "dropped at triage" in audit
        assert "summarized, scored below threshold" in audit
        assert "| 2/10 |" in audit
        assert "| ~1/10 |" in audit

    async def test_interest_labels_are_human_readable(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        text = (await run(settings, store)).file_path.read_text()  # type: ignore[union-attr]
        assert "OT/ICS security" in text
        assert "ot_ics" not in text  # raw keys never surface to the reader

    async def test_empty_period_still_renders_valid_markdown(
        self, settings, store: MemoryStore
    ) -> None:
        result = await run(settings, store)
        text = result.file_path.read_text()  # type: ignore[union-attr]
        assert "Nothing new in this period." in text
        assert text.startswith("---\n")

    async def test_stats_include_cost_and_asr_counts(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        result = await run(settings, store)
        assert result.stats["scanned"] == 5
        assert result.stats["summarized"] == 2
        assert result.stats["total_cost_usd"] > 0


class TestUntrustedContentInOutput:
    async def test_hostile_title_cannot_break_structure(self, settings, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="hostile",
                title="Ep 1\n# FAKE HEADING\n| broken | table |",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                tier1=tier1_block(9),
            )
        )
        text = (await run(settings, store)).file_path.read_text()  # type: ignore[union-attr]
        assert "\n# FAKE HEADING" not in text
        # Exactly one h1 — the digest's own.
        assert len([ln for ln in text.splitlines() if ln.startswith("# ")]) == 1

    async def test_injected_markdown_in_summary_is_defanged(
        self, settings, store: MemoryStore
    ) -> None:
        store.seed(
            make_episode(
                guid="inj",
                title="Normal title",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                tier1=tier1_block(
                    9,
                    summary_md="Legit text\n---\ntype: evil-frontmatter\n---\n# Injected h1",
                ),
            )
        )
        text = (await run(settings, store)).file_path.read_text()  # type: ignore[union-attr]
        assert "type: evil-frontmatter" not in text.split("---", 2)[1]  # not in frontmatter
        assert "\n# Injected h1" not in text
        assert "Legit text" in text

    async def test_unsafe_link_is_dropped(self, settings, store: MemoryStore) -> None:
        store.seed(
            make_episode(
                guid="badlink",
                title="Has a javascript link",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                link="javascript:alert(1)",
                tier1=tier1_block(9),
            )
        )
        text = (await run(settings, store)).file_path.read_text()  # type: ignore[union-attr]
        assert "javascript:" not in text


class TestAtomicWrites:
    def test_no_tmp_file_survives(self, tmp_path: Path) -> None:
        _atomic_write(tmp_path, Path("2026/x.md"), "content")
        assert list(tmp_path.rglob("*.tmp")) == []
        assert (tmp_path / "2026" / "x.md").read_text() == "content"

    def test_existing_file_is_never_overwritten(self, tmp_path: Path) -> None:
        """§5: a manual re-run must not clobber a digest already synced."""
        first = _atomic_write(tmp_path, Path("d.md"), "original")
        second = _atomic_write(tmp_path, Path("d.md"), "rerun")
        third = _atomic_write(tmp_path, Path("d.md"), "rerun again")
        assert first.name == "d.md"
        assert second.name == "d-r2.md"
        assert third.name == "d-r3.md"
        assert first.read_text() == "original"

    async def test_rerun_after_completion_writes_a_revision(
        self, settings, store: MemoryStore
    ) -> None:
        seed_corpus(store)
        first = await run(settings, store)
        # Episodes are claimed, so a second run covers an empty set but must not
        # overwrite the file already on disk.
        second = await run(settings, store)
        assert first.file_path != second.file_path
        assert second.file_path is not None
        assert second.file_path.name.endswith("-r2.md")


class TestPublishing:
    async def test_digestable_episodes_become_published(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        result = await run(settings, store)
        for guid in ("top", "also", "maybe"):
            doc = await store.get(make_episode(guid=guid)["_id"])
            assert doc is not None
            assert doc["status"] == S.PUBLISHED.value
            assert doc["digest_id"] == result.digest_id

    async def test_audit_rows_keep_their_status_but_are_claimed(
        self, settings, store: MemoryStore
    ) -> None:
        """SCORED_LOW must stay SCORED_LOW — the distinction is the audit trail."""
        seed_corpus(store)
        result = await run(settings, store)
        low = await store.get(make_episode(guid="low")["_id"])
        assert low is not None
        assert low["status"] == S.SCORED_LOW.value
        assert low["digest_id"] == result.digest_id

    async def test_claimed_episodes_never_appear_twice(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        await run(settings, store)
        second = await run(settings, store)
        assert second.episode_ids == []

    async def test_in_flight_episodes_are_left_for_next_time(
        self, settings, store: MemoryStore
    ) -> None:
        store.seed(
            make_episode(
                guid="pending",
                status=S.AWAITING_TRANSCRIPT,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
            ),
            make_episode(
                guid="ready",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                tier1=tier1_block(8),
            ),
        )
        result = await run(settings, store)
        assert len(result.episode_ids) == 1
        pending = await store.get(make_episode(guid="pending")["_id"])
        assert pending is not None
        assert pending["status"] == S.AWAITING_TRANSCRIPT.value
        assert pending["digest_id"] is None

    async def test_episodes_outside_the_window_are_excluded(
        self, settings, store: MemoryStore
    ) -> None:
        store.seed(
            make_episode(
                guid="old",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 6, 1, tzinfo=UTC),
                tier1=tier1_block(9),
            ),
            make_episode(
                guid="inside",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                tier1=tier1_block(9),
            ),
        )
        result = await run(settings, store)
        assert len(result.episode_ids) == 1


class TestDryRun:
    async def test_writes_nothing_and_claims_nothing(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        result = await run(settings, store, dry_run=True)
        assert result.dry_run is True
        assert result.file_path is None
        assert not list(settings.output.digest_dir.rglob("*.md"))
        top = await store.get(make_episode(guid="top")["_id"])
        assert top is not None
        assert top["status"] == S.READY_FOR_DIGEST.value
        assert top["digest_id"] is None

    async def test_still_reports_what_would_be_included(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        result = await run(settings, store, dry_run=True)
        assert len(result.episode_ids) == 5


class TestReconciliation:
    async def test_interrupted_marking_is_completed_on_the_next_run(
        self, settings, store: MemoryStore
    ) -> None:
        """§10.3: file written, marking incomplete → next run finishes the marking
        rather than emitting a duplicate digest."""
        seed_corpus(store)
        generator = DigestGenerator(settings, store)
        result = await generator.generate(since=PERIOD_FROM)

        # Simulate a crash between the file write and the marking.
        digest = await store.get(result.digest_id)
        assert digest is not None
        digest["marking_complete"] = False
        await store.put(digest)
        for guid in ("top", "also", "maybe", "low", "dropped"):
            doc = await store.get(make_episode(guid=guid)["_id"])
            assert doc is not None
            doc["digest_id"] = None
            doc["status"] = (
                S.READY_FOR_DIGEST.value
                if doc["status"] == S.PUBLISHED.value and guid in ("top", "also")
                else S.DIGEST_DIRECT.value
                if guid == "maybe"
                else doc["status"]
            )
            await store.put(doc)

        files_before = sorted(p.name for p in settings.output.digest_dir.rglob("*.md"))
        again = await generator.generate(since=PERIOD_FROM)

        assert again.reconciled is True
        # No second file for the same week.
        assert sorted(p.name for p in settings.output.digest_dir.rglob("*.md")) == files_before
        top = await store.get(make_episode(guid="top")["_id"])
        assert top is not None
        assert top["status"] == S.PUBLISHED.value


class TestEpisodeNotes:
    async def test_disabled_by_default(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        await run(settings, store)
        assert not (settings.output.digest_dir / "episodes").exists()

    async def test_enabled_writes_linked_notes(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(tmp_path, output={"episode_notes": True})
        seed_corpus(store)
        await run(settings, store)
        notes = list((settings.output.digest_dir / "episodes").rglob("*.md"))
        assert len(notes) == 2  # top picks + also relevant
        content = notes[0].read_text()
        assert content.startswith("---\n")
        assert "type: podcast-episode" in content
        assert "[[podcast-digest-2026-W31]]" in content


class TestPeriodKeys:
    @pytest.mark.parametrize(
        ("when", "expected"),
        [
            (datetime(2026, 7, 31, tzinfo=UTC), "2026-W31"),
            (datetime(2026, 1, 1, tzinfo=UTC), "2026-W01"),
            # ISO week years can differ from calendar years at the boundary.
            (datetime(2027, 1, 1, tzinfo=UTC), "2026-W53"),
        ],
    )
    def test_iso_week_key(self, when: datetime, expected: str) -> None:
        assert _iso_week_key(when) == expected


class TestGoldenFile:
    async def test_matches_golden_output(self, settings, store: MemoryStore) -> None:
        """Full-document snapshot. Regenerate with:
        REGENERATE_GOLDEN=1 uv run pytest tests/test_digest.py -k golden

        (Deliberately not PODAGENT_-prefixed: the env-isolation fixture strips
        those so a developer's shell cannot influence test results.)
        """
        import os

        seed_corpus(store)
        result = await run(settings, store)
        actual = result.file_path.read_text()  # type: ignore[union-attr]
        golden = Path(__file__).parent / "fixtures" / "digest_golden.md"

        if os.environ.get("REGENERATE_GOLDEN"):
            golden.write_text(actual)
            pytest.skip("golden file regenerated")

        assert actual == golden.read_text(), (
            "digest output changed; review the diff and regenerate the golden file "
            "if the change is intended"
        )


class TestWeeklyDigestIsThisWeekOnly:
    """A weekly digest must contain only what was published in its window.

    Two distinct ways historical material could leak in, so both are pinned:
    an episode ingested by the archive walk, and an episode whose publication
    date simply predates the period. The first is the one that bites — starting
    a backfill pulls in years of episodes, and if any of them reached the weekly
    digest, one click would bury the week's actual reading under a decade of
    back catalogue.
    """

    async def test_backfilled_episodes_never_enter_a_weekly_digest(
        self, settings, store: MemoryStore
    ) -> None:
        seed_corpus(store)
        # Published inside the window, fully summarised, high scoring — eligible
        # on every axis except that the archive walk is what ingested it.
        store.seed(
            make_episode(
                guid="from-archive",
                title="Archive episode that must not appear",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                tier1=tier1_block(10),
                origin=BACKFILL_ORIGIN,
            )
        )
        result = await run(settings, store)
        assert result.file_path is not None
        text = result.file_path.read_text()
        assert "Archive episode that must not appear" not in text
        assert "Deep dive: PLC malware" in text  # the week's own material still there

    async def test_a_backfill_started_midweek_does_not_change_the_digest(
        self, settings, store: MemoryStore, tmp_path: Path
    ) -> None:
        """The user-visible promise: running historical intake is digest-neutral.

        Two independent stores rather than a regenerate, so the comparison is
        between "the week alone" and "the week plus an archive haul" — not
        between a first and second pass over mutated documents.
        """
        seed_corpus(store)
        alone = (await run(settings, store)).file_path
        assert alone is not None
        baseline = alone.read_text()

        with_archive_store = MemoryStore()
        seed_corpus(with_archive_store)
        for i in range(5):
            with_archive_store.seed(
                make_episode(
                    guid=f"archive-{i}",
                    title=f"Archive episode {i}",
                    status=S.READY_FOR_DIGEST,
                    published_at=datetime(2026, 7, 27, tzinfo=UTC),
                    tier1=tier1_block(10),
                    origin=BACKFILL_ORIGIN,
                )
            )
        with_archive_settings = make_settings(tmp_path / "second")
        after = (await run(with_archive_settings, with_archive_store)).file_path
        assert after is not None
        assert after.read_text() == baseline

    async def test_episodes_published_before_the_window_are_left_out(
        self, settings, store: MemoryStore
    ) -> None:
        """Routine polling can pick up an old episode from a slow publisher."""
        seed_corpus(store)
        store.seed(
            make_episode(
                guid="old-but-normal",
                title="Ancient episode ingested normally",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2019, 3, 1, tzinfo=UTC),
                tier1=tier1_block(10),
            )
        )
        result = await run(settings, store)
        assert result.file_path is not None
        assert "Ancient episode ingested normally" not in result.file_path.read_text()

    async def test_an_episode_already_in_a_digest_is_not_listed_again(
        self, settings, store: MemoryStore
    ) -> None:
        """`digest_id` claims an episode exactly once, across every future run."""
        seed_corpus(store)
        store.seed(
            make_episode(
                guid="already-sent",
                title="Already in last week's digest",
                status=S.PUBLISHED,
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                tier1=tier1_block(10),
                digest_id="digest:2026-W30",
            )
        )
        result = await run(settings, store)
        assert result.file_path is not None
        assert "Already in last week's digest" not in result.file_path.read_text()


class TestLateEpisodesAreCaughtUp:
    """An episode finished after its week must still reach a digest.

    This used to be the opposite test, recorded as a deliberate consequence of
    newest-first ordering. It was a defect. Selection was bounded below by the
    window's own start, and each window starts where the last one ended, so an
    episode still being transcribed on Friday was behind the floor on Saturday
    and matched no window ever again — 70 of them, back to October, before
    anyone noticed. Claim-once is enforced by `digest_id`, so reaching further
    back cannot duplicate anything; the floor now only stops a fresh install
    emptying its whole initial ingest into digest one.
    """

    async def test_an_episode_finished_after_its_week_appears_in_the_next_digest(
        self, settings, store: MemoryStore
    ) -> None:
        seed_corpus(store)
        store.seed(
            make_episode(
                guid="stranded",
                title="Finished too late",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 20, tzinfo=UTC),  # before PERIOD_FROM
                tier1=tier1_block(10),
            )
        )
        result = await run(settings, store)
        assert result.file_path is not None
        assert "Finished too late" in result.file_path.read_text()

    async def test_it_is_marked_carried_over_rather_than_passed_off_as_new(
        self, settings, store: MemoryStore
    ) -> None:
        """A three-week-old episode arriving silently reads as a dating error."""
        store.seed(
            make_episode(
                guid="stranded",
                title="Finished too late",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 20, tzinfo=UTC),
                tier1=tier1_block(10),
            )
        )
        result = await run(settings, store)
        assert result.file_path is not None
        text = result.file_path.read_text()
        assert "carried over" in text
        assert "1 episode marked *carried over*" in text

    async def test_it_is_claimed_so_it_cannot_appear_twice(
        self, settings, store: MemoryStore
    ) -> None:
        store.seed(
            make_episode(
                guid="stranded",
                title="Finished too late",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 20, tzinfo=UTC),
                tier1=tier1_block(10),
            )
        )
        await run(settings, store)
        doc = next(d for d in store.docs_of_type("episode") if d["guid"] == "stranded")
        assert doc.get("digest_id") is not None
        assert doc["status"] == S.PUBLISHED.value

    async def test_the_handoff_between_two_consecutive_digests_loses_nothing(
        self, settings, store: MemoryStore
    ) -> None:
        """The case the old bound got wrong, run end to end.

        An episode is mid-pipeline when digest N runs, finishes afterwards, and
        must appear in digest N+1 — whose window starts where N's ended, i.e.
        already after the episode's publication date.
        """
        store.seed(
            make_episode(
                guid="slow",
                title="Still transcribing",
                status=S.AWAITING_TRANSCRIPT,
                published_at=datetime(2026, 7, 25, tzinfo=UTC),
                tier1=tier1_block(9),
            )
        )
        first = await run(settings, store, until=datetime(2026, 7, 26, tzinfo=UTC))
        assert first.file_path is not None
        assert "Still transcribing" not in first.file_path.read_text()

        # The transcript lands and it is summarised, after digest N.
        doc = next(d for d in store.docs_of_type("episode") if d["guid"] == "slow")
        doc["status"] = S.READY_FOR_DIGEST.value
        store.seed(doc)  # docs_of_type hands out copies

        generator = DigestGenerator(settings, store)
        second = await generator.generate(until=datetime(2026, 8, 1, tzinfo=UTC))
        assert second.file_path is not None
        assert "Still transcribing" in second.file_path.read_text()

    async def test_the_floor_still_keeps_ancient_history_out(
        self, settings, store: MemoryStore
    ) -> None:
        """Otherwise a fresh install's whole initial ingest lands in digest one."""
        store.seed(
            make_episode(
                guid="ancient",
                title="From another era",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2025, 1, 5, tzinfo=UTC),
                tier1=tier1_block(10),
            )
        )
        result = await run(settings, store)
        assert result.file_path is not None
        assert "From another era" not in result.file_path.read_text()

    async def test_widening_the_period_recovers_even_those(
        self, settings, store: MemoryStore
    ) -> None:
        """The recovery path for what was stranded before the floor was widened."""
        store.seed(
            make_episode(
                guid="ancient",
                title="From another era",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2025, 1, 5, tzinfo=UTC),
                tier1=tier1_block(10),
            )
        )
        result = await run(settings, store, since=datetime(2024, 12, 1, tzinfo=UTC))
        assert result.file_path is not None
        assert "From another era" in result.file_path.read_text()


class TestASecondRunInTheSameWeek:
    """Two files, one week, and both of them findable.

    The generator never overwrites: a second run writes `-r2` beside the first,
    so nothing already synced to a vault is rewritten. The document, though, is
    keyed by the ISO week — replacing it wholesale left the earlier file in the
    vault referenced by nothing and invisible to the console.
    """

    async def test_a_second_run_writes_a_second_file(self, settings, store: MemoryStore) -> None:
        seed_corpus(store)
        first = (await run(settings, store)).file_path
        assert first is not None

        store.seed(
            make_episode(
                guid="later",
                title="Published after the first digest",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 30, tzinfo=UTC),
                tier1=tier1_block(9),
            )
        )
        second = (await run(settings, store, since=datetime(2026, 7, 29, tzinfo=UTC))).file_path
        assert second is not None
        assert second != first
        assert second.name.endswith("-r2.md")
        assert first.exists(), "the first file must not be rewritten"

    async def test_both_runs_are_recorded_on_the_document(
        self, settings, store: MemoryStore
    ) -> None:
        """Otherwise the earlier file is orphaned — on disk, in no index."""
        seed_corpus(store)
        await run(settings, store)
        store.seed(
            make_episode(
                guid="later",
                title="Later",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 30, tzinfo=UTC),
                tier1=tier1_block(9),
            )
        )
        await run(settings, store, since=datetime(2026, 7, 29, tzinfo=UTC))

        doc = next(iter(store.docs_of_type("digest")))
        assert len(doc["runs"]) == 2
        assert doc["runs"][0]["file_path"].endswith("W31.md")
        assert doc["runs"][1]["file_path"].endswith("W31-r2.md")
        # The top level keeps describing the most recent run.
        assert doc["file_path"] == doc["runs"][-1]["file_path"]

    async def test_each_run_covers_only_its_own_period(self, settings, store: MemoryStore) -> None:
        """They are not two versions of one digest; they are two digests."""
        seed_corpus(store)
        await run(settings, store)
        store.seed(
            make_episode(
                guid="later",
                title="Later",
                status=S.READY_FOR_DIGEST,
                published_at=datetime(2026, 7, 30, tzinfo=UTC),
                tier1=tier1_block(9),
            )
        )
        await run(settings, store, since=datetime(2026, 7, 29, tzinfo=UTC))

        doc = next(iter(store.docs_of_type("digest")))
        first, second = doc["runs"]
        assert first["episode_ids"] != second["episode_ids"]
        assert set(first["episode_ids"]) & set(second["episode_ids"]) == set()


class TestAdoptingOrphanedDigestFiles:
    """Recovering a file written before runs were recorded."""

    async def test_a_file_on_disk_but_in_no_document_is_adopted(
        self, settings, store: MemoryStore, tmp_path: Path
    ) -> None:
        from podcast_agent.migrate import adopt_orphaned_digest_files

        seed_corpus(store)
        await run(settings, store)

        digest_dir = Path(settings.output.digest_dir)
        orphan = digest_dir / "2026" / "podcast-digest-2026-W31-r2.md"
        orphan.write_text(
            # Later than the run above, and written in local time with an
            # offset — as the generator writes it, and as the database does not.
            "---\nweek: 2026-W31\ngenerated: 2027-01-02T08:00:00+02:00\n"
            "period_from: 2026-07-30T13:10:03+02:00\nperiod_to: 2027-01-02T08:00:00+02:00\n"
            "episodes_scanned: 1\nepisodes_summarized: 1\n---\n\n# Later\n",
            encoding="utf-8",
        )

        result = await adopt_orphaned_digest_files(store, digest_dir)
        assert result == {"adopted": 1}

        doc = next(iter(store.docs_of_type("digest")))
        paths = [r["file_path"] for r in doc["runs"]]
        assert paths == ["2026/podcast-digest-2026-W31.md", "2026/podcast-digest-2026-W31-r2.md"]
        adopted = doc["runs"][1]
        assert adopted["adopted"] is True
        assert adopted["period"]["from"].startswith("2026-07-30")
        # Episode ids cannot be recovered, and are not guessed at.
        assert adopted["episode_ids"] == []

    async def test_it_is_idempotent(self, settings, store: MemoryStore) -> None:
        from podcast_agent.migrate import adopt_orphaned_digest_files

        seed_corpus(store)
        await run(settings, store)
        digest_dir = Path(settings.output.digest_dir)

        assert (await adopt_orphaned_digest_files(store, digest_dir))["adopted"] == 0
        assert (await adopt_orphaned_digest_files(store, digest_dir))["adopted"] == 0
        doc = next(iter(store.docs_of_type("digest")))
        assert len(doc.get("runs") or []) == 1
