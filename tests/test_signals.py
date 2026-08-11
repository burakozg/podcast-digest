"""Reader marks mirrored into the vault (stars and wrong-call flags).

The file exists to be read by something that cannot query the database, so most
of these tests are about what it says rather than what it contains — in
particular that it does not let an absent episode be mistaken for a rejected
one, which is the way this signal is easiest to misread.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.db import MemoryStore
from podcast_agent.main import build_app
from podcast_agent.signals import (
    OUTPUT_DIR,
    collect,
    cursor,
    export_new_marks,
    period_name,
    render,
)
from podcast_agent.state import EpisodeStatus

S = EpisodeStatus
KEY = {"X-API-Key": "test-admin-key"}


def marked(
    guid: str,
    *,
    starred: bool = False,
    verdict: str | None = None,
    status: EpisodeStatus = S.PUBLISHED,
    score: int = 8,
    at: str = "2026-08-02T12:00:00+00:00",
    **extra: Any,
) -> dict[str, Any]:
    doc = make_episode(
        guid=guid,
        title=guid,
        status=status,
        published_at=datetime(2026, 7, 1, tzinfo=UTC),
        tier1={
            "relevance_score": score,
            "summary_md": "s",
            "why_it_matters": f"why {guid} matters",
            "matched_interests": ["ot_ics"],
        },
        **extra,
    )
    doc["starred"] = starred
    if starred:
        doc["starred_at"] = at
    if verdict:
        doc["feedback"] = {"verdict": verdict, "at": at, "judged": {"relevance_score": score}}
    return doc


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


class TestWhatIsCollected:
    async def test_stars_flags_and_verdicts_are_separated(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(
            marked("liked", starred=True),
            marked("too-low", verdict="under"),
            marked("not-worth-it", verdict="over"),
        )
        marks = await collect(store, since=None)
        assert [d["title"] for d in marks["starred"]] == ["liked"]
        assert [d["title"] for d in marks["under"]] == ["too-low"]
        assert [d["title"] for d in marks["over"]] == ["not-worth-it"]

    async def test_an_unmarked_episode_appears_nowhere(self, store: MemoryStore, settings) -> None:
        store.seed(marked("ordinary"))
        marks = await collect(store, since=None)
        assert not any(marks.values())

    async def test_a_star_on_something_never_surfaced_is_ignored(
        self, store: MemoryStore, settings
    ) -> None:
        """Starring a dropped episode says something, but not about a ranking it
        never received — and the file's denominator is what was offered."""
        store.seed(marked("dropped", starred=True, status=S.DROPPED))
        assert (await collect(store, since=None))["starred"] == []


class TestWhatTheFileSays:
    async def _body(self, store: MemoryStore, settings) -> str:
        return render(
            await collect(store, since=None),
            since=None,
            until="2026-08-03T00:00:00+00:00",
            surfaced=40,
            settings=settings,
        )

    async def test_silence_is_explicitly_not_a_rejection(
        self, store: MemoryStore, settings
    ) -> None:
        """The single most misreadable thing about this data.

        Most good episodes are never starred. A model that infers dislike from
        absence would learn the opposite of the truth, and it will not have read
        the module docstring.
        """
        body = await self._body(store, settings)
        assert "not a rejection" in body
        assert "inverts the signal" in body

    async def test_the_denominator_is_stated(self, store: MemoryStore, settings) -> None:
        """ "12 starred" means nothing without "of how many"."""
        store.seed(marked("liked", starred=True))
        body = await self._body(store, settings)
        assert "offered 40 episodes" in body

    async def test_a_disagreement_carries_both_numbers(self, store: MemoryStore, settings) -> None:
        """The reader's verdict against what the pipeline thought — one without
        the other is half a signal."""
        store.seed(marked("too-low", verdict="under", score=3))
        body = await self._body(store, settings)
        assert "had scored this 3/10" in body

    async def test_empty_sections_say_so(self, store: MemoryStore, settings) -> None:
        body = await self._body(store, settings)
        assert body.count("Nothing marked this way in this period") == 3

    async def test_a_hostile_title_cannot_restructure_the_file(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(marked("# Injected heading\n## another", starred=True))
        body = await self._body(store, settings)
        assert "\n# Injected" not in body
        assert "\n## another" not in body

    async def test_frontmatter_carries_the_counts(self, store: MemoryStore, settings) -> None:
        """So a reader can filter on them without parsing the prose."""
        store.seed(marked("liked", starred=True), marked("bad", verdict="over"))
        body = await self._body(store, settings)
        assert "starred: 1" in body
        assert "not_worth_it: 1" in body


class TestOnlyWhatIsNew:
    """Each run carries its own period, not the whole history again.

    Marks accumulate a handful a week. A full snapshot rewritten weekly would be
    fifty near-identical files, and a reader could not tell which lines were new.
    """

    async def test_the_first_run_carries_the_backlog(self, store: MemoryStore, settings) -> None:
        store.seed(marked("old", starred=True, at="2026-01-01T00:00:00+00:00"))
        result = await export_new_marks(store, settings)
        assert result["marks"] == 1

    async def test_a_second_run_writes_nothing_when_nothing_changed(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(marked("liked", starred=True))
        await export_new_marks(store, settings)
        again = await export_new_marks(store, settings)

        assert again["marks"] == 0
        assert again["written"] is None
        files = list((settings.output.digest_dir / OUTPUT_DIR).glob("*.md"))
        assert len(files) == 1, "a second file would repeat the same marks"

    async def test_only_the_new_mark_appears_in_the_next_file(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(marked("first", starred=True, at="2026-08-01T00:00:00+00:00"))
        await export_new_marks(store, settings)

        store.seed(marked("second", starred=True, at="2026-08-09T00:00:00+00:00"))
        result = await export_new_marks(store, settings)

        assert result["marks"] == 1
        body = (settings.output.digest_dir / result["written"]).read_text()
        assert "second" in body
        assert "first" not in body, "already reported in the period it was made"

    async def test_the_cursor_moves_even_when_nothing_was_marked(
        self, store: MemoryStore, settings
    ) -> None:
        """Otherwise each empty run re-asks the same question over an
        ever-widening window."""
        await export_new_marks(store, settings)
        assert await cursor(store) is not None

    async def test_a_run_that_was_missed_is_not_lost(self, store: MemoryStore, settings) -> None:
        """The cursor is kept rather than the calendar trusted: a week the job
        did not run must still be picked up by the next one."""
        store.seed(
            marked("week-one", starred=True, at="2026-07-06T00:00:00+00:00"),
            marked("week-two", starred=True, at="2026-07-13T00:00:00+00:00"),
        )
        result = await export_new_marks(store, settings)
        body = (settings.output.digest_dir / result["written"]).read_text()
        assert "week-one" in body and "week-two" in body

    async def test_files_are_named_for_the_period_and_never_rewritten(
        self, store: MemoryStore, settings
    ) -> None:
        store.seed(marked("liked", starred=True))
        result = await export_new_marks(store, settings)
        assert result["written"] == f"{OUTPUT_DIR}/{period_name(settings)}.md"


class TestApi:
    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def test_it_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.post("/api/v1/signals/export").status_code == 401

    def test_it_reports_what_it_wrote(self, tmp_path, store: MemoryStore) -> None:
        store.seed(marked("liked", starred=True), marked("bad", verdict="over"))
        with self._client(tmp_path, store) as client:
            body = client.post("/api/v1/signals/export", headers=KEY).json()
        assert body["written"].startswith(f"{OUTPUT_DIR}/")
        assert body["starred"] == 1
        assert body["not_worth_it"] == 1

    def test_forcing_it_twice_does_not_repeat_a_mark(self, tmp_path, store: MemoryStore) -> None:
        store.seed(marked("liked", starred=True))
        with self._client(tmp_path, store) as client:
            client.post("/api/v1/signals/export", headers=KEY)
            again = client.post("/api/v1/signals/export", headers=KEY).json()
        assert again["marks"] == 0
