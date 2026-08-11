"""Weekly cross-episode synthesis (roadmap D1).

The section reads the week's summaries, not its transcripts, so it is one cheap
call however many episodes the week held. The rule that matters most here is
that it can never cost the reader their digest: the episode summaries are the
artefact and they are already written by the time this runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.db import MemoryStore
from podcast_agent.digest.generate import DigestGenerator
from podcast_agent.digest.synthesis import (
    MIN_EPISODES,
    WeeklySynthesizer,
    previous_theme_titles,
    select_episodes,
)
from podcast_agent.llm.base import LLMUnavailable
from podcast_agent.models import WeeklySynthesis, WeeklyTheme
from podcast_agent.state import EpisodeStatus

S = EpisodeStatus
PERIOD_FROM = datetime(2026, 7, 24, tzinfo=UTC)


def tier1(score: int, **kw: object) -> dict:
    block = {
        "relevance_score": score,
        "summary_md": "A summary of the discussion.",
        "why_it_matters": "It bears on OT segmentation.",
        "key_takeaways": ["Segment the network", "Patch the PLC"],
        "entities": ["Modbus"],
    }
    block.update(kw)
    return block


def summarised(guid: str, *, score: int = 8, **kw: object):
    return make_episode(
        guid=guid,
        status=S.READY_FOR_DIGEST,
        published_at=datetime(2026, 7, 28, tzinfo=UTC),
        tier1=tier1(score),
        **kw,
    )


SYNTHESIS = WeeklySynthesis(
    themes=[
        WeeklyTheme(
            title="Segmentation is losing",
            summary="Three shows argued flat OT networks are still the norm.",
            shows=["Risky Business", "Darknet Diaries"],
        )
    ],
    disagreements=["One show called SBOMs useless; another called them table stakes."],
    whats_new=["Nobody mentioned the Volt Typhoon story this week."],
)


def synth_llm(result=SYNTHESIS) -> FakeLLM:
    def handler(tier: str, system: str, user: str, model: type) -> object:
        if model is WeeklySynthesis:
            return result
        raise AssertionError(f"unexpected model {model}")

    return FakeLLM(handler)


class TestSelection:
    def test_only_episodes_with_a_real_summary_are_used(self) -> None:
        """A digest-direct one-liner is a guess from a feed description.

        Feeding those in would let the opening section assert things about
        episodes nobody read.
        """
        good = summarised("good")
        thin = make_episode(guid="thin", status=S.DIGEST_DIRECT, tier0={"relevance_guess": 7})
        assert [e["guid"] for e in select_episodes([thin, good])] == ["good"]

    def test_highest_scoring_first(self) -> None:
        low, high = summarised("low", score=5), summarised("high", score=10)
        assert [e["guid"] for e in select_episodes([low, high])] == ["high", "low"]

    def test_the_input_is_capped(self) -> None:
        """Thirty summaries fit a local model's window; a hundred do not."""
        many = [summarised(f"e{i}", score=(i % 10) + 1) for i in range(60)]
        assert len(select_episodes(many)) == 30

    def test_previous_themes_are_read_from_the_last_digest(self) -> None:
        previous = {"synthesis": {"themes": [{"title": "Old thread"}, {"title": ""}]}}
        assert previous_theme_titles(previous) == ["Old thread"]

    def test_no_previous_digest_is_not_an_error(self) -> None:
        assert previous_theme_titles(None) == []


class TestBuilding:
    async def _build(self, tmp_path: Path, episodes: list, llm=None, **kw):
        settings = make_settings(tmp_path, **kw)
        return await WeeklySynthesizer(settings, llm or synth_llm()).build(
            episodes, period_from="2026-07-24T00:00:00+00:00", period_to="2026-07-31T00:00:00+00:00"
        )

    async def test_it_returns_the_themes(self, tmp_path: Path) -> None:
        result = await self._build(tmp_path, [summarised(f"e{i}") for i in range(5)])
        assert result is not None
        assert result.themes[0].title == "Segmentation is losing"

    async def test_too_few_episodes_produces_nothing(self, tmp_path: Path) -> None:
        """Asking for three themes from two episodes invites invention."""
        episodes = [summarised(f"e{i}") for i in range(MIN_EPISODES - 1)]
        assert await self._build(tmp_path, episodes) is None

    async def test_the_summaries_reach_the_prompt(self, tmp_path: Path) -> None:
        seen: dict[str, str] = {}

        def handler(tier: str, system: str, user: str, model: type) -> object:
            seen["user"] = user
            seen["system"] = system
            return SYNTHESIS

        await self._build(tmp_path, [summarised(f"e{i}") for i in range(5)], FakeLLM(handler))
        assert "It bears on OT segmentation." in seen["user"]
        assert "Segment the network" in seen["user"]
        # The interest profile drives the weighting, so it has to be there.
        assert "ot_ics" in seen["system"]

    async def test_transcripts_are_never_sent(self, tmp_path: Path) -> None:
        """The whole economy of this pass is that it reads summaries."""
        seen: dict[str, str] = {}

        def handler(tier: str, system: str, user: str, model: type) -> object:
            seen["user"] = user
            return SYNTHESIS

        episodes = [summarised(f"e{i}") for i in range(5)]
        await self._build(tmp_path, episodes, FakeLLM(handler))
        assert "transcript" not in seen["user"].lower()


class TestItNeverCostsTheDigest:
    """Every failure mode is the same outcome: a digest without an opening."""

    async def _build(self, tmp_path: Path, llm):
        settings = make_settings(tmp_path)
        return await WeeklySynthesizer(settings, llm).build(
            [summarised(f"e{i}") for i in range(5)],
            period_from="a",
            period_to="b",
        )

    async def test_an_unavailable_model_returns_none(self, tmp_path: Path) -> None:
        def handler(*_a: object, **_k: object) -> object:
            raise LLMUnavailable("every endpoint failed")

        assert await self._build(tmp_path, FakeLLM(handler)) is None

    async def test_an_unexpected_error_returns_none(self, tmp_path: Path) -> None:
        def handler(*_a: object, **_k: object) -> object:
            raise RuntimeError("something else entirely")

        assert await self._build(tmp_path, FakeLLM(handler)) is None

    async def test_an_empty_response_returns_none(self, tmp_path: Path) -> None:
        """A model that produced nothing must not render an empty heading."""
        assert await self._build(tmp_path, synth_llm(WeeklySynthesis())) is None


class TestInTheDigest:
    async def _run(self, tmp_path: Path, store: MemoryStore, llm=None, **kw):
        settings = make_settings(tmp_path, **kw)
        return await DigestGenerator(settings, store, llm).generate(since=PERIOD_FROM)

    def _seed(self, store: MemoryStore, n: int = 5) -> None:
        store.seed(*[summarised(f"e{i}", score=9) for i in range(n)])

    async def test_the_section_is_rendered(self, tmp_path: Path, store: MemoryStore) -> None:
        self._seed(store)
        result = await self._run(tmp_path, store, synth_llm())
        assert result.file_path is not None
        text = result.file_path.read_text()
        assert "## Across the week" in text
        assert "Segmentation is losing" in text
        assert "Where they disagreed" in text
        assert "New since last week" in text

    async def test_without_an_llm_the_digest_is_unchanged(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Digest generation is a database read and a render, and stays that way."""
        self._seed(store)
        result = await self._run(tmp_path, store, None)
        assert result.file_path is not None
        assert "Across the week" not in result.file_path.read_text()

    async def test_the_toggle_switches_it_off(self, tmp_path: Path, store: MemoryStore) -> None:
        self._seed(store)
        result = await self._run(tmp_path, store, synth_llm(), pipeline={"weekly_synthesis": False})
        assert result.file_path is not None
        assert "Across the week" not in result.file_path.read_text()

    async def test_a_failing_model_still_produces_the_digest(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The one guarantee: the summaries are already written."""

        def handler(*_a: object, **_k: object) -> object:
            raise LLMUnavailable("down")

        self._seed(store)
        result = await self._run(tmp_path, store, FakeLLM(handler))
        assert result.file_path is not None
        text = result.file_path.read_text()
        assert "Across the week" not in text
        assert "# Podcast Digest" in text

    async def test_the_themes_are_stored_for_next_week(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        self._seed(store)
        await self._run(tmp_path, store, synth_llm())
        digest = next(iter(store.docs_of_type("digest")))
        assert digest["synthesis"]["themes"][0]["title"] == "Segmentation is losing"

    async def test_last_weeks_themes_are_offered_for_comparison(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """ "What's new" is meaningless without something to be new against."""
        seen: dict[str, str] = {}

        def handler(tier: str, system: str, user: str, model: type) -> object:
            seen["user"] = user
            return SYNTHESIS

        store.seed(
            {
                "_id": "digest:2026-W30",
                "type": "digest",
                "generated_at": "2026-07-24T00:00:00+00:00",
                "synthesis": {"themes": [{"title": "Last week's thread"}]},
            }
        )
        self._seed(store)
        await self._run(tmp_path, store, FakeLLM(handler))
        assert "Last week's thread" in seen["user"]


class TestPromptContract:
    @pytest.mark.parametrize("field", ["themes", "disagreements", "whats_new"])
    def test_every_output_field_is_specified(self, field: str) -> None:
        from podcast_agent.llm.prompts import load_prompt

        assert f"`{field}`" in load_prompt("digest_themes", "v1").system

    def test_the_summaries_are_declared_untrusted(self) -> None:
        """§10.2: they are model output over an automatic transcript."""
        from podcast_agent.llm.prompts import load_prompt

        assert "UNTRUSTED DATA" in load_prompt("digest_themes", "v1").system
