"""Content seeds (roadmap E3).

The digest answers "what happened". This answers "is there anything here I
should say something about?" — from the same summaries, at the cost of one call.

The tests that matter most are about restraint: it is off until asked, it drops
an angle it cannot attribute, and an empty answer is a real answer rather than a
failure to pad around.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.content import (
    OUTPUT_FILENAME,
    ContentSeedBuilder,
    render,
    select,
    write,
)
from podcast_agent.db import MemoryStore
from podcast_agent.llm.base import LLMUnavailable
from podcast_agent.main import build_app
from podcast_agent.models import ContentSeed, ContentSeeds, ContentThread
from podcast_agent.state import EpisodeStatus

S = EpisodeStatus
KEY = {"X-API-Key": "test-admin-key"}
RECENT = datetime.now(UTC) - timedelta(days=3)


def episode(
    guid: str,
    *,
    score: int = 8,
    interests: list[str] | None = None,
    status: EpisodeStatus = S.READY_FOR_DIGEST,
    days_ago: int = 3,
    **extra: Any,
):
    return make_episode(
        guid=guid,
        status=status,
        published_at=datetime.now(UTC) - timedelta(days=days_ago),
        tier1={
            "relevance_score": score,
            "summary_md": "A summary.",
            "why_it_matters": "It bears on segmentation.",
            "key_takeaways": ["Segment the network"],
            "entities": ["Modbus"],
            "matched_interests": interests or ["ot_ics"],
        },
        **extra,
    )


def content_settings(tmp_path, **over: Any):
    return make_settings(tmp_path, content={"enabled": True, "interests": ["ot_ics"], **over})


SEEDS = ContentSeeds(
    seeds=[
        ContentSeed(
            ref=1,
            angle="Everyone says flat OT networks are legacy; this argues they are being built new.",
            why_now="NIS2 deadlines land in October.",
            contrarian=True,
        )
    ],
    threads=[
        ContentThread(
            title="Segmentation theatre",
            argument="Three shows describe segmentation projects that changed nothing.",
            refs=[1],
        )
    ],
)


def seed_llm(result=SEEDS) -> FakeLLM:
    def handler(tier: str, system: str, user: str, model: type) -> object:
        assert model is ContentSeeds
        return result

    return FakeLLM(handler)


class TestSelection:
    async def test_the_score_floor_is_higher_than_the_digests(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """An episode has to be worth your time twice: to read, then to write about."""
        settings = content_settings(tmp_path, min_score=7)
        store.seed(episode("good", score=8), episode("marginal", score=6))
        assert [e["guid"] for e in await select(store, settings)] == ["good"]

    async def test_only_the_configured_interests_qualify(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Narrowing is the point — an unread list is worth nothing."""
        settings = content_settings(tmp_path, interests=["ot_ics"])
        store.seed(
            episode("keep", interests=["ot_ics"]),
            episode("drop", interests=["leadership_policy"]),
        )
        assert [e["guid"] for e in await select(store, settings)] == ["keep"]

    async def test_no_interests_configured_means_any(self, tmp_path, store: MemoryStore) -> None:
        settings = content_settings(tmp_path, interests=[])
        store.seed(episode("a", interests=["leadership_policy"]))
        assert len(await select(store, settings)) == 1

    async def test_unsummarised_episodes_are_never_candidates(
        self, tmp_path, store: MemoryStore
    ) -> None:
        settings = content_settings(tmp_path)
        store.seed(
            make_episode(guid="direct", status=S.DIGEST_DIRECT, published_at=RECENT),
            make_episode(guid="dropped", status=S.DROPPED, published_at=RECENT),
        )
        assert await select(store, settings) == []

    async def test_the_window_is_respected(self, tmp_path, store: MemoryStore) -> None:
        settings = content_settings(tmp_path, window_days=30)
        store.seed(episode("recent", days_ago=3), episode("old", days_ago=90))
        assert [e["guid"] for e in await select(store, settings)] == ["recent"]

    async def test_the_batch_is_capped(self, tmp_path, store: MemoryStore) -> None:
        settings = content_settings(tmp_path, max_episodes=3)
        store.seed(*[episode(f"e{i}") for i in range(10)])
        assert len(await select(store, settings)) == 3

    async def test_highest_scoring_first(self, tmp_path, store: MemoryStore) -> None:
        settings = content_settings(tmp_path)
        store.seed(episode("low", score=7), episode("high", score=10))
        assert next(e["guid"] for e in await select(store, settings)) == "high"


class TestBuilding:
    async def test_it_returns_seeds_and_their_episodes(self, tmp_path, store: MemoryStore) -> None:
        store.seed(episode("a"))
        seeds, episodes = await ContentSeedBuilder(
            content_settings(tmp_path), store, seed_llm()
        ).build()
        assert seeds is not None
        assert len(episodes) == 1

    async def test_summaries_reach_the_prompt_but_transcripts_do_not(
        self, tmp_path, store: MemoryStore
    ) -> None:
        seen: dict[str, str] = {}

        def handler(tier: str, system: str, user: str, model: type) -> object:
            seen["user"], seen["system"] = user, system
            return SEEDS

        store.seed(episode("a"))
        await ContentSeedBuilder(content_settings(tmp_path), store, FakeLLM(handler)).build()
        assert "Segment the network" in seen["user"]
        assert "transcript" not in seen["user"].lower()
        assert "ot_ics" in seen["system"]

    async def test_episodes_are_numbered_for_reference(self, tmp_path, store: MemoryStore) -> None:
        """A model asked to echo a title paraphrases it, and then nothing can
        be traced back to the episode it came from."""
        seen: dict[str, str] = {}

        def handler(tier: str, system: str, user: str, model: type) -> object:
            seen["user"] = user
            return SEEDS

        store.seed(episode("a"), episode("b"))
        await ContentSeedBuilder(content_settings(tmp_path), store, FakeLLM(handler)).build()
        assert "1. show:" in seen["user"]
        assert "2. show:" in seen["user"]

    async def test_no_candidates_is_not_an_error(self, tmp_path, store: MemoryStore) -> None:
        seeds, episodes = await ContentSeedBuilder(
            content_settings(tmp_path), store, seed_llm()
        ).build()
        assert seeds is None and episodes == []

    async def test_an_unavailable_model_costs_a_file_nobody_waited_for(
        self, tmp_path, store: MemoryStore
    ) -> None:
        def handler(*_a: object, **_k: object) -> object:
            raise LLMUnavailable("down")

        store.seed(episode("a"))
        seeds, episodes = await ContentSeedBuilder(
            content_settings(tmp_path), store, FakeLLM(handler)
        ).build()
        assert seeds is None
        assert len(episodes) == 1

    async def test_an_unexpected_error_is_swallowed(self, tmp_path, store: MemoryStore) -> None:
        def handler(*_a: object, **_k: object) -> object:
            raise RuntimeError("something else")

        store.seed(episode("a"))
        seeds, _ = await ContentSeedBuilder(
            content_settings(tmp_path), store, FakeLLM(handler)
        ).build()
        assert seeds is None


class TestRendering:
    def _render(self, tmp_path, seeds: ContentSeeds, episodes: list) -> str:
        return render(seeds, episodes, content_settings(tmp_path))

    def test_a_seed_names_the_episode_it_came_from(self, tmp_path) -> None:
        """An angle whose source cannot be named is the thing this must not produce."""
        text = self._render(tmp_path, SEEDS, [episode("a", title="Deep dive on PLCs")])
        assert "Deep dive on PLCs" in text
        assert "flat OT networks" in text

    def test_a_contrarian_seed_is_marked(self, tmp_path) -> None:
        assert "against the grain" in self._render(tmp_path, SEEDS, [episode("a")])

    def test_threads_list_their_episodes(self, tmp_path) -> None:
        text = self._render(tmp_path, SEEDS, [episode("a", title="PLC episode")])
        assert "Segmentation theatre" in text
        assert text.index("Segmentation theatre") < text.index("Single-episode angles")

    def test_a_seed_referencing_an_unknown_episode_is_dropped(self, tmp_path) -> None:
        """Rather than rendered without a source."""
        stray = ContentSeeds(seeds=[ContentSeed(ref=99, angle="From nowhere.")])
        text = self._render(tmp_path, stray, [episode("a")])
        assert "From nowhere." not in text

    def test_an_empty_answer_says_so_plainly(self, tmp_path) -> None:
        """A month with no opening is ordinary, not a failure to pad around."""
        text = self._render(tmp_path, ContentSeeds(), [episode("a")])
        assert "Nothing this period offered an opening" in text
        assert "ordinary outcome" in text

    def test_it_says_these_are_not_drafts(self, tmp_path) -> None:
        text = self._render(tmp_path, SEEDS, [episode("a")])
        assert "not drafts" in text
        assert "check the claim before you publish" in text

    def test_the_frontmatter_records_what_it_considered(self, tmp_path) -> None:
        text = self._render(tmp_path, SEEDS, [episode("a")])
        assert "type: content-seeds" in text
        assert "episodes_considered: 1" in text

    def test_a_hostile_title_cannot_become_a_link(self, tmp_path) -> None:
        """Titles come from feeds (§10.2)."""
        text = self._render(tmp_path, SEEDS, [episode("a", title="Evil](http://x) [click")])
        cite = next(line for line in text.splitlines() if "Evil" in line)
        # Exactly one unescaped `](` — the genuine link close at the end. The
        # title's own brackets are escaped, so they cannot terminate the link
        # early and turn the rest of it into text the reader can click.
        assert cite.count("](") - cite.count("\\](") == 1
        assert "\\[" in cite

    def test_writing_never_overwrites(self, tmp_path) -> None:
        settings = content_settings(tmp_path)
        first = write(settings, "one")
        second = write(settings, "two")
        assert first.name == OUTPUT_FILENAME
        assert second.name != first.name
        assert first.read_text() == "one"


class TestApi:
    def _client(self, tmp_path, store: MemoryStore, **over: Any) -> TestClient:
        return TestClient(
            build_app(content_settings(tmp_path, **over), store=store, llm=seed_llm())
        )

    def test_it_needs_the_key(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.get("/api/v1/content/seeds").status_code == 401

    def test_the_preview_spends_no_call(self, tmp_path, store: MemoryStore) -> None:
        calls = {"n": 0}

        def handler(*_a: object, **_k: object) -> object:
            calls["n"] += 1
            return SEEDS

        store.seed(episode("a"))
        app = build_app(content_settings(tmp_path), store=store, llm=FakeLLM(handler))
        with TestClient(app) as client:
            body = client.get("/api/v1/content/seeds", headers=KEY).json()
        assert body["count"] == 1
        assert calls["n"] == 0

    def test_generating_writes_the_file(self, tmp_path, store: MemoryStore) -> None:
        store.seed(episode("a"))
        with self._client(tmp_path, store) as client:
            body = client.post("/api/v1/content/seeds", headers=KEY).json()
        assert body["seeds"] == 1
        assert body["file_path"] == OUTPUT_FILENAME

    def test_it_is_off_until_asked(self, tmp_path, store: MemoryStore) -> None:
        """A system that suggests what to post unbidden is presumptuous."""
        settings = make_settings(tmp_path)
        assert settings.content.enabled is False
        store.seed(episode("a"))
        app = build_app(settings, store=store, llm=seed_llm())
        with TestClient(app) as client:
            response = client.post("/api/v1/content/seeds", headers=KEY)
        assert response.status_code == 409
        assert "switched off" in response.json()["detail"]

    def test_no_candidates_is_reported_rather_than_an_empty_file(
        self, tmp_path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            response = client.post("/api/v1/content/seeds", headers=KEY)
        assert response.status_code == 503
        assert "no candidate episodes" in response.json()["detail"]


class TestPromptContract:
    @pytest.mark.parametrize("field", ["seeds", "threads", "angle", "why_now", "contrarian"])
    def test_every_output_field_is_specified(self, field: str) -> None:
        from podcast_agent.llm.prompts import load_prompt

        assert f"`{field}`" in load_prompt("content_seeds", "v1").system

    def test_it_is_told_to_skip_rather_than_pad(self) -> None:
        """The failure mode is fifteen mediocre angles, not too few."""
        from podcast_agent.llm.prompts import load_prompt

        assert "Skip freely" in load_prompt("content_seeds", "v1").system

    def test_it_is_forbidden_from_inventing_facts(self) -> None:
        from podcast_agent.llm.prompts import load_prompt

        system = load_prompt("content_seeds", "v1").system
        assert "Never invent" in system
        assert "UNTRUSTED DATA" in system

    def test_it_is_told_not_to_write_the_post(self) -> None:
        """The angle is the argument, not its packaging."""
        from podcast_agent.llm.prompts import load_prompt

        assert "Do not write the post" in load_prompt("content_seeds", "v1").system


class TestConsole:
    def _page(self) -> str:
        return (Path(__file__).parent.parent / "podcast_agent/api/static/insights.html").read_text()

    def test_the_generate_button_exists(self) -> None:
        page = self._page()
        assert 'id="seedBuild"' in page
        assert 'id="seedState"' in page

    def test_the_preview_loads_but_generating_needs_a_click(self) -> None:
        """The preview costs nothing; a model call should not fire on page load."""
        page = self._page()
        assert "loadSeeds()" in page
        assert 'api("/api/v1/content/seeds", { method: "POST" })' in page
        # The POST only appears inside the click handler, never in load().
        load_body = page[page.index("function load()") : page.index("function load()") + 200]
        assert "buildSeeds" not in load_body

    def test_it_says_where_the_file_goes(self) -> None:
        assert "content-seeds.md" in self._page()

    def test_it_says_the_feature_is_off_by_default(self) -> None:
        assert "content.enabled" in self._page()
