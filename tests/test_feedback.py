"""Reader signals: starred, read, and the wrong-call verdict (roadmap B1).

Theme C is built entirely on these and cannot synthesise them. The pipeline
knows what it decided; only the reader knows whether the decision was right.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings

from podcast_agent.db import MemoryStore
from podcast_agent.feedback import set_read, set_starred, set_verdict
from podcast_agent.feedback import view as feedback_view
from podcast_agent.main import build_app
from podcast_agent.state import EpisodeStatus

S = EpisodeStatus
KEY = {"X-API-Key": "test-admin-key"}


@pytest.fixture
def episode(store: MemoryStore) -> dict[str, Any]:
    doc = make_episode(
        guid="e1",
        status=S.READY_FOR_DIGEST,
        tier1={"relevance_score": 8, "summary_md": "s", "interest_profile_version": "v-abc"},
    )
    store.seed(doc)
    return doc


class TestStarring:
    async def test_starring_records_when(self, store: MemoryStore, episode) -> None:
        updated = await set_starred(store, episode["_id"], True)
        assert updated["starred"] is True
        assert updated["starred_at"]

    async def test_unstarring_clears_the_timestamp(self, store: MemoryStore, episode) -> None:
        await set_starred(store, episode["_id"], True)
        updated = await set_starred(store, episode["_id"], False)
        assert updated["starred"] is False
        assert updated["starred_at"] is None


class TestReadMarking:
    async def test_read_is_a_timestamp_not_a_flag(self, store: MemoryStore, episode) -> None:
        """Time-to-read is the signal C1 wants; "have you read it" falls out."""
        updated = await set_read(store, episode["_id"], True)
        assert updated["read_at"]

    async def test_marking_unread_clears_it(self, store: MemoryStore, episode) -> None:
        await set_read(store, episode["_id"], True)
        assert (await set_read(store, episode["_id"], False))["read_at"] is None


class TestVerdict:
    async def test_it_records_the_direction(self, store: MemoryStore, episode) -> None:
        updated = await set_verdict(store, episode["_id"], "over", "too vendor-heavy")
        assert updated["feedback"]["verdict"] == "over"
        assert updated["feedback"]["note"] == "too vendor-heavy"
        assert updated["feedback"]["at"]

    async def test_it_captures_what_was_being_judged(self, store: MemoryStore, episode) -> None:
        """A verdict is only interpretable against the decision it disputes.

        Without this, a later re-score silently rewrites the thing being judged
        and the signal becomes unreadable.
        """
        updated = await set_verdict(store, episode["_id"], "under")
        judged = updated["feedback"]["judged"]
        assert judged["status"] == S.READY_FOR_DIGEST.value
        assert judged["relevance_score"] == 8
        assert judged["interest_profile_version"] == "v-abc"

    async def test_the_note_is_escaped(self, store: MemoryStore, episode) -> None:
        """Free text from a browser, rendered back into an HTML console."""
        updated = await set_verdict(store, episode["_id"], "over", "<script>alert(1)</script>")
        assert "<script>" not in updated["feedback"]["note"]

    async def test_it_can_be_cleared(self, store: MemoryStore, episode) -> None:
        await set_verdict(store, episode["_id"], "over")
        assert (await set_verdict(store, episode["_id"], None))["feedback"] is None

    async def test_both_directions_are_distinguishable(self, store: MemoryStore, episode) -> None:
        """`under` is the expensive kind — nobody sees what they were not shown."""
        assert (await set_verdict(store, episode["_id"], "under"))["feedback"]["verdict"] == "under"
        assert (await set_verdict(store, episode["_id"], "over"))["feedback"]["verdict"] == "over"

    def test_the_view_is_safe_on_an_untouched_episode(self) -> None:
        assert feedback_view({}) == {
            "starred": False,
            "starred_at": None,
            "read_at": None,
            "feedback": None,
        }


class TestApi:
    def _client(self, tmp_path, store: MemoryStore) -> TestClient:
        return TestClient(build_app(make_settings(tmp_path), store=store, llm=FakeLLM()))

    def _seed(self, store: MemoryStore, guid: str, **kw: Any) -> str:
        doc = make_episode(guid=guid, status=S.READY_FOR_DIGEST, tier1={"relevance_score": 8}, **kw)
        store.seed(doc)
        return doc["_id"]

    def test_starring_round_trips(self, tmp_path, store: MemoryStore) -> None:
        doc_id = self._seed(store, "a")
        with self._client(tmp_path, store) as client:
            assert client.post(f"/api/v1/episodes/{doc_id}/star", headers=KEY).status_code == 200
            body = client.get(f"/api/v1/episodes/{doc_id}", headers=KEY).json()
        assert body["starred"] is True

    def test_unstarring_takes_a_parameter(self, tmp_path, store: MemoryStore) -> None:
        doc_id = self._seed(store, "a")
        with self._client(tmp_path, store) as client:
            client.post(f"/api/v1/episodes/{doc_id}/star", headers=KEY)
            client.post(f"/api/v1/episodes/{doc_id}/star?starred=false", headers=KEY)
            body = client.get(f"/api/v1/episodes/{doc_id}", headers=KEY).json()
        assert body["starred"] is False

    def test_the_verdict_endpoint_validates_direction(self, tmp_path, store: MemoryStore) -> None:
        doc_id = self._seed(store, "a")
        with self._client(tmp_path, store) as client:
            bad = client.post(
                f"/api/v1/episodes/{doc_id}/feedback", headers=KEY, json={"verdict": "sideways"}
            )
        assert bad.status_code == 422

    def test_a_missing_episode_is_404_not_500(self, tmp_path, store: MemoryStore) -> None:
        with self._client(tmp_path, store) as client:
            assert client.post("/api/v1/episodes/nope/star", headers=KEY).status_code == 404
            assert client.post("/api/v1/episodes/nope/read", headers=KEY).status_code == 404

    def test_the_signals_need_the_key(self, tmp_path, store: MemoryStore) -> None:
        doc_id = self._seed(store, "a")
        with self._client(tmp_path, store) as client:
            assert client.post(f"/api/v1/episodes/{doc_id}/star").status_code == 401

    def test_the_list_view_carries_the_signals(self, tmp_path, store: MemoryStore) -> None:
        """A browsing surface must show what is starred without a request a row."""
        doc_id = self._seed(store, "a")
        with self._client(tmp_path, store) as client:
            client.post(f"/api/v1/episodes/{doc_id}/star", headers=KEY)
            listed = client.get("/api/v1/episodes", headers=KEY).json()["episodes"]
        assert listed[0]["starred"] is True

    def test_filtering_by_starred(self, tmp_path, store: MemoryStore) -> None:
        keep = self._seed(store, "keep")
        self._seed(store, "drop")
        with self._client(tmp_path, store) as client:
            client.post(f"/api/v1/episodes/{keep}/star", headers=KEY)
            body = client.get("/api/v1/episodes?starred=true", headers=KEY).json()
        assert [e["_id"] for e in body["episodes"]] == [keep]

    def test_filtering_by_unread(self, tmp_path, store: MemoryStore) -> None:
        read = self._seed(store, "read")
        unread = self._seed(store, "unread")
        with self._client(tmp_path, store) as client:
            client.post(f"/api/v1/episodes/{read}/read", headers=KEY)
            body = client.get("/api/v1/episodes?unread=true", headers=KEY).json()
        assert [e["_id"] for e in body["episodes"]] == [unread]

    def test_filtering_by_flagged(self, tmp_path, store: MemoryStore) -> None:
        flagged = self._seed(store, "flagged")
        self._seed(store, "fine")
        with self._client(tmp_path, store) as client:
            client.post(
                f"/api/v1/episodes/{flagged}/feedback", headers=KEY, json={"verdict": "over"}
            )
            body = client.get("/api/v1/episodes?flagged=true", headers=KEY).json()
        assert [e["_id"] for e in body["episodes"]] == [flagged]

    def test_a_signal_filter_alone_does_not_require_a_score(
        self, tmp_path, store: MemoryStore
    ) -> None:
        """Starring an unscored episode must not hide it from the starred list."""
        doc = make_episode(guid="unscored", status=S.NEW)
        store.seed(doc)
        with self._client(tmp_path, store) as client:
            client.post(f"/api/v1/episodes/{doc['_id']}/star", headers=KEY)
            body = client.get("/api/v1/episodes?starred=true", headers=KEY).json()
        assert [e["_id"] for e in body["episodes"]] == [doc["_id"]]


class TestConsole:
    """The signals have to be reachable without opening a drawer per episode."""

    def _page(self) -> str:
        from pathlib import Path

        return (Path(__file__).parent.parent / "podcast_agent/api/static/episodes.html").read_text()

    def test_the_star_toggles_from_the_row(self) -> None:
        page = self._page()
        assert "data-star=" in page
        # It must not also open the drawer.
        assert "ev.stopPropagation()" in page

    def test_both_verdict_directions_are_offered(self) -> None:
        page = self._page()
        assert 'id="dWrongOver"' in page
        assert 'id="dWrongUnder"' in page
        assert 'id="dWrongClear"' in page

    def test_the_signals_are_filterable(self) -> None:
        page = self._page()
        assert 'id="fSignal"' in page
        for param in ("starred", "unread", "flagged"):
            assert f'params.set("{param}", "true")' in page

    def test_the_column_count_matches_the_header(self) -> None:
        """An empty-state colspan that disagrees breaks the table layout."""
        import re

        page = self._page()
        head = page[page.index("<thead>") : page.index("</thead>")]
        # `<th>` and `<th ...>`, but not `<thead>`.
        columns = len(re.findall(r"<th[ >]", head))
        assert f'colspan="{columns}"' in page
