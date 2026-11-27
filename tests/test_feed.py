"""The private podcast feed: the one path served without the proxy's login."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_episode, make_settings
from pydantic import ValidationError

from podcast_agent.api.settings import _validate_against
from podcast_agent.db import MemoryStore
from podcast_agent.main import build_app
from podcast_agent.utils import digest_doc_id

TOKEN = "t0ken-" + "a" * 34
BASE = f"/podcast-feed/{TOKEN}"
AUDIO = bytes(range(256)) * 4


def _run(relative: str, generated_at: str = "2026-08-07T06:00:00Z") -> dict[str, Any]:
    return {"file_path": relative, "episode_ids": ["a", "b"], "generated_at": generated_at}


def _seed(store: MemoryStore, week: str, *runs: dict[str, Any]) -> None:
    store.seed(
        {
            "_id": digest_doc_id(week),
            "type": "digest",
            "runs": list(runs),
            "file_path": runs[-1]["file_path"],
            "generated_at": runs[-1]["generated_at"],
        }
    )


def _write(digest_dir: Path, relative: str, data: bytes = AUDIO) -> None:
    path = digest_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def digest_dir(tmp_path: Path) -> Path:
    return tmp_path / "digests"


@pytest.fixture
def client(tmp_path: Path, store: MemoryStore, digest_dir: Path) -> Iterator[TestClient]:
    _seed(store, "2026-W31", _run("2026/d-W31.md", "2026-07-31T06:00:00Z"))
    _seed(store, "2026-W32", _run("2026/d-W32.md"), _run("2026/d-W32-r2.md"))
    _seed(store, "2026-W33", _run("2026/d-W33.md", "2026-08-14T06:00:00Z"))
    # W33 has not been narrated; W32's regeneration has not been either.
    _write(digest_dir, "2026/d-W31.mp3")
    _write(digest_dir, "2026/d-W32.mp3")
    settings = make_settings(tmp_path, feed_token=TOKEN)
    with TestClient(build_app(settings, store=store, llm=FakeLLM())) as test_client:
        yield test_client


def _items(client: TestClient) -> list[ET.Element]:
    response = client.get(f"{BASE}/feed.xml")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/rss+xml")
    return ET.fromstring(response.content).findall("channel/item")


def test_feed_lists_narrated_weeks_newest_first(client: TestClient) -> None:
    items = _items(client)

    assert [i.findtext("title") for i in items] == ["Week 32, 2026", "Week 31, 2026"]
    enclosure = items[0].find("enclosure")
    assert enclosure is not None
    assert enclosure.attrib == {
        "url": f"http://testserver{BASE}/audio/d-W32.mp3",
        "length": str(len(AUDIO)),
        "type": "audio/mpeg",
    }
    assert items[1].findtext("pubDate") == "Fri, 31 Jul 2026 06:00:00 +0000"


def test_a_week_without_themes_keeps_the_plain_wording(client: TestClient) -> None:
    item = _items(client)[0]

    assert item.findtext("title") == "Week 32, 2026"
    assert item.findtext("description") == "<p>2 episodes from week 32 of 2026.</p>"


def test_show_notes_carry_themes_chapters_and_top_picks(
    client: TestClient, store: MemoryStore, digest_dir: Path
) -> None:
    _write(digest_dir, "2026/d-W33.mp3")
    pick = make_episode(
        guid="pick",
        title="Agents <script> loose",
        link="https://example.com/ep?a=1&b=2",
        tier1={
            "relevance_score": 9,
            "summary_md": "S.",
            "why_it_matters": r"Matters a lot (ai\_agent\_security).",
        },
    )
    store.seed(pick)
    run = {
        **_run("2026/d-W33.md", "2026-08-14T06:00:00Z"),
        "episode_ids": [pick["_id"]],
        "narration": {
            "audio_path": str(digest_dir / "2026/d-W33.mp3"),
            "duration_s": 3725.4,
            "chapters": [
                {"start_s": 0.0, "title": "Introduction"},
                {"start_s": 3661.9, "title": "Agents & identity"},
            ],
        },
    }
    _seed(store, "2026-W33", run)
    store._docs[digest_doc_id("2026-W33")]["synthesis"] = {
        "themes": [{"title": "Agent identity", "summary": "Everyone worried.", "shows": []}],
        "disagreements": [],
        "whats_new": ["A <new> thing."],
    }

    item = _items(client)[0]

    assert item.findtext("title") == "W33 · Agent identity"
    assert item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration") == "3725"
    notes = item.findtext("description") or ""
    assert notes.startswith("<p>Agent identity</p><p>00:00:00 Introduction</p>")
    assert "<p>01:01:01 Agents &amp; identity</p>" in notes
    assert 'href="https://example.com/ep?a=1&amp;b=2"' in notes
    assert "Matters a lot (ai_agent_security)." in notes
    assert "<script>" not in notes and "<new>" not in notes
    assert notes == item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded")


def test_chapters_of_other_audio_are_not_used(
    client: TestClient, store: MemoryStore, digest_dir: Path
) -> None:
    run = _run("2026/d-W31.md", "2026-07-31T06:00:00Z")
    run["narration"] = {
        "audio_path": "/elsewhere/d-W31.opus",
        "chapters": [{"start_s": 0.0, "title": "Stale chapter"}],
    }
    _seed(store, "2026-W31", run)

    assert "Stale chapter" not in (_items(client)[1].findtext("description") or "")


def test_cover_art_is_served_behind_the_token(client: TestClient) -> None:
    channel = ET.fromstring(client.get(f"{BASE}/feed.xml").content)
    image = channel.find("channel/{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
    assert image is not None

    response = client.get(image.attrib["href"])

    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")
    assert client.get(f"/podcast-feed/{'b' * 40}/podcast-cover.jpg").status_code == 404


def test_each_episode_has_its_own_cover(client: TestClient) -> None:
    urls = []
    for item in _items(client):
        image = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}image")
        assert image is not None
        urls.append(image.attrib["href"])
    assert len(set(urls)) == 2

    first, second = (client.get(url) for url in urls)
    assert first.headers["content-type"] == "image/jpeg"
    assert first.content != second.content
    assert client.head(urls[0]).status_code == 200
    assert client.get(urls[0].replace(TOKEN, "b" * 40)).status_code == 404
    assert client.get(f"{BASE}/cover/d-W32-0000000000.jpg").status_code == 404


def test_regenerated_week_replaces_audio_once_narrated(
    client: TestClient, digest_dir: Path
) -> None:
    _write(digest_dir, "2026/d-W32-r2.mp3")

    items = _items(client)

    assert len(items) == 2
    assert items[0].findtext("guid") == "d-W32-r2"


def test_audio_downloads_and_seeks(client: TestClient) -> None:
    url = f"{BASE}/audio/d-W31.mp3"

    whole = client.get(url)
    assert whole.content == AUDIO
    assert whole.headers["content-type"] == "audio/mpeg"

    part = client.get(url, headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.content == AUDIO[100:200]

    head = client.head(url)
    assert head.status_code == 200
    assert head.headers["content-length"] == str(len(AUDIO))
    assert client.head(f"{BASE}/feed.xml").status_code == 200


@pytest.mark.parametrize("name", ["d-W31.md", "d-W99.mp3", "..%2F..%2Fsecret.mp3"])
def test_only_listed_audio_is_served(client: TestClient, digest_dir: Path, name: str) -> None:
    _write(digest_dir, "2026/d-W31.md", b"notes")
    _write(digest_dir.parent, "secret.mp3")

    assert client.get(f"{BASE}/audio/{name}").status_code == 404


def test_path_outside_the_digest_directory_is_ignored(
    client: TestClient, store: MemoryStore, digest_dir: Path
) -> None:
    _write(digest_dir.parent, "outside.mp3")
    _seed(store, "2026-W40", _run("../outside.md"))

    assert len(_items(client)) == 2
    assert client.get(f"{BASE}/audio/outside.mp3").status_code == 404


@pytest.mark.parametrize("path", ["feed.xml", "audio/d-W31.mp3"])
def test_wrong_token_is_not_found(client: TestClient, path: str) -> None:
    assert client.get(f"/podcast-feed/{'b' * 40}/{path}").status_code == 404


def test_console_can_find_the_address(client: TestClient) -> None:
    response = client.get("/api/v1/feed")

    assert response.json() == {"path": f"{BASE}/feed.xml"}
    assert response.headers["cache-control"] == "no-store"
    assert client.get(response.json()["path"]).status_code == 200


def test_the_address_is_not_served_under_the_open_prefix(client: TestClient) -> None:
    assert client.get("/podcast-feed/feed").status_code == 404
    assert client.get(f"{BASE}/feed").status_code == 404


def test_console_is_told_when_the_feed_is_off(tmp_path: Path, store: MemoryStore) -> None:
    settings = make_settings(tmp_path)
    with TestClient(build_app(settings, store=store, llm=FakeLLM())) as client:
        assert client.get("/api/v1/feed").json() == {"path": None}


@pytest.mark.parametrize("unset", [None, ""])
def test_feed_is_off_without_a_token(tmp_path: Path, store: MemoryStore, unset: str | None) -> None:
    settings = make_settings(tmp_path, feed_token=unset)
    with TestClient(build_app(settings, store=store, llm=FakeLLM())) as client:
        assert client.get(f"{BASE}/feed.xml").status_code == 404


def test_weak_token_is_refused_without_being_echoed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as caught:
        make_settings(tmp_path, feed_token="guessable-secret")

    assert "at least 32" in str(caught.value)
    assert "guessable-secret" not in str(caught.value)


def test_console_settings_still_validate_with_a_token_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A masked secret in the round-trip would fail the length rule."""
    monkeypatch.setenv("PODAGENT_FEED_TOKEN", TOKEN)
    settings = make_settings(tmp_path)

    _validate_against(settings, {}, baseline=settings)
