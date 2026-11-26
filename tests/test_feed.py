"""The private podcast feed: the one path served without the proxy's login."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_settings
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
