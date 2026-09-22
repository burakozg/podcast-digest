"""The one vault note this application reads back.

The tests that matter here are the ones about *direction*: a tick has to reach
the database, a box nobody touched must not overwrite what the console did, and
the reader's own writing in that note has to survive being re-rendered. The rest
is rendering, which is cosmetic by comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from helpers import make_settings

from podcast_agent.config import Settings, VaultConfig
from podcast_agent.db import MemoryStore
from podcast_agent.entities import EPISODE_NAMES_DOC_ID
from podcast_agent.reading_queue import (
    Item,
    parse,
    render,
    sync_reading_queue,
)
from podcast_agent.vault import LiveSyncVault, retire_moved

QUEUE = "11 podcasts/_unread.md"


class FakeVault(LiveSyncVault):
    """The real path routing, with the two note operations held in memory.

    Subclassed rather than stubbed so `vault_path` stays the one the projection
    uses: a queue linking to a path the episode notes are not written to would
    pass a test built on a second guess at that routing and dangle in the vault.
    """

    def __init__(self, cfg: VaultConfig, note: str | None = None) -> None:
        super().__init__(cfg, "secret")
        self.notes: dict[str, str] = {cfg.queue_note: note} if note is not None else {}
        self.writes: list[str] = []

    async def read_note(self, path: str) -> str | None:
        return self.notes.get(path)

    async def write_note(
        self, path: str, markdown: str, *, mtime_ms: int, merge: bool = False
    ) -> str | None:
        self.notes[path] = markdown
        self.writes.append(markdown)
        return path


def _settings(tmp_path: Path, **vault: Any) -> Settings:
    return make_settings(
        tmp_path,
        output={"digest_dir": tmp_path / "digests", "work_dir": tmp_path / "work"},
        vault={
            "enabled": True,
            "couchdb_url": "http://vault-couch.lan:5984",
            "db": "myvault",
            "folder": "11 podcasts/digests",
            "episodes_folder": "11 podcasts/episodes",
            "queue_note": QUEUE,
            **vault,
        },
    )


def _vault(settings: Settings, note: str | None = None) -> FakeVault:
    return FakeVault(settings.vault, note)


async def _seed(
    store: MemoryStore,
    *,
    episode_id: str = "episode:test-show:one",
    title: str = "An episode",
    read_at: str | None = None,
    starred: bool = False,
    imported: bool = False,
    summary: str = "Some summary.",
    note_name: str = "2026-09-18-an-episode",
) -> str:
    doc: dict[str, Any] = {
        "_id": episode_id,
        "type": "episode",
        "title": title,
        "podcast_slug": "video-digest" if imported else "test-show",
        "podcast_name": "Video Digest" if imported else "Test Show",
        "published_at": "2026-09-18T00:00:00+00:00",
        "tier1": {"summary_md": summary} if summary else {},
        "read_at": read_at,
        "starred": starred,
    }
    if imported:
        doc["origin"] = "imported"
        doc["source_note_path"] = f"13 video-summaries/{note_name}.md"
    else:
        doc["origin"] = "routine"
        names = (await store.get(EPISODE_NAMES_DOC_ID)) or {
            "_id": EPISODE_NAMES_DOC_ID,
            "names": {},
        }
        names["names"][episode_id] = note_name
        await store.put(names)
    await store.put(doc)
    return episode_id


def _note_path(imported: bool = False, note_name: str = "2026-09-18-an-episode") -> str:
    if imported:
        return f"13 video-summaries/{note_name}.md"
    return f"11 podcasts/episodes/Test Show/{note_name}.md"


def _ticked(markdown: str, path: str, *, read: bool | None = None, star: bool | None = None) -> str:
    """The note as a phone would leave it after tapping a box."""
    out = []
    current = None
    for line in markdown.splitlines():
        if path.lower() in line.lower():
            current = path
            if read is not None:
                line = line.replace("- [ ]", f"- [{'x' if read else ' '}]", 1).replace(
                    "- [x]", f"- [{'x' if read else ' '}]", 1
                )
        elif current and "⭐" in line:
            if star is not None:
                line = line.replace("- [ ]", f"- [{'x' if star else ' '}]", 1).replace(
                    "- [x]", f"- [{'x' if star else ' '}]", 1
                )
            current = None
        out.append(line)
    return "\n".join(out) + "\n"


# --- taps reaching the database ----------------------------------------------


class TestWhatTheReaderTapped:
    @pytest.mark.asyncio
    async def test_a_ticked_box_marks_the_episode_read(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        episode_id = await _seed(store)
        vault = _vault(settings)

        # First pass writes the queue and records what each box said.
        await sync_reading_queue(store, settings, vault)
        assert (await store.get(episode_id))["read_at"] is None

        vault.notes[QUEUE] = _ticked(vault.notes[QUEUE], _note_path(), read=True)
        result = await sync_reading_queue(store, settings, vault)

        assert (await store.get(episode_id))["read_at"] is not None
        assert result["applied"] == [{"episode_id": episode_id, "read": True}]
        assert result["unread_total"] == 0

    @pytest.mark.asyncio
    async def test_unticking_under_recently_read_puts_it_back(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        episode_id = await _seed(store)
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)
        vault.notes[QUEUE] = _ticked(vault.notes[QUEUE], _note_path(), read=True)
        await sync_reading_queue(store, settings, vault)
        assert "Recently read" in vault.notes[QUEUE]

        vault.notes[QUEUE] = _ticked(vault.notes[QUEUE], _note_path(), read=False)
        await sync_reading_queue(store, settings, vault)

        assert (await store.get(episode_id))["read_at"] is None

    @pytest.mark.asyncio
    async def test_the_indented_box_stars_and_unstars(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        episode_id = await _seed(store)
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)
        vault.notes[QUEUE] = _ticked(vault.notes[QUEUE], _note_path(), star=True)
        await sync_reading_queue(store, settings, vault)
        assert (await store.get(episode_id))["starred"] is True

        vault.notes[QUEUE] = _ticked(vault.notes[QUEUE], _note_path(), star=False)
        await sync_reading_queue(store, settings, vault)
        assert (await store.get(episode_id))["starred"] is False

    @pytest.mark.asyncio
    async def test_a_box_nobody_touched_does_not_undo_the_console(self, tmp_path: Path) -> None:
        """The failure this whole design turns on.

        The note still says unread because that is what it said when it was
        written; the console has marked it read since. An implementation that
        treated the note as the source of truth would silently unread it.
        """
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        episode_id = await _seed(store)
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)
        # As the console does it, without touching the note.
        doc = await store.get(episode_id)
        doc["read_at"] = "2026-09-20T10:00:00+00:00"
        await store.put(doc)

        result = await sync_reading_queue(store, settings, vault)

        assert (await store.get(episode_id))["read_at"] == "2026-09-20T10:00:00+00:00"
        assert result["applied"] == []
        assert result["unread_total"] == 0

    @pytest.mark.asyncio
    async def test_a_deleted_star_line_says_nothing_about_the_star(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        episode_id = await _seed(store, starred=True)
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)
        vault.notes[QUEUE] = "\n".join(
            line for line in vault.notes[QUEUE].splitlines() if "⭐" not in line
        )
        await sync_reading_queue(store, settings, vault)

        assert (await store.get(episode_id))["starred"] is True

    @pytest.mark.asyncio
    async def test_a_link_we_did_not_write_is_left_alone(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        episode_id = await _seed(store)
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)
        vault.notes[QUEUE] += "\n- [x] [[99 topics/something else|Not ours]]\n"
        result = await sync_reading_queue(store, settings, vault)

        assert result["applied"] == []
        assert (await store.get(episode_id))["read_at"] is None


# --- the note itself ---------------------------------------------------------


class TestTheNote:
    @pytest.mark.asyncio
    async def test_the_readers_own_writing_survives_a_rebuild(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        await _seed(store)
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)
        vault.notes[QUEUE] = vault.notes[QUEUE].replace(
            "# Unread", "# Unread\n\nMy own note: read the OT ones first.\n"
        )

        await _seed(store, episode_id="episode:test-show:two", title="Another", note_name="two")
        await sync_reading_queue(store, settings, vault)

        assert "My own note: read the OT ones first." in vault.notes[QUEUE]
        assert "Another" in vault.notes[QUEUE]

    @pytest.mark.asyncio
    async def test_an_unchanged_queue_is_not_rewritten(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        await _seed(store)
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)
        assert len(vault.writes) == 1

        result = await sync_reading_queue(store, settings, vault)
        assert len(vault.writes) == 1
        assert result["written"] is False

    @pytest.mark.asyncio
    async def test_videos_and_podcasts_are_both_listed(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        await _seed(store)
        await _seed(
            store,
            episode_id="episode:video-digest:abc",
            title="A talk",
            imported=True,
            note_name="2026-09-17-a-talk",
        )
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)
        note = vault.notes[QUEUE]

        assert "## Podcasts · 1" in note
        assert "## Videos · 1" in note
        # video-digest's own note, at the path it told us on import — this app
        # writes no note for an imported episode.
        assert "[[13 video-summaries/2026-09-17-a-talk.md|A talk]]" in note

    @pytest.mark.asyncio
    async def test_an_episode_with_no_note_is_not_listed(self, tmp_path: Path) -> None:
        """A queue line that cannot be opened is worse than an absent one."""
        store = MemoryStore()
        settings = _settings(tmp_path)
        settings.output.episode_notes = True
        await _seed(store)
        # Summarised, but never given a note name.
        await store.put(
            {
                "_id": "episode:test-show:nameless",
                "type": "episode",
                "origin": "routine",
                "title": "Nameless",
                "podcast_name": "Test Show",
                "published_at": "2026-09-19T00:00:00+00:00",
                "tier1": {"summary_md": "text"},
            }
        )
        vault = _vault(settings)

        await sync_reading_queue(store, settings, vault)

        assert "Nameless" not in vault.notes[QUEUE]
        assert "An episode" in vault.notes[QUEUE]

    @pytest.mark.asyncio
    async def test_the_cap_says_what_it_left_off(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path, queue_limit=10)
        settings.output.episode_notes = True
        for index in range(12):
            await _seed(
                store,
                episode_id=f"episode:test-show:{index}",
                title=f"Episode {index}",
                note_name=f"note-{index}",
            )
        vault = _vault(settings)

        result = await sync_reading_queue(store, settings, vault)

        assert result["listed"] == 10
        assert result["unread_total"] == 12
        assert "2 older unread not listed" in vault.notes[QUEUE]

    @pytest.mark.asyncio
    async def test_a_disabled_vault_is_a_no_op(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = _settings(tmp_path, enabled=False)
        assert (await sync_reading_queue(store, settings, None))["skipped"]


class TestTheEndpoint:
    """`POST /api/v1/vault/reading-queue` — the same pass on demand."""

    def test_it_refuses_rather_than_silently_doing_nothing(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        from fastapi.testclient import TestClient
        from helpers import FakeLLM

        from podcast_agent.main import build_app

        settings = make_settings(tmp_path, vault={"enabled": False})
        with TestClient(build_app(settings, store=store, llm=FakeLLM())) as client:
            response = client.post("/api/v1/vault/reading-queue")

        assert response.status_code == 409
        assert "vault.enabled" in response.json()["detail"]


class TestParsingAndRendering:
    def test_a_bracketed_title_still_parses(self) -> None:
        """`[un]prompted` is a real alias in this vault, and a lazy regex stops
        at its inner bracket — the line then never matches and the item it names
        can never be ticked."""
        item = Item(
            episode_id="e1",
            title="[un]prompted | episode",
            show="Show",
            published_at="2026-09-18T00:00:00+00:00",
            note_path="11 podcasts/episodes/Show/note.md",
            starred=False,
            read_at=None,
            video=False,
        )
        note = render(unread=[item], recently_read=[], unread_total=1)
        marks = parse(note)

        assert marks == {"11 podcasts/episodes/show/note.md": {"read": False, "starred": False}}

    def test_indentation_is_not_what_decides_meaning(self) -> None:
        note = "- [x] [[a/b.md|T]] · Show\n  - [x] ⭐\n"
        assert parse(note) == {"a/b.md": {"read": True, "starred": True}}

    def test_prose_and_headings_are_not_items(self) -> None:
        assert parse("# Unread\n\nSome prose.\n\n## Podcasts · 0\n") == {}


class TestTheSweepLeavesItAlone:
    @respx.mock
    @pytest.mark.asyncio
    async def test_the_queue_note_is_not_retired(self, tmp_path: Path) -> None:
        """It sits directly under the owned root, which is exactly the shape
        `retire_moved` deletes — a note left at a path this application has
        stopped using. Retiring it would take a person's unticked queue off
        every device they own."""
        settings = _settings(tmp_path)
        vault = LiveSyncVault(settings.vault, "secret")
        respx.get(url__startswith="http://vault-couch.lan:5984/myvault/_all_docs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "rows": [
                        {
                            "doc": {
                                "_id": QUEUE.lower(),
                                "path": QUEUE,
                                "type": "plain",
                                "children": ["h:tabc"],
                            }
                        },
                        {
                            "doc": {
                                "_id": "11 podcasts/old-layout.md",
                                "path": "11 podcasts/old-layout.md",
                                "type": "plain",
                                "children": ["h:tdef"],
                            }
                        },
                    ]
                },
            )
        )
        retired_put = respx.put(url__startswith="http://vault-couch.lan:5984/myvault/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        retired = await retire_moved(vault, {"11 podcasts/episodes/show/note.md"})

        assert retired == ["11 podcasts/old-layout.md"]
        assert retired_put.call_count == 1
