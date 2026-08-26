"""Projecting written Markdown into an Obsidian vault over LiveSync.

The format assertions here are the load-bearing ones: LiveSync's document shape
is reverse-engineered rather than published, so nothing but a test pins it. If
one of these fails after a plugin upgrade, the projection is what changed, not
the test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from helpers import FakeLLM, make_settings

from podcast_agent.config import VaultConfig
from podcast_agent.db import MemoryStore
from podcast_agent.digest.generate import DigestGenerator
from podcast_agent.main import build_app
from podcast_agent.vault import (
    LiveSyncVault,
    VaultUnavailable,
    build_vault,
    entity_slugs,
    owned_root,
    project_file,
    projectable,
    retire_moved,
    slim_digest,
    sync_all,
    to_vault_markdown,
)

KEY = {"X-API-Key": "test-admin-key"}
COUCH = "http://vault-couch.lan:5984"
DB = "myvault"


def _cfg(**overrides: Any) -> VaultConfig:
    return VaultConfig(
        **{
            "enabled": True,
            "couchdb_url": COUCH,
            "db": DB,
            "user": "podagent",
            "folder": "11 podcasts/digests",
            **overrides,
        }
    )


def _vault(**overrides: Any) -> LiveSyncVault:
    return build_vault(_cfg(**overrides), "secret")


def _doc_url(doc_id: str) -> str:
    """CouchDB URL for a doc id, with the slashes percent-encoded as the code does."""
    from urllib.parse import quote

    return f"{COUCH}/{DB}/{quote(doc_id, safe='')}"


def _no_stale_entries() -> None:
    """Answer the listing `sync_all` makes when it looks for notes left at a path
    this application has stopped using. Empty: these vaults have no history."""
    respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
        return_value=httpx.Response(200, json={"rows": []})
    )


# --- the wire format ---------------------------------------------------------


class TestTheDocumentsLiveSyncExpects:
    @respx.mock
    @pytest.mark.asyncio
    async def test_a_file_becomes_a_chunk_and_an_entry(self) -> None:
        markdown = "# Week 34\n\nSomething happened.\n"
        chunk_id = "h:t" + hashlib.sha1(markdown.encode()).hexdigest()[:24]
        path = "11 podcasts/digests/2026/podcast-digest-2026-W34.md"

        chunk = respx.put(_doc_url(chunk_id)).mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        entry = respx.put(_doc_url(path.lower())).mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        written = await _vault().project(
            Path("2026/podcast-digest-2026-W34.md"), markdown, mtime_ms=1_700_000_000_000
        )

        assert written == path
        assert chunk.called and entry.called

        import json

        assert json.loads(chunk.calls[0].request.read()) == {
            "_id": chunk_id,
            "data": markdown,
            "type": "leaf",
        }

        body = json.loads(entry.calls[0].request.read())
        # The exact shape a live LiveSync client writes. Each of these has a
        # reason: `children` is how the text is found, `type: plain` is how the
        # plugin knows it is a file rather than one of its own records, and a
        # missing `eden` makes older clients discard the entry.
        assert body["_id"] == path.lower()
        assert body["path"] == path
        assert body["children"] == [chunk_id]
        assert body["type"] == "plain"
        assert body["eden"] == {}
        assert body["size"] == len(markdown.encode("utf-8"))
        assert body["mtime"] == 1_700_000_000_000

    def test_size_counts_bytes_not_characters(self) -> None:
        # An entry whose size disagrees with the chunk is how a client decides
        # the file changed underneath it and starts a conflict it cannot win.
        markdown = "Café — naïve\n"
        assert len(markdown) != len(markdown.encode("utf-8"))

    @respx.mock
    @pytest.mark.asyncio
    async def test_identical_content_shares_one_chunk(self) -> None:
        markdown = "same text\n"
        chunk_id = "h:t" + hashlib.sha1(markdown.encode()).hexdigest()[:24]
        chunk = respx.put(_doc_url(chunk_id)).mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        vault = _vault()
        await vault.project(Path("2026/a.md"), markdown, mtime_ms=1)
        await vault.project(Path("2026/b.md"), markdown, mtime_ms=1)

        # Content-addressed: two different files, and both entries point at the
        # one chunk rather than storing the text twice.
        assert chunk.call_count == 2

    def test_the_vault_path_is_the_folder_plus_the_relative_path(self) -> None:
        vault = _vault()
        assert vault.vault_path(Path("2026/w34.md")) == "11 podcasts/digests/2026/w34.md"
        assert (
            vault.vault_path(Path("signals/2026-W34.md"))
            == "11 podcasts/digests/signals/2026-W34.md"
        )


# --- a human's deletion is respected ----------------------------------------


class TestADeletedFileStaysDeleted:
    """The deliberate divergence from taster, which resurrects on purpose.

    A tasting note is a record worth restoring. A digest is generated output —
    pruning one from the vault is an instruction, not an accident.

    Scope, established against a real CouchDB rather than assumed: deleting in
    Obsidian is a *soft* delete (LiveSync keeps the doc and flags it) and that is
    what these cover.
    """

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_soft_deleted_entry_is_not_rewritten(self) -> None:
        markdown = "# gone\n"
        path = "11 podcasts/digests/2026/w34.md"
        respx.put(url__startswith=f"{COUCH}/{DB}/h%3At").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        entry = respx.put(_doc_url(path)).mock(return_value=httpx.Response(409, json={}))
        respx.get(_doc_url(path)).mock(
            return_value=httpx.Response(
                200, json={"_id": path, "_rev": "2-abc", "deleted": True, "children": []}
            )
        )

        written = await _vault().project(Path("2026/w34.md"), markdown, mtime_ms=1)

        assert written is None
        # The conflicting PUT happened once; no second, resurrecting one did.
        assert entry.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_losing_a_race_leaves_the_file_alone(self) -> None:
        # Live when we wrote, gone by the time we looked. Not the hard-delete
        # path: CouchDB answers a PUT with no _rev over a tombstone with 201, so
        # that case never produces a conflict at all (see the module docstring).
        path = "11 podcasts/digests/2026/w34.md"
        respx.put(url__startswith=f"{COUCH}/{DB}/h%3At").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        entry = respx.put(_doc_url(path)).mock(return_value=httpx.Response(409, json={}))
        respx.get(_doc_url(path)).mock(return_value=httpx.Response(404, json={"error": "deleted"}))

        written = await _vault().project(Path("2026/w34.md"), "# gone\n", mtime_ms=1)

        assert written is None
        assert entry.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_purged_document_does_come_back_and_that_is_documented(self) -> None:
        """The limit of the promise, pinned so the docstring cannot drift from it.

        A hard tombstone accepts a plain create, and after compaction it is not
        distinguishable from a document that never existed. Only Obsidian's own
        soft deletes are honourable, and those are the ones a person makes.
        """
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        written = await _vault().project(Path("2026/w34.md"), "# back\n", mtime_ms=1)
        assert written == "11 podcasts/digests/2026/w34.md"


class TestRewritingWhatIsAlreadyThere:
    @respx.mock
    @pytest.mark.asyncio
    async def test_an_unchanged_file_is_left_alone(self) -> None:
        markdown = "# same\n"
        chunk_id = "h:t" + hashlib.sha1(markdown.encode()).hexdigest()[:24]
        path = "11 podcasts/digests/2026/w34.md"
        respx.put(_doc_url(chunk_id)).mock(return_value=httpx.Response(409, json={}))
        respx.get(_doc_url(chunk_id)).mock(
            return_value=httpx.Response(
                200, json={"_id": chunk_id, "_rev": "1-a", "data": markdown}
            )
        )
        entry = respx.put(_doc_url(path)).mock(return_value=httpx.Response(409, json={}))
        respx.get(_doc_url(path)).mock(
            return_value=httpx.Response(
                200, json={"_id": path, "_rev": "3-x", "children": [chunk_id], "ctime": 111}
            )
        )

        written = await _vault().project(Path("2026/w34.md"), markdown, mtime_ms=999)

        assert written is None
        assert entry.call_count == 1  # no rewrite of identical content

    @respx.mock
    @pytest.mark.asyncio
    async def test_changed_content_updates_in_place_and_keeps_ctime(self) -> None:
        # The signals file for a period is rewritten as new marks arrive; its
        # creation time should stay the first one, not follow every edit.
        path = "11 podcasts/digests/signals/2026-W34.md"
        entry_id = path.lower()  # LiveSync keys entries by lowercased path
        respx.put(url__startswith=f"{COUCH}/{DB}/h%3At").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        entry = respx.put(_doc_url(entry_id)).mock(
            side_effect=[httpx.Response(409, json={}), httpx.Response(201, json={"ok": True})]
        )
        respx.get(_doc_url(entry_id)).mock(
            return_value=httpx.Response(
                200, json={"_id": entry_id, "_rev": "3-x", "children": ["h:told"], "ctime": 111}
            )
        )

        written = await _vault().project(Path("signals/2026-W34.md"), "# new\n", mtime_ms=999)

        assert written == path
        assert entry.call_count == 2

        import json

        body = json.loads(entry.calls[1].request.read())
        assert body["_rev"] == "3-x"
        assert body["ctime"] == 111  # preserved
        assert body["mtime"] == 999  # updated


# --- failure is the operator's problem, never the digest's -------------------


class TestAnUnreachableVaultNeverFailsTheDigest:
    @pytest.mark.asyncio
    async def test_no_url_configured_is_reported_not_guessed(self) -> None:
        vault = build_vault(VaultConfig(), None)
        with pytest.raises(VaultUnavailable, match="couchdb_url is not set"):
            await vault.project(Path("2026/w34.md"), "# x\n", mtime_ms=1)

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_connection_error_becomes_vault_unavailable(self) -> None:
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            side_effect=httpx.ConnectError("no route to host")
        )
        with pytest.raises(VaultUnavailable, match="unreachable"):
            await _vault().project(Path("2026/w34.md"), "# x\n", mtime_ms=1)

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_rejected_write_does_not_leak_the_password(self) -> None:
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )
        with pytest.raises(VaultUnavailable) as excinfo:
            await _vault().project(Path("2026/w34.md"), "# x\n", mtime_ms=1)
        assert "secret" not in str(excinfo.value)
        assert "401" in str(excinfo.value)

    @respx.mock
    @pytest.mark.asyncio
    async def test_project_file_swallows_it_so_the_caller_carries_on(self, tmp_path: Path) -> None:
        # The whole point of the helper: the digest on disk is already correct,
        # and a sleeping database must not turn it into a failure.
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(side_effect=httpx.ConnectError("down"))
        written = tmp_path / "2026" / "w34.md"
        written.parent.mkdir(parents=True)
        written.write_text("# digest\n", encoding="utf-8")

        await project_file(_vault(), tmp_path, written)  # must not raise

    @pytest.mark.asyncio
    async def test_no_vault_configured_is_a_no_op(self, tmp_path: Path) -> None:
        written = tmp_path / "w34.md"
        written.write_text("# digest\n", encoding="utf-8")
        await project_file(None, tmp_path, written)


# --- configuration -----------------------------------------------------------


class TestConfiguration:
    def test_enabling_without_a_url_refuses_to_boot(self) -> None:
        with pytest.raises(ValueError, match="couchdb_url is required"):
            VaultConfig(enabled=True)

    def test_off_by_default_with_no_url(self) -> None:
        cfg = VaultConfig()
        assert cfg.enabled is False
        assert cfg.couchdb_url is None

    def test_the_folder_is_normalised(self) -> None:
        assert VaultConfig(folder="/11 podcasts/").folder == "11 podcasts"

    def test_an_empty_folder_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            VaultConfig(folder="/")

    def test_a_non_http_url_is_refused(self) -> None:
        with pytest.raises(ValueError, match="http"):
            VaultConfig(enabled=True, couchdb_url="ftp://nas.lan")

    def test_it_is_not_console_overridable(self) -> None:
        # Where notes are sent is deployment topology, like asr.remote_url —
        # not something a typo in a browser should be able to redirect.
        from podcast_agent.settings_store import OVERRIDABLE_SECTIONS

        assert "vault" not in OVERRIDABLE_SECTIONS

    def test_settings_carries_the_section_and_its_own_secret(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        assert settings.vault.enabled is False
        # A different database owned by a different application, so its password
        # is not the one this app uses for its own store.
        assert settings.vault_couchdb_password is None


# --- the catch-up endpoint ---------------------------------------------------


class TestTheSyncEndpoint:
    """`POST /api/v1/vault/sync` — for the first run after switching it on, and
    for whatever a spell of downtime missed."""

    @staticmethod
    def _client(
        tmp_path: Path, store: MemoryStore, digests: Path | None = None, **vault: Any
    ) -> TestClient:
        extra: dict[str, Any] = {"vault": vault or {"enabled": False}}
        if digests is not None:
            extra["output"] = {"digest_dir": digests}
        settings = make_settings(tmp_path, **extra)
        app = build_app(settings, store=store, llm=FakeLLM())
        return TestClient(app)

    def test_it_refuses_rather_than_silently_doing_nothing(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        with self._client(tmp_path, store) as client:
            response = client.post("/api/v1/vault/sync", headers=KEY)
        assert response.status_code == 409
        assert "vault.enabled" in response.json()["detail"]

    @respx.mock
    def test_it_projects_digests_and_signals_but_not_the_archive(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        digests = tmp_path / "digests"
        for relative in (
            "2026/podcast-digest-2026-W33.md",
            "2026/podcast-digest-2026-W34.md",
            "signals/2026-W34.md",
            # Two hundred of these exist in production. A personal vault is a
            # hundred and eighty files; they do not belong in it.
            "archive/risky-business/2019-01-01-something.md",
        ):
            path = digests / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
        (digests / "2026" / "podcast-digest-2026-W34.mp3").write_bytes(b"not markdown")

        route = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        _no_stale_entries()

        with self._client(
            tmp_path,
            store,
            digests,
            enabled=True,
            couchdb_url=COUCH,
            db=DB,
        ) as client:
            response = client.post("/api/v1/vault/sync", headers=KEY)

        assert response.status_code == 200
        body = response.json()
        assert body["considered"] == 3
        assert sorted(body["projected"]) == [
            "11 podcasts/digests/2026/podcast-digest-2026-W33.md",
            "11 podcasts/digests/2026/podcast-digest-2026-W34.md",
            "11 podcasts/digests/signals/2026-W34.md",
        ]
        # Nothing under archive/, and no audio.
        paths = [str(call.request.url) for call in route.calls]
        assert not any("archive" in p for p in paths)
        assert not any(".mp3" in p for p in paths)

    @respx.mock
    def test_a_dead_vault_reports_how_far_it_got(self, tmp_path: Path, store: MemoryStore) -> None:
        digests = tmp_path / "digests"
        (digests / "2026").mkdir(parents=True)
        (digests / "2026" / "podcast-digest-2026-W34.md").write_text("# w34\n", encoding="utf-8")
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(side_effect=httpx.ConnectError("down"))

        with self._client(
            tmp_path,
            store,
            digests,
            enabled=True,
            couchdb_url=COUCH,
            db=DB,
        ) as client:
            response = client.post("/api/v1/vault/sync", headers=KEY)

        assert response.status_code == 503
        assert "projected 0 of 1" in response.json()["detail"]


# --- the generator's own hook ------------------------------------------------


class TestTheDigestProjectsItself:
    """It no longer does, and that is the point.

    Each entry's summary is replaced in the vault by a link to that episode's own
    note, and those notes are written by the job that runs *after* the digest.
    Projecting at generation time would publish a digest 45 minutes before the
    things it points at exist."""

    @staticmethod
    def _settings(tmp_path: Path, **vault: Any) -> Any:
        return make_settings(
            tmp_path,
            vault=vault or {"enabled": False},
            output={"digest_dir": tmp_path / "digests"},
        )

    @respx.mock
    @pytest.mark.asyncio
    async def test_generating_a_digest_does_not_reach_the_vault_on_its_own(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        from test_digest import PERIOD_FROM, seed_corpus

        seed_corpus(store)
        route = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        settings = self._settings(tmp_path, enabled=True, couchdb_url=COUCH, db=DB)

        result = await DigestGenerator(
            settings, store, None, build_vault(settings.vault, "secret")
        ).generate(since=PERIOD_FROM)

        assert result.file_path is not None and result.file_path.exists()
        assert route.call_count == 0, "the digest was projected before its episode notes existed"

    @respx.mock
    @pytest.mark.asyncio
    async def test_the_sync_pass_is_what_publishes_it(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        from test_digest import PERIOD_FROM, seed_corpus

        seed_corpus(store)
        settings = self._settings(tmp_path, enabled=True, couchdb_url=COUCH, db=DB)
        await DigestGenerator(settings, store, None, None).generate(since=PERIOD_FROM)

        # Registered first so it wins over the catch-all below: respx resolves a
        # request against the earliest matching route.
        _no_stale_entries()
        respx.get(url__startswith=f"{COUCH}/{DB}/").mock(return_value=httpx.Response(404, json={}))
        route = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        result = await sync_all(build_vault(settings.vault, "secret"), settings.output.digest_dir)

        assert result["projected"]
        assert route.call_count > 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_dead_vault_does_not_fail_the_digest(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        # The reason project_file swallows: the file on disk is the deliverable,
        # and a database asleep on another machine is not the digest's fault.
        from test_digest import PERIOD_FROM, seed_corpus

        seed_corpus(store)
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(side_effect=httpx.ConnectError("down"))
        settings = self._settings(tmp_path, enabled=True, couchdb_url=COUCH, db=DB)

        result = await DigestGenerator(
            settings, store, None, build_vault(settings.vault, "secret")
        ).generate(since=PERIOD_FROM)

        assert result.file_path is not None
        assert result.file_path.exists()
        assert result.file_path.read_text(encoding="utf-8").strip()

    @respx.mock
    @pytest.mark.asyncio
    async def test_the_signals_file_is_projected_too(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        # It exists to be read by a model working over the vault, which means it
        # has to actually be in the vault.
        from test_signals import marked

        from podcast_agent.signals import export_new_marks

        store.seed(marked("liked", starred=True))
        route = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        settings = self._settings(tmp_path, enabled=True, couchdb_url=COUCH, db=DB)

        result = await export_new_marks(store, settings, build_vault(settings.vault, "secret"))

        assert result["written"]
        assert any("podcasts%2Fdigests%2Fsignals%2F" in str(c.request.url) for c in route.calls)


class TestEmptyEnvironmentValuesDoNotBrickTheBoot:
    """Compose substitutes `${VAULT_COUCHDB_URL:-}` to an empty string when the
    var is absent. Every deployment that has not opted in sends exactly that."""

    def test_an_empty_url_reads_as_unset(self) -> None:
        assert VaultConfig(couchdb_url="").couchdb_url is None
        assert VaultConfig(couchdb_url="   ").couchdb_url is None

    def test_an_empty_url_still_blocks_enabling(self) -> None:
        with pytest.raises(ValueError, match="couchdb_url is required"):
            VaultConfig(enabled=True, couchdb_url="")

    def test_the_shipped_compose_defaults_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # What `docker compose config` produces with no vault vars set.
        monkeypatch.setenv("PODAGENT_VAULT__COUCHDB_URL", "")
        monkeypatch.setenv("PODAGENT_VAULT__DB", "vault")
        monkeypatch.setenv("PODAGENT_VAULT_COUCHDB_PASSWORD", "")
        cfg = VaultConfig(couchdb_url="", db="vault")
        assert cfg.enabled is False and cfg.couchdb_url is None


# --- writing for a vault rather than for a console ---------------------------


class TestVaultFlavouredMarkdown:
    """The digest on disk serves two readers the vault does not: the console,
    which renders it as HTML, and the narrator, which reads it aloud. Both would
    show `[[...]]` verbatim (verified — wikilinks pass straight through
    `md_to_speech_text`), so the conversion happens here, at the vault boundary,
    and the file on disk is left alone."""

    def test_entities_with_a_note_become_links(self) -> None:
        out = to_vault_markdown(
            "**Mentioned:** Anthropic, NIS2, CVE-2026-69414",
            known_entities={"anthropic", "nis2", "cve-2026-69414"},
        )
        assert (
            out
            == "**Mentioned:** [[anthropic|Anthropic]], [[nis2|NIS2]], [[cve-2026-69414|CVE-2026-69414]]"
        )

    def test_entities_without_a_note_stay_plain(self) -> None:
        # The threshold means most entities never get a note. Linking them anyway
        # would fill the digest with dangling links, which is worse than text.
        out = to_vault_markdown(
            "**Mentioned:** Anthropic, Some One-Off Thing",
            known_entities={"anthropic"},
        )
        assert out == "**Mentioned:** [[anthropic|Anthropic]], Some One-Off Thing"

    def test_nothing_is_linked_when_no_notes_exist(self) -> None:
        body = "**Mentioned:** Anthropic, NIS2"
        assert to_vault_markdown(body, known_entities=set()) == body
        assert to_vault_markdown(body) == body

    def test_a_name_that_would_break_the_syntax_is_left_alone(self) -> None:
        out = to_vault_markdown(
            "**Mentioned:** Weird|Name, Anthropic", known_entities={"weird-name", "anthropic"}
        )
        assert "[[weird-name|Weird|Name]]" not in out
        assert "[[anthropic|Anthropic]]" in out

    def test_the_dead_audio_embed_becomes_a_line_of_text(self) -> None:
        # ![[x.mp3]] is an *embed*: Obsidian renders an inline player. The audio
        # is ~100 MB a week and stays on the NAS, so in the vault it is a broken
        # player directly under the digest's H1.
        out = to_vault_markdown("# Digest\n\n![[podcast-digest-2026-W34.mp3]]\n\nBody.")
        assert "![[" not in out
        assert "podcast-digest-2026-W34.mp3" in out
        assert "not synced" in out

    def test_other_embeds_are_untouched(self) -> None:
        body = "![[some-diagram.png]]"
        assert to_vault_markdown(body) == body

    def test_the_summary_body_is_not_mangled(self) -> None:
        body = "## Key takeaways\n\n- Anthropic said a thing\n- **Mentioned** in passing\n"
        assert to_vault_markdown(body, known_entities={"anthropic"}) == body


class TestEntityNotesAreFiledAsTopics:
    def test_they_go_to_the_topics_folder_not_the_podcast_one(self) -> None:
        v = _vault()
        assert v.vault_path(Path("entities/anthropic.md")) == "99 topics/anthropic.md"
        assert v.vault_path(Path("2026/w34.md")) == "11 podcasts/digests/2026/w34.md"
        assert (
            v.vault_path(Path("signals/2026-W34.md")) == "11 podcasts/digests/signals/2026-W34.md"
        )

    def test_the_two_folders_are_configured_separately(self) -> None:
        v = build_vault(_cfg(folder="a/b", entities_folder="c"), None)
        assert v.vault_path(Path("entities/x.md")) == "c/x.md"
        assert v.vault_path(Path("2026/x.md")) == "a/b/2026/x.md"

    def test_an_empty_topics_folder_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            VaultConfig(entities_folder="/")

    def test_known_slugs_come_from_the_notes_that_exist(self, tmp_path: Path) -> None:
        assert entity_slugs(tmp_path) == set()  # no entities/ directory at all
        (tmp_path / "entities").mkdir()
        (tmp_path / "entities" / "anthropic.md").write_text("x", encoding="utf-8")
        (tmp_path / "entities" / "nis2.md").write_text("x", encoding="utf-8")
        assert entity_slugs(tmp_path) == {"anthropic", "nis2"}


class TestTheFullSyncPass:
    def _tree(self, tmp_path: Path) -> Path:
        base = tmp_path / "digests"
        for rel, body in (
            ("2026/podcast-digest-2026-W34.md", "**Mentioned:** Anthropic, Nobody\n"),
            ("signals/2026-W34.md", "# marks\n"),
            ("entities/anthropic.md", "# Anthropic\n"),
            ("archive/risky-business/old.md", "# archive\n"),
        ):
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        (base / "2026" / "podcast-digest-2026-W34.mp3").write_bytes(b"audio")
        return base

    def test_it_covers_digests_signals_and_topics_but_not_the_archive(self, tmp_path: Path) -> None:
        paths = {p.name for p in projectable(self._tree(tmp_path))}
        assert paths == {
            "podcast-digest-2026-W34.md",
            "2026-W34.md",
            "anthropic.md",
        }

    @respx.mock
    @pytest.mark.asyncio
    async def test_it_links_against_the_notes_it_is_projecting(self, tmp_path: Path) -> None:
        base = self._tree(tmp_path)
        _no_stale_entries()
        # Topic notes are read before being written (they are merged); nothing is
        # in the vault yet, so every read is a miss.
        respx.get(url__startswith=f"{COUCH}/{DB}/").mock(return_value=httpx.Response(404, json={}))
        route = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        result = await sync_all(_vault(), base)

        assert result["considered"] == 3
        assert result["linkable_entities"] == 1
        # The digest went up with Anthropic linked and Nobody left plain.
        import json

        chunks = [
            json.loads(c.request.read())
            for c in route.calls
            if "%2F2026%2F" not in str(c.request.url) and "h%3At" in str(c.request.url)
        ]
        digest_chunk = next(c for c in chunks if "Mentioned" in (c.get("data") or ""))
        assert "[[anthropic|Anthropic]]" in digest_chunk["data"]
        assert "Nobody" in digest_chunk["data"] and "[[nobody" not in digest_chunk["data"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_dead_vault_reports_how_far_it_got(self, tmp_path: Path) -> None:
        base = self._tree(tmp_path)
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(VaultUnavailable, match="projected 0 of 3"):
            await sync_all(_vault(), base)


class TestTheThresholdIsOneNumber:
    """The endpoint and the Friday job must agree, or a manual call quietly
    rebuilds the vault at a different size than the cron does."""

    def test_the_endpoint_defaults_to_the_configured_threshold(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        settings = make_settings(
            tmp_path,
            pipeline={"entity_note_min_mentions": 9},
            output={"digest_dir": tmp_path / "digests"},
        )
        app = build_app(settings, store=store, llm=FakeLLM())
        with TestClient(app) as client:
            body = client.post("/api/v1/entities/notes", headers=KEY).json()
        assert body["min_mentions"] == 9

    def test_an_explicit_value_still_wins(self, tmp_path: Path, store: MemoryStore) -> None:
        settings = make_settings(
            tmp_path,
            pipeline={"entity_note_min_mentions": 9},
            output={"digest_dir": tmp_path / "digests"},
        )
        app = build_app(settings, store=store, llm=FakeLLM())
        with TestClient(app) as client:
            body = client.post("/api/v1/entities/notes?min_mentions=3", headers=KEY).json()
        assert body["min_mentions"] == 3

    def test_it_is_tunable_from_the_console(self) -> None:
        from podcast_agent.settings_store import OVERRIDABLE_PIPELINE_KEYS

        assert "entity_note_min_mentions" in OVERRIDABLE_PIPELINE_KEYS


# --- topic notes are shared property ----------------------------------------


class TestTopicNotesMergeAgainstTheVault:
    """The merge reads the note as the *vault* holds it, not as we hold it.

    Nothing syncs back, so our copy under digest_dir cannot know what a reader —
    or another application — wrote into a topic note. Merging into our own copy
    would rewrite the vault from a source blind to its contents.
    """

    OURS = (
        "---\ntype: topic\npodcasts_mentions: 97\n---\n\n# Anthropic\n\n"
        "<!-- begin:podcast-digest -->\n## From podcasts\n\n- fresh\n<!-- end:podcast-digest -->\n"
    )
    PATH = "99 topics/anthropic.md"

    def _in_vault(self, body: str) -> None:
        """Stand up a note in the mocked vault: entry plus its one chunk."""
        respx.get(_doc_url(self.PATH)).mock(
            return_value=httpx.Response(
                200,
                json={"_id": self.PATH, "_rev": "3-x", "children": ["h:tOLD"], "ctime": 111},
            )
        )
        respx.get(_doc_url("h:tOLD")).mock(
            return_value=httpx.Response(200, json={"_id": "h:tOLD", "data": body, "type": "leaf"})
        )

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_readers_own_writing_is_not_overwritten(self) -> None:
        self._in_vault(
            "---\ntype: topic\ntags: [topic, ai]\n---\n\n# Anthropic\n\n"
            "My own view.\n\n"
            "<!-- begin:podcast-digest -->\n## From podcasts\n\n- stale\n<!-- end:podcast-digest -->\n"
        )
        puts = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        written = await _vault().project(
            Path("entities/anthropic.md"), self.OURS, mtime_ms=1, merge=True
        )
        assert written == self.PATH

        import json

        chunk = next(
            json.loads(c.request.read()) for c in puts.calls if "h%3At" in str(c.request.url)
        )
        assert "My own view." in chunk["data"]
        assert "tags: [topic, ai]" in chunk["data"]
        assert "- fresh" in chunk["data"] and "- stale" not in chunk["data"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_another_applications_section_survives_our_run(self) -> None:
        self._in_vault(
            "---\ntype: topic\nclippings_count: 12\n---\n\n# Anthropic\n\n"
            "<!-- begin:podcast-digest -->\n## From podcasts\n\n- stale\n<!-- end:podcast-digest -->\n\n"
            "<!-- begin:notes-app -->\n## From clippings\n\n- their line\n<!-- end:notes-app -->\n"
        )
        puts = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        await _vault().project(Path("entities/anthropic.md"), self.OURS, mtime_ms=1, merge=True)

        import json

        chunk = next(
            json.loads(c.request.read()) for c in puts.calls if "h%3At" in str(c.request.url)
        )
        assert "- their line" in chunk["data"]
        assert "clippings_count: 12" in chunk["data"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_an_unchanged_section_writes_nothing(self) -> None:
        # Weekly rebuilds mostly produce identical notes; each should be a no-op
        # rather than a new revision on 137 documents.
        self._in_vault(self.OURS)
        written = await _vault().project(
            Path("entities/anthropic.md"), self.OURS, mtime_ms=1, merge=True
        )
        assert written is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_topic_the_reader_deleted_is_not_rebuilt(self) -> None:
        respx.get(_doc_url(self.PATH)).mock(
            return_value=httpx.Response(
                200, json={"_id": self.PATH, "_rev": "2-a", "deleted": True, "children": []}
            )
        )
        entry = respx.put(_doc_url(self.PATH)).mock(return_value=httpx.Response(409, json={}))
        respx.put(url__startswith=f"{COUCH}/{DB}/h%3At").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        written = await _vault().project(
            Path("entities/anthropic.md"), self.OURS, mtime_ms=1, merge=True
        )
        assert written is None
        assert entry.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_digests_are_not_merged_they_are_ours_outright(self) -> None:
        # A digest has no other writer, and merging one would preserve last
        # week's body forever.
        gets = respx.get(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(404, json={})
        )
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )
        await _vault().project(Path("2026/w34.md"), "# Digest\n", mtime_ms=1)
        assert gets.call_count == 0  # no read-before-write for our own files


class TestTheDigestIsSlimmedForTheVault:
    """The same summary lived twice — once in the weekly digest, once in the
    episode's own note — and the digest is 127 KB of mostly that."""

    ENTRY = (
        "### 🎙️ Show — A Thing  `9/10`\n"
        "<!-- ep:episode:abc -->\n"
        "\n"
        "*Published 2026-08-20 · 53 min*\n"
        "\n"
        "**Why it matters:** It matters.\n"
        "\n"
        "<!-- full -->\n"
        "The long summary.\n"
        "\n"
        "**Key takeaways:**\n"
        "\n"
        "- A point\n"
        "<!-- /full -->\n"
        "**Mentioned:** Anthropic\n"
    )

    def test_the_summary_gives_way_to_a_link(self) -> None:
        out = slim_digest(self.ENTRY, {"episode:abc": "2026-08-20-a-thing"})
        assert "The long summary." not in out
        assert "A point" not in out
        assert "[[2026-08-20-a-thing|Read the full summary]]" in out

    def test_what_only_the_digest_has_is_kept(self) -> None:
        out = slim_digest(self.ENTRY, {"episode:abc": "2026-08-20-a-thing"})
        for keep in (
            "### 🎙️ Show — A Thing",
            "`9/10`",
            "Published 2026-08-20",
            "**Why it matters:**",
        ):
            assert keep in out, keep
        # The entity line is what makes the digest a hub in the graph.
        assert "**Mentioned:** Anthropic" in out

    def test_an_entry_whose_note_does_not_exist_yet_keeps_its_text(self) -> None:
        # The ordering safeguard. The digest is generated at 06:00 and the
        # episode notes at 06:45; stripping unconditionally would publish a
        # gutted digest pointing at notes that had not been written.
        out = slim_digest(self.ENTRY, {})
        assert "The long summary." in out
        assert "A point" in out
        assert "[[" not in out

    def test_the_anchors_never_reach_the_reader(self) -> None:
        for names in ({"episode:abc": "n"}, {}):
            out = slim_digest(self.ENTRY, names)
            assert "<!--" not in out and "-->" not in out

    def test_an_entry_missing_its_region_is_left_alone(self) -> None:
        broken = "### A\n<!-- ep:episode:x -->\n\nno region here\n"
        out = slim_digest(broken, {"episode:x": "note"})
        assert "no region here" in out
        assert "<!--" not in out

    def test_it_runs_as_part_of_the_vault_transform(self) -> None:
        out = to_vault_markdown(self.ENTRY, episode_notes={"episode:abc": "2026-08-20-a-thing"})
        assert "[[2026-08-20-a-thing|Read the full summary]]" in out
        assert "The long summary." not in out


class TestTheConsoleNeverSeesTheAnchors:
    def test_they_are_stripped_before_rendering(self, tmp_path: Path) -> None:
        # `md_to_safe_html` escapes raw HTML rather than dropping it, so an
        # anchor left in would render as visible `&lt;!-- ep:… --&gt;`.
        from podcast_agent.digest.read import read_digest

        base = tmp_path / "digests"
        (base / "2026").mkdir(parents=True)
        (base / "2026" / "d.md").write_text(
            "---\nweek: x\n---\n\n# D\n\n<!-- ep:episode:abc -->\n\n<!-- full -->\ntext\n<!-- /full -->\n",
            encoding="utf-8",
        )
        out = read_digest(base, "2026/d.md")
        assert "<!--" not in out["markdown"]
        assert "ep:episode" not in out["html"]
        assert "text" in out["markdown"]


class TestFinishingAMove:
    """A note that changes path has to leave the old one, or the vault holds the
    same filename twice — and Obsidian resolves a `[[wikilink]]` with two targets
    to one of them without saying which. This is the only thing here that
    deletes, and it deletes nothing it did not itself write at a path it has
    since abandoned."""

    @staticmethod
    def _entry(path: str) -> dict[str, Any]:
        return {
            "_id": path.lower(),
            "_rev": "1-abc",
            "path": path,
            "children": ["h:tabc"],
            "type": "plain",
            "eden": {},
        }

    def _listing(self, *paths: str) -> respx.Route:
        return respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
            return_value=httpx.Response(
                200, json={"rows": [{"doc": self._entry(p)} for p in paths]}
            )
        )

    def test_the_root_that_holds_both_of_our_folders(self) -> None:
        assert owned_root(_vault()) == "11 podcasts"

    def test_no_root_when_the_two_are_configured_apart(self) -> None:
        # Nothing can be swept wholesale, because no folder is ours outright.
        assert owned_root(_vault(folder="A", episodes_folder="B")) is None

    def test_no_root_when_one_folder_contains_the_other(self) -> None:
        # `11 podcasts` would otherwise be swept for not being `11 podcasts`,
        # which is to say every digest would be deleted.
        assert owned_root(_vault(folder="11 podcasts", episodes_folder="11 podcasts/episodes")) is (
            None
        )

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_note_at_the_folder_we_used_to_use_is_retired(self) -> None:
        self._listing("13 podcast-episodes/2026-07-28-a-thing.md")
        put = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        retired = await retire_moved(_vault(), {"11 podcasts/episodes/test show/x.md"})

        assert retired == ["13 podcast-episodes/2026-07-28-a-thing.md"]
        body = json.loads(put.calls[0].request.read())
        # LiveSync's own deletion, not CouchDB's: the document stays and carries
        # a flag, which is what replicating clients act on. A real tombstone
        # would leave every client that already has the file holding it.
        assert body["deleted"] is True
        assert body["_rev"] == "1-abc"
        assert "_deleted" not in body

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_note_still_on_disk_is_left_alone(self) -> None:
        self._listing("11 podcasts/episodes/Test Show/x.md")
        put = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        retired = await retire_moved(_vault(), {"11 podcasts/episodes/test show/x.md"})

        assert retired == []
        assert put.call_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_an_episode_note_at_a_show_name_no_longer_used_is_retired(self) -> None:
        self._listing(
            "11 podcasts/episodes/Test Show/x.md",
            "11 podcasts/episodes/Old Name/x.md",
        )
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        retired = await retire_moved(_vault(), {"11 podcasts/episodes/test show/x.md"})

        assert retired == ["11 podcasts/episodes/Old Name/x.md"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_nothing_is_swept_when_the_writer_did_not_run(self) -> None:
        # An empty set is "no list was built", not "the corpus is empty". Acting
        # on it would delete every episode note in the vault the first time
        # output.episode_notes was switched off.
        self._listing("11 podcasts/episodes/Test Show/x.md")
        put = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        assert await retire_moved(_vault(), set()) == []
        assert put.call_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_digest_is_never_retired(self) -> None:
        # Digests are not regenerated, so disk is not authoritative for them.
        self._listing("11 podcasts/digests/2026/podcast-digest-2026-W34.md")
        put = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        assert await retire_moved(_vault(), {"11 podcasts/episodes/test show/x.md"}) == []
        assert put.call_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_digest_at_the_old_path_is_retired_though(self) -> None:
        # `11 podcasts/2026/` is where digests went before they had a subfolder
        # of their own. It is under the root and in neither of our folders.
        self._listing("11 podcasts/2026/podcast-digest-2026-W34.md")
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        retired = await retire_moved(_vault(), {"11 podcasts/episodes/test show/x.md"})

        assert retired == ["11 podcasts/2026/podcast-digest-2026-W34.md"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_topic_note_is_out_of_reach_entirely(self) -> None:
        # `99 topics` is shared with two other applications and the reader; it is
        # under no root of ours, so no listing even looks at it.
        listing = self._listing()
        await retire_moved(_vault(), {"11 podcasts/episodes/test show/x.md"})
        asked = {
            json.loads(c.request.url.params["startkey"])
            for c in listing.calls  # type: ignore[arg-type]
        }
        assert asked == {"13 podcast-episodes/", "11 podcasts/"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_note_someone_else_already_changed_is_left_where_it_is(self) -> None:
        self._listing("13 podcast-episodes/x.md")
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(409, json={"error": "conflict"})
        )

        assert await retire_moved(_vault(), {"11 podcasts/episodes/test show/x.md"}) == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_soft_deleted_entry_is_not_retired_twice(self) -> None:
        respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "rows": [
                        {"doc": {**self._entry("13 podcast-episodes/x.md"), "deleted": True}},
                        # A chunk sharing the key range would be a disaster to
                        # write `deleted` onto; only file entries are files.
                        {"doc": {"_id": "13 podcast-episodes/y", "type": "leaf", "data": "x"}},
                    ]
                },
            )
        )
        put = respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        assert await retire_moved(_vault(), {"11 podcasts/episodes/test show/x.md"}) == []
        assert put.call_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_an_empty_digest_directory_sweeps_nothing(self, tmp_path: Path) -> None:
        # An unmounted volume looks exactly like a corpus with nothing in it.
        # Emptying the vault to match would be the worst possible reading of it.
        listing = self._listing("13 podcast-episodes/x.md")
        result = await sync_all(_vault(), tmp_path)
        assert result["retired"] == []
        assert listing.call_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_truncated_pass_sweeps_nothing(self, tmp_path: Path) -> None:
        """The one that got out. A `limit` below the number of files on disk
        makes the set of episode notes look short by everything it did not
        reach, and a sweep that trusts it retires them. It did: 566 notes,
        against the live vault, through an endpoint whose default limit was
        below the size of the corpus."""
        episodes = tmp_path / "episodes" / "Test Show"
        episodes.mkdir(parents=True)
        for i in range(3):
            (episodes / f"note-{i}.md").write_text(f"# {i}\n", encoding="utf-8")

        listing = self._listing("11 podcasts/episodes/Test Show/note-2.md")
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        result = await sync_all(_vault(), tmp_path, limit=2)

        assert result["considered"] == 2
        assert result["retired"] == []
        assert listing.call_count == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_complete_pass_still_sweeps(self, tmp_path: Path) -> None:
        episodes = tmp_path / "episodes" / "Test Show"
        episodes.mkdir(parents=True)
        (episodes / "note.md").write_text("# note\n", encoding="utf-8")

        self._listing("11 podcasts/episodes/Gone Show/note.md")
        respx.put(url__startswith=f"{COUCH}/{DB}/").mock(
            return_value=httpx.Response(201, json={"ok": True})
        )

        result = await sync_all(_vault(), tmp_path, limit=10)

        assert result["retired"] == ["11 podcasts/episodes/Gone Show/note.md"]
