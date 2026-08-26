"""Projecting written Markdown into an Obsidian vault, over Self-hosted LiveSync.

The digest is already a file on disk. This is the hop that makes it readable in
Obsidian on a phone, and it is deliberately not a file copy: LiveSync replicates
a vault against a CouchDB database, so writing LiveSync's *own* document shape
into that database materialises the file on every client that syncs. Nothing has
to be awake but the NAS, and no folder has to be mounted anywhere.

The wire format is not documented by the plugin; it was reverse-engineered from
documents a live LiveSync v0.25 client wrote (E2EE off) and has been in
production in the taster project since — see its `backend/app/couchdb_client.py`,
which this module is a port of. Two documents per file:

* a **chunk**, ``{_id: "h:t<hash>", data: <markdown>, type: "leaf"}``, holding the
  text and content-addressed so identical content is stored once;
* an **entry**, keyed by the lowercased vault path, carrying
  ``{path, children: [chunk ids], ctime, mtime, size, type: "plain", eden: {}}``.

LiveSync fetches children strictly by id, so using our own ``h:t`` namespace
rather than matching its internal xxhash scheme costs at most a duplicate chunk.

**Where this deliberately differs from taster: a deleted file stays deleted.**
taster resurrects vault files on purpose — a tasting note is a record, and its
rebuild exists to restore one a human removed by accident. A digest is generated
output. If you prune last month's digest from the vault, re-projecting it would
be the software arguing with you, so a deleted entry is left alone and reported.

The precise scope of that promise, verified against a real CouchDB rather than
assumed: deleting a note in Obsidian is a **soft** delete — LiveSync keeps the
document and sets ``deleted: true`` — and that is what is honoured. A **hard**
CouchDB tombstone is a different matter: ``PUT`` with no ``_rev`` over one
returns 201, not 409, so the write simply succeeds and the file comes back. That
is not a decision so much as a limit, because once compaction has run a purged
tombstone is indistinguishable from a document that never existed. Hard deletes
come from direct database operations, not from anything Obsidian does.

There is one thing here that does delete, and it is the second half of a move
rather than a policy about old files: when a note starts being written at a new
path — the folders were restructured, a show renamed itself — the entry at the
old path is retired, in exactly the way Obsidian's own deletion replicates.
Leaving it would put the same filename in the vault twice, and a ``[[wikilink]]``
with two possible targets resolves to one of them without saying which, so half
a move reads as a working vault that opens the wrong note. See
:func:`retire_moved`, which sweeps only folders this application has abandoned
and never touches a digest or a topic note.

Every failure raises :class:`VaultUnavailable`, the same blanket translation
:class:`~.speech.SpeechUnavailable` uses and for the same reason: the digest is
already written and is perfectly good, so a sync target being down is an
operator problem and must never mark anything failed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .config import VaultConfig
from .logging_setup import get_logger
from .notes import ENTITIES_DIR, EPISODES_DIR, merge_owned_section
from .sanitize import slugify
from .vault_anchors import EPISODE_ANCHOR, strip_vault_anchors

log = get_logger(__name__)

#: Ids LiveSync treats as chunks live in the ``h:`` namespace; ``h:t`` is the
#: sub-namespace taster claimed for content it writes itself, and sharing it is
#: correct rather than a collision — both sides address chunks by content, so
#: identical text legitimately resolves to one document.
_CHUNK_PREFIX = "h:t"


class VaultUnavailable(Exception):
    """The vault database cannot be reached, or refused the write."""


def _chunk_id(content: str) -> str:
    return _CHUNK_PREFIX + hashlib.sha1(content.encode("utf-8")).hexdigest()[:24]  # noqa: S324


def _q(doc_id: str) -> str:
    # Entry ids are vault paths and contain "/" — left unencoded, CouchDB parses
    # them as db/doc/attachment segments and the write lands somewhere else.
    return quote(doc_id, safe="")


#: The narration embed `narrate.py` puts under the digest's H1. The audio is
#: ~100 MB a week and stays on the NAS, so in the vault this is a dead embed in
#: the most prominent position on the page — an audio player that never loads.
_AUDIO_EMBED = re.compile(r"^!\[\[([^\]\n]+\.(?:mp3|opus|wav))\]\]$", re.MULTILINE)

#: The digest's plain-text entity line. Deliberately not changed in the template:
#: the file on disk is also read by the console (which would show raw brackets)
#: and by the narrator (which would read them aloud). The vault is the only
#: consumer that wants links, so the conversion belongs here, at that boundary.
_MENTIONED = re.compile(r"^\*\*Mentioned:\*\* (.+)$", re.MULTILINE)

#: Characters that would break out of `[[slug|Label]]` syntax.
_UNLINKABLE = frozenset("|[]")


def entity_slugs(digest_dir: Path) -> set[str]:
    """Filenames in ``entities/``, which is what a wikilink may point at.

    Read from disk rather than recomputed so linking can never invent a target:
    an entity below the note threshold simply stays plain text, and the digest
    gains no dangling links. Cheap — one directory listing per projection run.
    """
    try:
        return {p.stem for p in (digest_dir / ENTITIES_DIR).glob("*.md")}
    except OSError:
        return set()


#: The part of an entry that duplicates the episode's own note: its summary and
#: takeaways. Everything outside stays — the heading, the score, why it matters,
#: the entities. The anchors themselves are defined in `.vault_anchors`, shared
#: with the console, which strips rather than uses them.
_FULL_REGION = re.compile(r"[ \t]*<!-- full -->\n(?P<body>.*?)[ \t]*<!-- /full -->\n?", re.DOTALL)


def slim_digest(markdown: str, episode_notes: dict[str, str]) -> str:
    """Replace each entry's summary with a link to that episode's own note.

    The same text lived twice in the vault — once in the weekly digest, once in
    the episode note — and the digest is 127 KB of mostly that. What the digest
    keeps is what only it has: the week's synthesis, the ranking, why each pick
    earned its place, and the audit table.

    **An entry is only slimmed when its note exists.** Ordering makes that
    essential rather than defensive: the digest is generated at 06:00 on Friday
    and the episode notes at 06:45, so a transform that stripped unconditionally
    would publish a gutted digest pointing at notes that had not been written
    yet. Here a missing note simply means the full text stays.
    """

    # Walked rather than substituted: the region belonging to an anchor is the
    # *next* one after it, which a single regex cannot express without assuming
    # no entry is ever missing its region.
    out: list[str] = []
    cursor = 0
    for anchor in EPISODE_ANCHOR.finditer(markdown):
        out.append(markdown[cursor : anchor.start()])
        cursor = anchor.end()
        note = episode_notes.get(anchor.group("id"))
        region = _FULL_REGION.search(markdown, cursor)
        if region is None:
            continue
        out.append(markdown[cursor : region.start()])
        if note:
            out.append(f"→ [[{note}|Read the full summary]]\n\n")
        else:
            out.append(region.group("body"))
        cursor = region.end()
    out.append(markdown[cursor:])
    # Belt and braces: an entry whose region was missing keeps its text, but must
    # not keep a raw comment where a reader can see it.
    return strip_vault_anchors("".join(out))


def to_vault_markdown(
    markdown: str,
    *,
    known_entities: set[str] | None = None,
    episode_notes: dict[str, str] | None = None,
) -> str:
    """The same digest, written for a vault rather than for a console.

    Three changes, all of which exist because the file on disk serves readers the
    vault does not: the dead audio embed becomes a line of text, entities that
    have a note become links to it, and each entry's summary gives way to a link
    to the episode's own note rather than being stored twice.
    """
    markdown = slim_digest(markdown, episode_notes or {})

    def _audio(match: re.Match[str]) -> str:
        return f"*Narrated audio: `{match.group(1)}` — kept on the server, not synced.*"

    markdown = _AUDIO_EMBED.sub(_audio, markdown)

    if not known_entities:
        return markdown

    def _mentioned(match: re.Match[str]) -> str:
        linked = []
        for raw in match.group(1).split(", "):
            name = raw.strip()
            slug = slugify(name)
            if slug in known_entities and not (_UNLINKABLE & set(name)):
                linked.append(f"[[{slug}|{name}]]")
            else:
                linked.append(name)
        return "**Mentioned:** " + ", ".join(linked)

    return _MENTIONED.sub(_mentioned, markdown)


class LiveSyncVault:
    """Writes files into the vault's CouchDB in LiveSync's document format.

    Its own client rather than the shared :func:`~.net.build_client` one, and
    deliberately not behind :class:`~.net.UrlGuard`: that guard exists to stop
    *feed-supplied* URLs reaching private addresses, and this URL is
    operator-supplied and expected to be a LAN address. Same reasoning as
    :class:`~.speech.OpenAISpeechBackend`.
    """

    def __init__(self, cfg: VaultConfig, password: str | None) -> None:
        self._cfg = cfg
        base = (cfg.couchdb_url or "").rstrip("/")
        self._client = (
            httpx.AsyncClient(
                base_url=base,
                auth=(cfg.user, password or ""),
                timeout=cfg.timeout_s,
            )
            if base
            else None
        )

    @property
    def name(self) -> str:
        return f"vault:{(self._cfg.couchdb_url or '').rstrip('/')}/{self._cfg.db}"

    @property
    def folder(self) -> str:
        return self._cfg.folder

    @property
    def episodes_folder(self) -> str:
        return self._cfg.episodes_folder

    def vault_path(self, relative: Path | str) -> str:
        """Where a file written under ``digest_dir`` lands in the vault.

        Entity notes are routed out of the digest folder and filed flat as
        topics: they are what every digest's links point at, and burying them
        under the raw-capture folder would make the graph read as more podcast
        output rather than as subjects the corpus keeps returning to.
        """
        rel = Path(relative)
        if rel.parts:
            routed = {
                ENTITIES_DIR: self._cfg.entities_folder,
                EPISODES_DIR: self._cfg.episodes_folder,
            }.get(rel.parts[0])
            if routed:
                return f"{routed}/{Path(*rel.parts[1:]).as_posix()}"
        return f"{self._cfg.folder}/{rel.as_posix()}"

    async def _existing_markdown(self, entry_id: str) -> str | None:
        """The note as the vault currently holds it, reassembled from its chunks.

        None when there is no live note — including a soft-deleted one, so a
        topic the reader threw away is not quietly rebuilt by the merge.
        """
        entry = await self._get(entry_id)
        if entry is None or entry.get("deleted"):
            return None
        parts = []
        for chunk_id in entry.get("children") or []:
            chunk = await self._get(str(chunk_id))
            if chunk is None:
                return None  # torn note; safer to rewrite than to merge into half
            parts.append(str(chunk.get("data") or ""))
        return "".join(parts)

    async def project(
        self,
        relative: Path | str,
        markdown: str,
        *,
        mtime_ms: int,
        known_entities: set[str] | None = None,
        episode_notes: dict[str, str] | None = None,
        merge: bool = False,
    ) -> str | None:
        """Write one file into the vault. Returns its vault path, or None if skipped.

        Skipped means the file is already there byte for byte, or a human deleted
        it and that deletion is being respected.

        The text is adapted for a vault on the way in — see
        :func:`to_vault_markdown`. The comparison that decides "already there" is
        made against the adapted text, because that is what the vault holds.
        """
        if self._client is None:
            raise VaultUnavailable("vault.couchdb_url is not set")

        markdown = to_vault_markdown(
            markdown, known_entities=known_entities, episode_notes=episode_notes
        )
        path = self.vault_path(relative)

        if merge:
            # Against the vault, not against our own file: nothing syncs back, so
            # our copy on disk cannot know what a person — or another
            # application — wrote into this note. Merging into it would rewrite
            # the vault from a source blind to the vault's contents.
            current = await self._existing_markdown(path.lower())
            markdown = merge_owned_section(current, markdown)
            if current is not None and markdown == current:
                return None  # our section already says exactly this

        chunk_id = _chunk_id(markdown)

        await self._put_chunk(chunk_id, markdown)

        entry: dict[str, Any] = {
            "_id": path.lower(),  # LiveSync keys entries by lowercased path
            "path": path,
            "children": [chunk_id],
            "ctime": mtime_ms,
            "mtime": mtime_ms,
            "size": len(markdown.encode("utf-8")),
            "type": "plain",
            "eden": {},
        }
        written = await self._put_entry(entry)
        if written:
            log.info("vault.projected", path=path, bytes=entry["size"])
        return path if written else None

    async def _put_chunk(self, chunk_id: str, markdown: str) -> None:
        """Ensure the chunk exists.

        Content-addressed, so a live chunk with this id already holds this exact
        text and is correct as it stands. A soft-deleted one is revived
        unconditionally — unlike an entry, a chunk carries no intent: an entry
        whose children cannot be fetched is a file that renders empty, which is
        worse than either keeping it or deleting it.

        The tombstone branch below is for a race (the chunk was live when we
        wrote and deleted before we looked); an ordinary hard-deleted chunk never
        gets here, because a PUT with no ``_rev`` over a tombstone returns 201.
        """
        body = {"_id": chunk_id, "data": markdown, "type": "leaf"}
        response = await self._put(chunk_id, body)
        if response.status_code != 409:
            return

        existing = await self._get(chunk_id)
        if existing is not None and not existing.get("deleted"):
            return
        rev = existing.get("_rev") if existing else await self._tombstone_rev(chunk_id)
        if rev is None:
            raise VaultUnavailable(f"conflict on chunk {chunk_id} with no revision to take over")
        await self._put_or_raise(chunk_id, {**body, "_rev": rev})

    async def _put_entry(self, entry: dict[str, Any]) -> bool:
        """Write the file entry. False when nothing needed writing."""
        entry_id = str(entry["_id"])
        response = await self._put(entry_id, entry)
        if response.status_code != 409:
            return True

        existing = await self._get(entry_id)
        if existing is None:
            # Raced: the document was live when we tried to write and gone by the
            # time we looked. Not the ordinary hard-delete path — a PUT over a
            # tombstone returns 201 and never reaches here. Leaving it alone is
            # the same call as below, and the loser of a race should not win it.
            log.info("vault.skipped_deleted", path=entry["path"], deletion="raced")
            return False
        if existing.get("deleted"):
            # What deleting a note in Obsidian produces: LiveSync keeps the
            # document and flags it. Respected — see the module docstring.
            log.info("vault.skipped_deleted", path=entry["path"], deletion="soft")
            return False
        if list(existing.get("children") or []) == entry["children"]:
            return False  # already present, byte for byte

        # Same path, different content: a re-render of a file that exists. The
        # generator does not overwrite digests (it writes -r2), so this is the
        # signals file, whose period file is rewritten in place by design.
        await self._put_or_raise(
            entry_id,
            {**entry, "_rev": existing["_rev"], "ctime": existing.get("ctime") or entry["ctime"]},
        )
        return True

    async def _put(self, doc_id: str, body: dict[str, Any]) -> httpx.Response:
        assert self._client is not None
        try:
            response = await self._client.put(f"/{self._cfg.db}/{_q(doc_id)}", json=body)
        except httpx.HTTPError as exc:
            raise VaultUnavailable(f"{self.name} unreachable: {type(exc).__name__}: {exc}") from exc
        if response.status_code in (201, 202, 409):
            return response
        # Credentials never reach the message — only what was attempted.
        raise VaultUnavailable(
            f"{self.name} refused a write: HTTP {response.status_code} {response.text[:200]}"
        )

    async def _put_or_raise(self, doc_id: str, body: dict[str, Any]) -> None:
        response = await self._put(doc_id, body)
        if response.status_code == 409:
            raise VaultUnavailable(f"{self.name}: repeated conflict writing {doc_id}")

    async def _get(self, doc_id: str) -> dict[str, Any] | None:
        assert self._client is not None
        try:
            response = await self._client.get(f"/{self._cfg.db}/{_q(doc_id)}")
        except httpx.HTTPError as exc:
            raise VaultUnavailable(f"{self.name} unreachable: {type(exc).__name__}: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise VaultUnavailable(
                f"{self.name} refused a read: HTTP {response.status_code} {response.text[:200]}"
            )
        doc: dict[str, Any] = response.json()
        return doc

    async def _tombstone_rev(self, doc_id: str) -> str | None:
        """The revision of a hard-deleted document's leaf, so it can be written
        over. A plain GET 404s for these, so the deleted leaf is asked for
        explicitly."""
        assert self._client is not None
        try:
            response = await self._client.get(
                f"/{self._cfg.db}/{_q(doc_id)}",
                params={"open_revs": "all"},
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise VaultUnavailable(f"{self.name} unreachable: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            return None
        for row in response.json():
            ok = row.get("ok") if isinstance(row, dict) else None
            if isinstance(ok, dict) and ok.get("_rev"):
                rev: str = ok["_rev"]
                return rev
        return None

    async def entries_under(self, prefix: str) -> list[dict[str, Any]]:
        """Every live file entry filed under ``prefix``, deepest included.

        ``_all_docs`` over a key range rather than ``_find``: entry ids *are*
        vault paths, so the primary index already sorts them the way this needs
        and the query stays a range scan. A Mango selector on ``path`` would
        scan the whole database, chunks and all, for want of an index nobody
        else needs.
        """
        assert self._client is not None
        start = f"{prefix.rstrip('/').lower()}/"
        try:
            response = await self._client.get(
                f"/{self._cfg.db}/_all_docs",
                params={
                    "include_docs": "true",
                    # CouchDB's own idiom for a prefix scan. Not exact — a path
                    # sorting after U+FFF0 would fall outside the range — but the
                    # miss costs a file left where it is, which is the harmless
                    # direction for a range whose purpose is to find things to
                    # delete. The prefix is rechecked below regardless.
                    "startkey": json.dumps(start),
                    "endkey": json.dumps(start + "\ufff0"),
                },
            )
        except httpx.HTTPError as exc:
            raise VaultUnavailable(f"{self.name} unreachable: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            raise VaultUnavailable(
                f"{self.name} refused a listing: HTTP {response.status_code} {response.text[:200]}"
            )
        entries = []
        for row in response.json().get("rows") or []:
            doc = row.get("doc")
            # Only files, and only ones that are still there: a soft-deleted
            # entry keeps its document, so the deleted flag is the whole check.
            if (
                isinstance(doc, dict)
                and doc.get("type") == "plain"
                and not doc.get("deleted")
                and str(doc.get("_id") or "").startswith(start)
            ):
                entries.append(doc)
        return entries

    async def retire(self, entry: dict[str, Any]) -> bool:
        """Delete a file from the vault the way Obsidian's own deletion does.

        Which is not a CouchDB delete: LiveSync keeps the document and sets
        ``deleted: true``, and every client replicating that flag removes the
        file. Writing a real tombstone instead would leave clients that had
        already synced the file holding it forever.

        Used only to finish a move — the new path is written first, then the old
        one retired. It is emphatically not a general delete: nothing here
        removes a note because it is old, and a digest is never retired at all.
        """
        entry_id = str(entry["_id"])
        response = await self._put(
            entry_id, {**entry, "deleted": True, "mtime": int(time.time() * 1000)}
        )
        if response.status_code == 409:
            # Something wrote this path between the listing and now. Whatever it
            # wrote is newer than what we are trying to remove, so leave it.
            log.info("vault.retire_conflict", path=entry.get("path"))
            return False
        log.info("vault.retired", path=entry.get("path"))
        return True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def build_vault(cfg: VaultConfig, password: str | None) -> LiveSyncVault:
    return LiveSyncVault(cfg, password)


#: What reaches the vault. Weekly digests, reader signals, and the entity notes
#: the digests link to — deliberately not the per-podcast archive notes, which
#: outnumber a personal vault several times over, nor the narration audio.
def projectable(digest_dir: Path) -> list[Path]:
    return sorted(
        [p for p in digest_dir.glob("[0-9][0-9][0-9][0-9]/*.md") if p.is_file()]
        + [p for p in (digest_dir / "signals").glob("*.md") if p.is_file()]
        + [p for p in (digest_dir / ENTITIES_DIR).glob("*.md") if p.is_file()]
        # One level down, always: episode notes are filed under their show. A
        # file loose at the top is left over from the flat layout, and picking
        # it up would project a second copy under the name the grouped one
        # already has.
        + [p for p in (digest_dir / EPISODES_DIR).glob("*/*.md") if p.is_file()]
    )


#: Where this application filed notes before the current layout, and which it
#: therefore has to clear up after. Entries under these are retired once their
#: replacements have been written.
#:
#: The move has to finish, not merely start: two files with one name make every
#: ``[[wikilink]]`` to that name ambiguous, and Obsidian resolves the ambiguity
#: silently, so half a move looks like a working vault that quietly opens the
#: wrong note.
LEGACY_FOLDERS = ("13 podcast-episodes",)


def _under(path: str, folder: str) -> bool:
    return path.lower().startswith(folder.lower().rstrip("/") + "/")


def owned_root(vault: LiveSyncVault) -> str | None:
    """The folder that contains this application's own two folders, if there is one.

    ``11 podcasts/digests`` and ``11 podcasts/episodes`` give ``11 podcasts``,
    and everything under it that is in neither is something we used to write and
    no longer do. None when the two are configured apart, because then there is
    no folder this application can claim to own outright and nothing to sweep.
    """
    ours = vault.folder.split("/")
    theirs = vault.episodes_folder.split("/")
    common: list[str] = []
    for mine, other in zip(ours, theirs, strict=False):
        if mine.lower() != other.lower():
            break
        common.append(mine)
    # A root has to *contain* both. Configured as `11 podcasts` and
    # `11 podcasts/episodes`, the common part is one of them, and sweeping it
    # would mean deleting the digests.
    if not common or len(common) >= min(len(ours), len(theirs)):
        return None
    return "/".join(common)


async def retire_moved(vault: LiveSyncVault, episode_ids: set[str]) -> list[str]:
    """Remove vault files this application wrote at a path it has stopped using.

    Two kinds, and they are found differently because we know different things
    about them:

    * **Episode notes** are rewritten in full on every run, so what is on disk
      is the complete set and anything else under the episodes folder is at an
      old path — a show that renamed itself, or the flat layout that preceded
      grouping by show. ``episode_ids`` is that complete set; empty means the
      writer did not run, and nothing is swept on the strength of a list that
      was never built.
    * **Anything else under the owned root**, plus the folders in
      :data:`LEGACY_FOLDERS`. These are folders wholesale abandoned, so the
      contents go without needing to be compared to anything.

    Digests are never retired. They are not regenerated, so disk is not
    authoritative for them, and pruning one because it is no longer on this
    machine is a deletion nobody asked for.
    """
    prefixes = list(LEGACY_FOLDERS)
    root = owned_root(vault)
    if root:
        prefixes.append(root)

    stale: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        for entry in await vault.entries_under(prefix):
            entry_id = str(entry["_id"])
            path = str(entry.get("path") or entry_id)
            if _under(path, vault.folder):
                continue
            if _under(path, vault.episodes_folder) and (not episode_ids or entry_id in episode_ids):
                continue
            stale[entry_id] = entry

    retired = []
    for entry in stale.values():
        if await vault.retire(entry):
            retired.append(str(entry.get("path") or entry["_id"]))
    return retired


async def sync_all(
    vault: LiveSyncVault,
    digest_dir: Path,
    *,
    episode_notes: dict[str, str] | None = None,
    #: A ceiling, not a page size: above what the corpus holds, so a routine run
    #: always sees all of it. A pass that hits this stops sweeping notes left at
    #: an abandoned path, because it can no longer tell one from a file it
    #: simply did not reach.
    limit: int = 5000,
) -> dict[str, Any]:
    """Project everything that belongs in the vault, in one pass.

    Idempotent, so it is also the cheapest way to make links catch up. A digest
    is written once and never rewritten, so any link that only becomes possible
    later — an entity that earns a topic note, an episode that gains one — would
    otherwise stay plain text in it forever. Running the whole set after each
    rebuild fixes that: unchanged files cost one content comparison and are
    skipped.

    ``episode_notes`` maps episode id to note filename and is what lets each
    digest entry give way to a link. Absent, entries keep their full text, which
    is the safe direction.

    It also finishes moves — see :func:`retire_moved` — but only when it has
    seen the whole corpus.

    Raises :class:`VaultUnavailable`, carrying how far it got.
    """
    everything = projectable(digest_dir)
    candidates = everything[:limit]
    # Once for the pass, not once per file: it decides which mentions become
    # links, and it is the same answer every time.
    known = entity_slugs(digest_dir)
    # Entries are keyed by the lowercased path, so that is what the comparison
    # in `retire_moved` has to be made in.
    episode_ids = {
        vault.vault_path(p.relative_to(digest_dir)).lower()
        for p in candidates
        if p.relative_to(digest_dir).parts[:1] == (EPISODES_DIR,)
    }

    projected: list[str] = []
    skipped = 0
    for path in candidates:
        try:
            relative = path.relative_to(digest_dir)
            written = await vault.project(
                relative,
                path.read_text(encoding="utf-8"),
                mtime_ms=int(path.stat().st_mtime * 1000),
                known_entities=known,
                episode_notes=episode_notes,
                # Topic notes are shared with the reader and with whatever else
                # writes to the vault; digests are ours alone.
                merge=relative.parts[:1] == (ENTITIES_DIR,),
            )
        except VaultUnavailable as exc:
            raise VaultUnavailable(
                f"{exc} (projected {len(projected)} of {len(candidates)} before stopping)"
            ) from exc
        if written:
            projected.append(written)
        else:
            skipped += 1

    # Only after every file has been written, and only when this pass saw the
    # whole corpus. Two ways it might not have, and both would be read as
    # "these notes are gone" by a sweep that trusted the list:
    #
    # * nothing on disk at all — an unmounted volume looks exactly like a corpus
    #   with nothing in it;
    # * a pass that stopped at `limit` — which is not a statement about what
    #   exists, only about how far this call was willing to go.
    #
    # The second is not hypothetical: it retired 566 episode notes the first
    # time this ran through an endpoint whose default limit was below the size
    # of the corpus.
    complete = bool(candidates) and len(candidates) == len(everything)
    if not complete and everything:
        log.info(
            "vault.sweep_skipped",
            considered=len(candidates),
            available=len(everything),
            detail="a truncated pass cannot say what no longer exists",
        )
    retired = await retire_moved(vault, episode_ids) if complete else []

    return {
        "considered": len(candidates),
        "projected": projected,
        "skipped": skipped,
        "retired": retired,
        "linkable_entities": len(known),
    }


async def project_file(
    vault: LiveSyncVault | None,
    base_dir: Path,
    written: Path,
    *,
    known_entities: set[str] | None = None,
) -> None:
    """Project a file just written under ``base_dir``, and never fail because of it.

    The single call every writer uses. It swallows :class:`VaultUnavailable` on
    purpose: the caller has already written the file that matters, and a digest
    that exists on disk must not be reported as a failure because a database on
    another machine was asleep. The next write re-projects, and
    ``POST /api/v1/vault/sync`` catches up whatever was missed.
    """
    if vault is None:
        return
    try:
        await vault.project(
            written.relative_to(base_dir),
            written.read_text(encoding="utf-8"),
            mtime_ms=int(written.stat().st_mtime * 1000),
            known_entities=entity_slugs(base_dir) if known_entities is None else known_entities,
        )
    except VaultUnavailable as exc:
        log.error("vault.deferred", file=str(written), error=str(exc))
    except OSError as exc:
        log.error("vault.unreadable", file=str(written), error=str(exc))
