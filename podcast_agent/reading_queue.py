"""The unread queue as a vault note you can tick on a phone, offline.

The console at `/admin/episodes` already has unread-only browsing, starring and
read marking. None of it is reachable from a phone that is not on the LAN, and
making it reachable means a service worker, which means HTTPS, which means a
certificate for a domain that cannot have one. The vault is already on the
phone — Self-hosted LiveSync replicates it to Obsidian, and every summarised
episode already has a note there (`digest/episode_notes.py`), as does every
imported video, written by video-digest itself.

So the missing piece is not the reading material. It is *which* of it is unread,
and somewhere to tap. Both fit in one note:

    11 podcasts/_unread.md

listing the unread summaries as Markdown tasks. Ticking one marks it read here;
unticking one under "Recently read" puts it back. Each item carries a second,
indented task for the star. Obsidian's reading view toggles a checkbox with one
tap, offline, and LiveSync replicates the edited file whenever the phone next
sees home — which is the entire synchronisation mechanism, already deployed,
already backed up, and needing no new network path.

**This is the only file this application reads back out of the vault.**
Everything else it writes is output. That makes the direction of truth the one
thing to get right, and the rule is:

* a box whose state differs from what we last *wrote* is a person's tap, and it
  wins — it is applied to the episode document;
* a box that matches what we wrote carries no intent, so the database wins and
  the note is re-rendered from it.

What we last wrote is recorded in :data:`QUEUE_DOC_ID` rather than inferred, so
a console change and a phone tap between two polls do not have to be told apart
by guesswork.

Three things this deliberately does not do:

* **It does not timestamp the tap.** Nothing records when a checkbox was ticked,
  only that the file changed, so `read_at` is the time the poll saw it. A week
  offline lands as one moment, and `signals.py` reads that as a reading time it
  is not. Recorded here because it is invisible in the data.
* **It does not list an item whose note is not in the vault.** A queue entry
  that cannot be opened is worse than an absent one, and the note's path is also
  the only stable key a tick can be matched back by — see :func:`parse`.
* **It does not resurrect itself.** Delete the note in Obsidian and the write is
  skipped like any other soft-deleted file; the queue stops updating until the
  note is restored. That is the vault writer's existing promise and this keeps
  it rather than making an exception for its own file.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .db import ConflictError, Doc, NotFoundError, Store, typed_sort, update_doc
from .digest.episode_notes import pinned_episode_names, show_folder
from .feedback import set_read, set_starred
from .logging_setup import get_logger
from .notes import EPISODES_DIR, KEY_PREFIX, merge_owned_section, wrap
from .state import IMPORTED_ORIGIN
from .utils import iso_now, parse_iso
from .vault import LiveSyncVault, VaultUnavailable

log = get_logger(__name__)

#: Where the last-written state of the note is kept: what each box said when we
#: rendered it, and which episode each link belongs to. Without it a tick and a
#: console change are indistinguishable, and the note's own text cannot say
#: which of the two happened.
QUEUE_DOC_ID = "control:reading_queue"

#: Episodes read per page while walking the corpus, and the ceiling on how far
#: the walk goes. The walk exists because "has a summary and is unread" cannot
#: be selected on: `tier1.summary_md` is nested and absent far more often than
#: present, and `read_at` is absent rather than null on an episode nobody has
#: touched, which Mango cannot index either. Both are answered in Python over a
#: bounded slice of the corpus — the same compromise `api/routes.py` makes for
#: the console's own unread filter. The cap is several times the size of this
#: corpus, so a routine run always sees all of it; a run that hits it lists the
#: newest it found and says so, rather than quietly showing a slice.
BATCH = 500
MAX_SCAN = 5000

#: The star's marker. A second checkbox on the same line is not task syntax —
#: only the leading one counts — so the star is an indented task of its own.
STAR = "⭐"

#: A Markdown task line, however it is indented. Obsidian writes a tab; a phone
#: keyboard or another editor may not, so indentation never decides what a line
#: means — order does (see :func:`parse`).
_TASK = re.compile(r"^\s*[-*]\s+\[(?P<box>[ xX])\]\s*(?P<rest>.*)$")

#: The target of a wikilink, up to the alias. Greedy to the LAST `]]` on the
#: line, not `[^\]]*`: a title can itself contain a bracket pair — "[un]prompted"
#: is a real one in this vault — and a lazy match stops at that inner bracket,
#: so the line silently fails to match and the item it names can never be
#: ticked. Titles are stripped of brackets on the way out (:func:`_label`) too;
#: this is the half that protects links written before that.
_LINK = re.compile(r"\[\[(?P<target>[^\]|]+)(?:\|.*)?\]\]")

#: Characters that would break out of `[[path|Label]]`.
_UNLINKABLE = str.maketrans({"[": "", "]": "", "|": "-"})


@dataclass(frozen=True, slots=True)
class Item:
    """One line of the queue: an episode, and where its note is."""

    episode_id: str
    title: str
    show: str
    published_at: str | None
    note_path: str
    starred: bool
    read_at: str | None
    video: bool

    @property
    def key(self) -> str:
        """What a tick is matched back by. See :func:`parse`."""
        return self.note_path.lower()


def _label(text: str) -> str:
    return (text or "").translate(_UNLINKABLE).strip() or "(untitled)"


def _day(published_at: str | None) -> str:
    moment = parse_iso(published_at)
    return moment.strftime("%d %b %Y") if moment else "undated"


def _lines(item: Item) -> list[str]:
    """An item as its two task lines: the item itself, then its star."""
    box = "x" if item.read_at else " "
    star = "x" if item.starred else " "
    meta = " · ".join(part for part in (item.show, _day(item.published_at)) if part)
    return [
        f"- [{box}] [[{item.note_path}|{_label(item.title)}]] · {meta}",
        f"\t- [{star}] {STAR}",
    ]


def render(*, unread: list[Item], recently_read: list[Item], unread_total: int) -> str:
    """The note as we would write it fresh: frontmatter, title, our region.

    Only the region between the ownership markers is replaced on a later run
    (see :func:`~.notes.merge_owned_section`), so anything the reader writes
    above or below it in this note is theirs and survives. The frontmatter count
    is a prefixed key for the same reason — it is ours to replace — while
    `type:` is seeded once and never overwritten.
    """
    body: list[str] = [
        "Tick a box to mark it read, here and in the console. The indented box "
        "is the star. Untick one under **Recently read** to put it back.",
    ]

    videos = [item for item in unread if item.video]
    podcasts = [item for item in unread if not item.video]
    for heading, items in (("Podcasts", podcasts), ("Videos", videos)):
        if not items:
            continue
        body += ["", f"## {heading} · {len(items)}", ""]
        for item in items:
            body += _lines(item)

    hidden = unread_total - len(unread)
    if hidden > 0:
        # Said out loud rather than left to be noticed: a queue that silently
        # stops at a limit reads exactly like a queue that has been finished.
        body += ["", f"*{hidden} older unread not listed — read those in the console.*"]

    if recently_read:
        body += ["", f"## Recently read · {len(recently_read)}", ""]
        for item in recently_read:
            body += _lines(item)

    if not unread and not recently_read:
        body += ["", "*Nothing unread.*"]

    region = wrap("\n".join(body))
    front = f"---\ntype: reading-queue\n{KEY_PREFIX}unread: {unread_total}\n---\n"
    return f"{front}\n# Unread\n\n{region}\n"


def parse(markdown: str) -> dict[str, dict[str, bool]]:
    """``{note path lowercased -> {"read": …, "starred": …}}`` as the note says.

    Keyed by the wikilink target rather than by an id hidden in a comment: the
    target is already on the line, is unique per note, and stays readable when
    the reader opens the file in source mode. It is matched back to an episode
    through the map recorded when the note was written, so a note renamed
    between two polls goes unmatched and is logged, rather than a tick landing
    on the wrong episode.

    A task line carrying a link starts an item; a task line without one belongs
    to the item above it. Indentation is not consulted — Obsidian writes a tab,
    a phone keyboard may write spaces, and a reflowed list must not silently
    stop being parseable.

    ``starred`` is absent, not False, when an item has no star line. A reader who
    deletes that line has said nothing about the star, and reporting False there
    would unstar the episode on the next poll.
    """
    marks: dict[str, dict[str, bool]] = {}
    current: str | None = None
    for line in markdown.splitlines():
        task = _TASK.match(line)
        if task is None:
            continue
        checked = task.group("box").lower() == "x"
        link = _LINK.search(task.group("rest"))
        if link is not None:
            current = link.group("target").strip().lower()
            marks[current] = {"read": checked}
        elif current is not None and STAR in task.group("rest"):
            marks[current]["starred"] = checked
    return marks


async def _episodes(store: Store) -> list[Doc]:
    """Every episode, newest first, up to :data:`MAX_SCAN`."""
    found: list[Doc] = []
    while len(found) < MAX_SCAN:
        page = await store.find(
            {"type": "episode"},
            sort=typed_sort("published_at", "desc"),
            limit=BATCH,
            skip=len(found),
        )
        found += page
        if len(page) < BATCH:
            return found
    log.info(
        "reading_queue.scan_capped",
        scanned=len(found),
        detail="older unread episodes may be missing from the queue",
    )
    return found


def _note_path(
    episode: Doc, settings: Settings, vault: LiveSyncVault, pinned: dict[str, str]
) -> str | None:
    """Where this episode's note is in the vault, or None if it has none.

    Two branches, because these are two applications' notes. An imported video's
    note belongs to video-digest, which tells us its path on import; a podcast
    episode's note is ours, and its name is the one *pinned* for it — never
    recomputed, which is the rule the filename contract exists for. The path is
    assembled by the projection's own routing rather than rebuilt here, so the
    link cannot drift from where the note is actually written.
    """
    if str(episode.get("origin") or "") == IMPORTED_ORIGIN:
        return str(episode.get("source_note_path") or "").strip() or None
    if not settings.output.episode_notes:
        # The notes are not being written at all, so every link would dangle.
        return None
    name = pinned.get(str(episode["_id"]))
    if not name:
        return None
    return vault.vault_path(Path(EPISODES_DIR) / show_folder(episode) / f"{name}.md")


def _item(episode: Doc, note_path: str) -> Item:
    return Item(
        episode_id=str(episode["_id"]),
        title=str(episode.get("title") or ""),
        show=str(episode.get("podcast_name") or ""),
        published_at=episode.get("published_at"),
        note_path=note_path,
        starred=bool(episode.get("starred")),
        read_at=episode.get("read_at") or None,
        video=str(episode.get("origin") or "") == IMPORTED_ORIGIN,
    )


def _summarised(episode: Doc) -> bool:
    return bool((episode.get("tier1") or {}).get("summary_md"))


async def _apply_taps(
    store: Store, seen: dict[str, dict[str, bool]], shadow: Doc
) -> list[dict[str, Any]]:
    """Apply every box whose state differs from what we last wrote.

    A box that still says what we wrote is not evidence of anything — it is a
    line nobody touched — so it is left alone and the database stays
    authoritative for it. Only a difference is a person's tap.
    """
    links: dict[str, str] = dict(shadow.get("links") or {})
    written: dict[str, dict[str, bool]] = dict(shadow.get("marks") or {})
    applied: list[dict[str, Any]] = []

    for key, state in seen.items():
        episode_id = links.get(key)
        if episode_id is None:
            # A link we did not write: the reader's own, or a note renamed
            # between two polls. Never guessed at — a tick applied to the wrong
            # episode is worse than one that does nothing.
            log.info("reading_queue.unknown_line", link=key)
            continue
        before = written.get(episode_id) or {}
        for field, setter in (("read", set_read), ("starred", set_starred)):
            if field not in state or field not in before or state[field] == before[field]:
                continue
            try:
                await setter(store, episode_id, state[field])
            except (NotFoundError, ConflictError) as exc:
                # One unwritable episode must not cost the other taps in the
                # same poll; the box keeps its state and the next poll retries.
                log.warning(
                    "reading_queue.tap_failed",
                    episode_id=episode_id,
                    field=field,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            applied.append({"episode_id": episode_id, field: state[field]})
    if applied:
        log.info("reading_queue.taps_applied", count=len(applied))
    return applied


def _recently_read(
    shadow: Doc, queued: dict[str, Item], applied: list[dict[str, Any]], keep: int
) -> list[Item]:
    """The last few items to leave the queue, so a tick can be undone.

    Membership is remembered rather than queried: "recently read" means read
    *out of this queue*, which no index can answer — `read_at` is not indexed,
    and an episode read in the console a year after it was published sorts
    nowhere near the walk that found it. Anything read while it was listed here
    goes to the front, whether it was ticked on the phone or marked in the
    console.
    """
    order: list[str] = [str(change["episode_id"]) for change in applied if change.get("read")]
    order += [str(episode_id) for episode_id in (shadow.get("recently_read") or [])]

    seen: set[str] = set()
    kept: list[Item] = []
    for episode_id in order:
        item = queued.get(episode_id)
        # An item since marked unread is in the unread list again, and must not
        # also sit under "Recently read" — one item, one box.
        if item is None or not item.read_at or episode_id in seen:
            continue
        seen.add(episode_id)
        kept.append(item)
        if len(kept) >= keep:
            break
    return kept


async def _save(store: Store, listed: list[Item], recently_read: list[Item]) -> None:
    """Record what the note now says, so the next poll can spot a tap."""
    marks = {
        item.episode_id: {"read": bool(item.read_at), "starred": item.starred} for item in listed
    }

    def _write(doc: Doc) -> None:
        doc.setdefault("type", "control")
        doc.setdefault("key", QUEUE_DOC_ID.split(":", 1)[-1])
        doc["marks"] = marks
        doc["links"] = {item.key: item.episode_id for item in listed}
        doc["recently_read"] = [item.episode_id for item in recently_read]
        doc["updated_at"] = iso_now()

    try:
        await update_doc(store, QUEUE_DOC_ID, _write)
    except NotFoundError:
        seed: Doc = {"_id": QUEUE_DOC_ID}
        _write(seed)
        try:
            await store.put(seed)
        except ConflictError:
            # Another poll created it first; fold ours into theirs.
            await update_doc(store, QUEUE_DOC_ID, _write)


async def sync_reading_queue(
    store: Store, settings: Settings, vault: LiveSyncVault | None
) -> dict[str, Any]:
    """Absorb the note's taps, then re-render it from the database.

    In that order, and it matters: rendering first would overwrite a tap made
    while the previous render was still the newest thing the phone had.

    A vault that will not answer is not an error — the taps are still in the
    note, and the next poll reads them. Same contract as every other vault
    writer here: the database is the deliverable, the vault is a projection.
    """
    if vault is None or not settings.vault.enabled:
        return {"skipped": "vault.enabled is false"}

    path = settings.vault.queue_note
    shadow = (await store.get(QUEUE_DOC_ID)) or {}

    try:
        current = await vault.read_note(path)
        applied = await _apply_taps(store, parse(current), shadow) if current else []

        pinned = await pinned_episode_names(store)
        items: list[Item] = []
        unread_total = 0
        for episode in await _episodes(store):
            if not _summarised(episode):
                continue
            note_path = _note_path(episode, settings, vault, pinned)
            if note_path is None:
                continue
            item = _item(episode, note_path)
            if item.read_at is None:
                unread_total += 1
            items.append(item)

        unread = [item for item in items if item.read_at is None][: settings.vault.queue_limit]
        recent = _recently_read(
            shadow,
            {item.episode_id: item for item in items},
            applied,
            settings.vault.queue_read_limit,
        )

        merged = merge_owned_section(
            current, render(unread=unread, recently_read=recent, unread_total=unread_total)
        )
        # Nothing to say is not a reason to write: an unchanged note would still
        # be a new revision, and every device replicating this vault would pull
        # it. The same rule the rest of the projection follows.
        written = (
            await vault.write_note(path, merged, mtime_ms=int(time.time() * 1000))
            if merged != current
            else None
        )
    except VaultUnavailable as exc:
        log.error("reading_queue.deferred", error=str(exc))
        return {"note": path, "error": str(exc)}

    await _save(store, unread + recent, recent)

    log.info(
        "reading_queue.synced",
        note=path,
        listed=len(unread),
        unread_total=unread_total,
        recently_read=len(recent),
        applied=len(applied),
        written=bool(written),
    )
    return {
        "note": path,
        "listed": len(unread),
        "unread_total": unread_total,
        "recently_read": len(recent),
        "applied": applied,
        "written": bool(written),
    }
