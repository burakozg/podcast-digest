"""One note per summarised episode — the thing a topic note should link to.

A topic note lists every episode that mentioned an entity. Until now each of
those lines pointed at the *weekly digest* the episode shipped in, and for
anything from the archive it pointed nowhere at all: `digest_weeks` maps only
documents whose id starts ``digest:``, and archive episodes are published under
``archive:<slug>:<month>``. Measured before this existed, 2,210 of 2,702 lines
led nowhere.

The two other applications writing to this vault — security-digest and the
clippings importer — both give each item its own note and link that. This makes
podcast-digest do the same, which is what turns a topic note from a list of
things that were said into something you can read.

Corpus-wide rather than per-digest, deliberately: the archive is most of the
corpus and never passes through a weekly digest, so a writer hung off digest
generation could only ever cover the recent slice. It replaces the per-digest
writer that used to live in :mod:`.generate` — two writers producing notes for
the same episodes in different layouts is how they drift apart.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

from ..config import Settings
from ..db import Doc, Store, typed_sort
from ..entities import EPISODE_NAMES_DOC_ID, pin_note_names
from ..logging_setup import get_logger
from ..notes import EPISODES_DIR
from ..sanitize import slugify
from .generate import BASIS_LABELS, _build_env, summary_view

log = get_logger(__name__)

#: Documents read per page while walking the corpus. Matches `entities.BATCH`;
#: both walks read the same collection for the same reason.
BATCH = 500


def note_name(episode: Doc) -> str:
    """The filename an episode's note would get if it were being named today.

    Only ever consulted for an episode that has no pinned name — see
    :func:`resolve_episode_note_names`. Shaped like security-digest's story
    notes (``<date>-<title-slug>``) so the two read alike in one folder listing.
    """
    published = str(episode.get("published_at") or "")[:10] or "undated"
    # Tested against the *raw* title, not its slug: `slugify("")` answers
    # "untitled", so a slug-based check never fires and every untitled episode
    # published on one date would collide onto a single note.
    raw = str(episode.get("title") or "").strip()
    # The id is unique and stable, which is what a name needs to be; readability
    # is already lost for an episode nobody titled.
    title = slugify(raw) if raw else slugify(str(episode.get("_id") or "episode"))
    # No truncation here: `slugify` already caps length, and a second cap on top
    # is one more rule to keep in step with it for no gain.
    return f"{published}-{title}"


#: Characters no filesystem this vault syncs to will accept in a path segment.
#: the Windows illegal set, which is the strictest of the three (macOS objects only
#: to ``/``, iOS to ``/`` and ``:``) — a folder named for a show has to survive on
#: whichever device opens the vault, not just the one that wrote it.
_UNSAFE_IN_PATH = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')

_SPACES = re.compile(r"\s+")


def show_folder(episode: Doc) -> str:
    """The folder an episode's note is filed under: its show, readably.

    Not a slug. This is the one part of the layout a person reads as a folder
    name rather than clicks, and ``risky-business`` beside ``Risky Business``
    elsewhere in the vault would look like two different things.

    Unpinned, deliberately, and this is safe only because of how links work:
    Obsidian resolves ``[[wikilinks]]`` by filename, so a show that renames
    itself moves its notes without breaking a single link. What the rename does
    leave is the notes at the old path, which is why :func:`write_episode_notes`
    prunes and the projection retires — two files with one name would make every
    link to that name ambiguous, and Obsidian picks one silently.
    """
    raw = str(episode.get("podcast_name") or "").strip()
    # Replaced with a space rather than dropped: "AI | Security" collapsing to
    # "AI Security" reads; collapsing to "AISecurity" does not.
    cleaned = _SPACES.sub(" ", _UNSAFE_IN_PATH.sub(" ", raw)).strip(" .")
    # Trailing dots and spaces are legal to create on Linux and unopenable on
    # Windows, so they are stripped above rather than truncated into existence.
    return cleaned[:80].strip(" .") or slugify(str(episode.get("podcast_slug") or "show"))


async def resolve_episode_note_names(store: Store, episodes: list[Doc]) -> dict[str, str]:
    """``episode_id -> filename``, choosing a name only for the unnamed.

    Pinned for the same reason topic notes are: publishers edit titles, and a
    name recomputed from a title moves the note out from under every link that
    points at it. ``episode_id`` never changes, so it is the key.

    Names are unique across the whole corpus, not merely within a show. Filing
    per show would let two shows that published "Week in review" on one day
    coexist on disk — and then a ``[[2026-08-21-week-in-review]]`` in a topic
    note would have two possible targets, with Obsidian choosing between them
    and saying nothing.
    """
    pinned = await pinned_episode_names(store)
    taken = set(pinned.values())
    proposed: dict[str, str] = {}
    for episode in episodes:
        episode_id = str(episode["_id"])
        if episode_id in pinned:
            continue
        name = _unclaimed(note_name(episode), episode, taken)
        taken.add(name)
        proposed[episode_id] = name
    if not proposed:
        return pinned
    return await pin_note_names(store, proposed, doc_id=EPISODE_NAMES_DOC_ID)


def _unclaimed(name: str, episode: Doc, taken: set[str]) -> str:
    """``name``, or the nearest variant of it nothing else has been given."""
    if name not in taken:
        return name
    # The show first, because it says which of the two collided notes this is;
    # a bare counter would leave the reader to open both to find out.
    show = slugify(str(episode.get("podcast_slug") or ""), fallback="")
    if show and f"{name}-{show}" not in taken:
        return f"{name}-{show}"
    suffix = 2
    while f"{name}-{suffix}" in taken:
        suffix += 1
    return f"{name}-{suffix}"


async def pinned_episode_names(store: Store) -> dict[str, str]:
    """``episode_id -> filename`` as already recorded, without writing anything.

    What the projection uses. Reading the pin rather than re-deriving means a
    catch-up run can slim a digest correctly at any time, whether or not the
    note-writing job has run since.
    """
    doc = await store.get(EPISODE_NAMES_DOC_ID)
    return dict((doc or {}).get("names") or {})


async def summarised_episodes(store: Store, *, limit_docs: int = 20_000) -> list[Doc]:
    """Every episode carrying a Tier-1 summary, newest first.

    Includes episodes scored below the digest threshold. They contribute
    entities to topic notes exactly as any other summarised episode does, so
    excluding them would leave a second class of line that leads nowhere and is
    indistinguishable from the first.
    """
    found: list[Doc] = []
    skip = 0
    while skip < limit_docs:
        page = await store.find(
            {"type": "episode"},
            sort=typed_sort("published_at", "desc"),
            limit=BATCH,
            skip=skip,
        )
        if not page:
            break
        for episode in page:
            if (episode.get("tier1") or {}).get("summary_md"):
                found.append(episode)
        skip += len(page)
        if len(page) < BATCH:
            break
    return found


async def write_episode_notes(
    store: Store,
    settings: Settings,
    *,
    week_of: dict[str, str] | None = None,
) -> dict[str, str]:
    """Write one note per summarised episode. Returns ``episode_id -> filename``.

    The mapping is the point as much as the files are: it is what lets a topic
    note link the episode rather than the digest it shipped in.
    """
    episodes = await summarised_episodes(store)
    if not episodes:
        return {}

    names = await resolve_episode_note_names(store, episodes)
    directory = settings.output.digest_dir / EPISODES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    template = _build_env().get_template("episode.md.j2")
    weeks = week_of or {}

    current: set[Path] = set()
    for episode in episodes:
        episode_id = str(episode["_id"])
        view: dict[str, Any] = summary_view(settings, episode, BASIS_LABELS)
        # Only a weekly digest gives a week to link back to. Archive episodes
        # have none, and the template omits the line rather than emitting a
        # link to a note that was never written.
        week = weeks.get(str(episode.get("digest_id") or ""))
        rendered = template.render(e=view, week=week)
        path = directory / show_folder(episode) / f"{names[episode_id]}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Not `_atomic_write`: that never overwrites, and these notes are
        # rewritten in place as summaries and entity links change. Same temp
        # file + rename, so a sync client still never sees a partial file.
        _replace(path, rendered)
        current.add(path)

    removed = _prune(directory, current)
    log.info(
        "episode_notes.written", count=len(current), removed=len(removed), directory=str(directory)
    )
    return {episode_id: names[episode_id] for episode_id in (str(e["_id"]) for e in episodes)}


def _prune(directory: Path, current: set[Path]) -> list[Path]:
    """Delete notes under ``directory`` that this run did not write.

    Every summarised episode is written on every run, so anything left over is
    a note at a path we have stopped using — a show that renamed itself, or the
    flat layout this folder had before it was grouped by show. Leaving those
    behind would put the same filename in the vault twice, and a ``[[wikilink]]``
    with two possible targets resolves to one of them without saying which.

    Only ``.md``, and only ours: the folder holds nothing else, and a prune that
    reaches wider than the writer it belongs to is how an unrelated file gets
    deleted by something that never claimed to manage it.
    """
    removed = []
    for path in directory.rglob("*.md"):
        if path in current or not path.is_file():
            continue
        path.unlink()
        removed.append(path)
    for folder in sorted(directory.rglob("*"), reverse=True):
        # A show folder emptied by the loop above, rather than every empty
        # directory anywhere — `rmdir` refuses a folder with anything in it.
        if folder.is_dir():
            with contextlib.suppress(OSError):
                folder.rmdir()
    return removed


def _replace(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
