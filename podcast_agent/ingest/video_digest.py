"""Mirror video-digest's finished summaries in, as read-only episodes.

video-digest summarises videos and writes its own vault notes. This app is
used purely as the **reader** for them: `/admin/episodes` already has
read/unread, starring, search and export, and none of that exists in the
vault. Nothing here re-summarises, re-scores, or publishes anything.

Three properties make that true, and each is load-bearing:

* **`origin: IMPORTED_ORIGIN`.** Every pipeline and digest selector carries
  `ROUTINE_ONLY` (see `state.py`), so a third origin is invisible to all of
  them with no change to any of them. `migrate.backfill_origins` stamps an
  origin onto any episode lacking one at startup, which would silently
  convert these into routine episodes — so it is always written explicitly.
* **`status: IMPORTED`.** Absent from `ACTIVE_STATUSES`, from
  `DIGESTABLE_STATUSES`/`CLAIMABLE_STATUSES` (so imports are not counted as
  work awaiting a digest) and from the `SURFACED`/`ELIGIBLE` sets that drive
  show ranking. It has no `ALLOWED_TRANSITIONS` entry, so anything that tries
  to move an import through the pipeline raises `IllegalTransition` rather
  than quietly succeeding.
* **`OURS_ONLY` on the two vault writers** (`digest/episode_notes.py`,
  `entities.py`). Those select every episode carrying a summary, regardless of
  origin, and would otherwise write a *second* copy of a note video-digest has
  already written. That is the duplication this design exists to avoid.

Pulling rather than being pushed: this app's CouchDB is bound to loopback on
purpose, so nothing outside the NAS host can write to it, and polling is what
the rest of this ingest package already does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import Settings
from ..db import Doc, Store, update_doc
from ..logging_setup import get_logger
from ..sanitize import html_to_text, safe_url
from ..state import IMPORTED_ORIGIN, EpisodeStatus
from ..utils import episode_doc_id, iso, iso_now, podcast_doc_id

log = get_logger(__name__)

#: The synthetic show every imported video hangs off.
SHOW_SLUG = "video-digest"
SHOW_NAME = "Video Digest"

#: Rows per request. video-digest caps at 200.
PAGE = 50

#: Stop after this many pages in one run — a bound on a first import over a
#: long backlog, not an expectation. The next run continues from the top.
MAX_PAGES = 40


@dataclass(frozen=True, slots=True)
class ImportStats:
    fetched: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0

    def plus(self, **kw: int) -> ImportStats:
        # asdict, not __dict__: slots=True means there is no instance dict.
        current = asdict(self)
        return ImportStats(**{**current, **{k: current[k] + v for k, v in kw.items()}})


def _summary_fields(digest: dict[str, Any]) -> dict[str, Any]:
    """video-digest's `VideoDigest` in this app's `tier1` shape.

    `entities` is deliberately left empty. `entities.aggregate` is excluded by
    origin already, but an empty list means that even if that exclusion were
    ever removed, an import still contributes no mentions to a shared
    `99 topics/` note — belt as well as braces, since the failure is silent
    and lands in another application's file.
    """
    return {
        "summary_md": str(digest.get("summary_md") or ""),
        "why_it_matters": str(digest.get("tldr") or ""),
        "key_takeaways": [str(p) for p in (digest.get("key_points") or [])],
        "entities": [],
        "summary_basis": "imported",
        "relevance_score": _score(str(digest.get("relevance") or "")),
    }


#: video-digest scores relevance as a word; the console renders `n/10`.
_RELEVANCE = {"critical": 10, "high": 8, "medium": 5, "low": 2}


def _score(relevance: str) -> int:
    return _RELEVANCE.get(relevance.lower(), 5)


def _published_at(video: dict[str, Any]) -> str | None:
    """When the video was released — not when we happened to import it.

    The console sorts this column and every other show fills it with a release
    date, so putting an import timestamp here would make one column mean two
    things with nothing on screen saying which. It also flattens real history:
    measured on the first import, all 28 videos spanned thirteen months of
    upload dates and collapsed into two days of import dates, which is how a
    seventeen-talk conference reads as seventeen unrelated items.

    `upload_date` is a bare date, so it is widened to midnight UTC rather than
    left to sort as a short string against full timestamps. Both the
    `YYYY-MM-DD` video-digest sends today and yt-dlp's own `YYYYMMDD` are
    accepted; anything else falls back to `written_at`, because a missing or
    unparseable date must not silently become 1970 and bury the episode.
    """
    raw = str((video.get("metadata") or {}).get("upload_date") or "").strip()
    compact = raw.replace("-", "")
    if len(compact) == 8 and compact.isdigit():
        try:
            day = datetime.strptime(compact, "%Y%m%d").replace(tzinfo=UTC)
        except ValueError:  # a real-looking date that is not one, e.g. 20260231
            return video.get("written_at")
        return iso(day)
    # No usable upload date: when video-digest wrote the summary is the closest
    # honest stand-in, and it is never `updated_at`, which a re-render moves.
    return video.get("written_at")


def _episode_doc(video: dict[str, Any]) -> Doc:
    metadata = video.get("metadata") or {}
    digest = video.get("digest") or {}
    video_id = str(video.get("video_id") or "")
    # The row id, not the bare video id: it is what video-digest's own export
    # is keyed by, and it is adapter-qualified, so a future non-YouTube
    # adapter cannot collide with a YouTube id.
    guid = str(video.get("id") or video_id)
    return {
        "_id": episode_doc_id(SHOW_SLUG, guid),
        "type": "episode",
        "origin": IMPORTED_ORIGIN,
        "status": EpisodeStatus.IMPORTED.value,
        "podcast_slug": SHOW_SLUG,
        "podcast_name": SHOW_NAME,
        "guid": guid,
        "title": html_to_text(str(metadata.get("title") or ""), max_chars=500) or "(untitled)",
        "link": safe_url(f"https://www.youtube.com/watch?v={video_id}") or "",
        "description_raw": str(digest.get("tldr") or ""),
        "published_at": _published_at(video),
        "duration_s": metadata.get("duration_s"),
        "tier1": _summary_fields(digest),
        # Explicitly null rather than absent: `_awaiting_digest` and the
        # console both read it, and Mango cannot match a missing field.
        "digest_id": None,
        "starred": False,
        "starred_at": None,
        "read_at": None,
        # No tier0 (it would skew backfill/estimate's escalation ratio), no
        # transcript_at (retention.py keys off it), no feedback key at all
        # (signals.collect selects on `{"$exists": True}`, which a null would
        # match), no archive_month.
        "source_app": "video-digest",
        "source_note_path": video.get("note_path"),
        "created_at": iso_now(),
        "updated_at": iso_now(),
    }


#: What a re-import refreshes. Everything absent from this list is the
#: reader's, not the exporter's — `read_at`, `starred`, `starred_at` and
#: `feedback` are marks a person made here and must survive a re-import, which
#: is the whole point of mirroring rather than rendering on demand.
_REFRESHED = (
    "title",
    "link",
    "description_raw",
    "published_at",
    "duration_s",
    "tier1",
    "source_note_path",
    "status",
    "origin",
    "podcast_slug",
    "podcast_name",
)


async def _upsert(store: Store, doc: Doc) -> str:
    """Create, or refresh content while leaving the reader's marks alone.

    Read-modify-write through `update_doc`, so a concurrent star or read from
    the console retries rather than being clobbered.
    """
    if await store.create(doc):
        return "created"

    changed = False

    def mutate(current: Doc) -> Doc:
        nonlocal changed
        for key in _REFRESHED:
            if current.get(key) != doc[key]:
                current[key] = doc[key]
                changed = True
        if changed:
            current["updated_at"] = iso_now()
        return current

    await update_doc(store, str(doc["_id"]), mutate)
    return "updated" if changed else "unchanged"


async def ensure_show(store: Store) -> None:
    """The `type: "podcast"` doc, so the console's show filter lists it.

    `overrides.enabled: false` matters: `FeedIngester.run` polls every enabled
    console-source podcast, and this show has no real feed to poll. Top-level
    `feed_url` and `overrides.feed_url` are kept identical so
    `seed_podcast_docs` never rewrites the doc.
    """
    doc_id = podcast_doc_id(SHOW_SLUG)
    if await store.get(doc_id) is not None:
        return
    placeholder = "https://video-digest.invalid/not-a-feed"
    await store.create(
        {
            "_id": doc_id,
            "type": "podcast",
            "slug": SHOW_SLUG,
            "name": SHOW_NAME,
            "feed_url": placeholder,
            "source": "console",
            "overrides": {
                "name": SHOW_NAME,
                "feed_url": placeholder,
                # Never polled: the episodes arrive by import, and the URL
                # above resolves nowhere on purpose.
                "enabled": False,
            },
            "etag": None,
            "last_modified": None,
            "last_polled_at": None,
            "last_error": None,
            "consecutive_failures": 0,
            "created_at": iso_now(),
        }
    )
    log.info("video_digest.show_created", slug=SHOW_SLUG)


class VideoDigestImporter:
    def __init__(self, settings: Settings, store: Store) -> None:
        self._settings = settings
        self._store = store

    def _api_key(self) -> str | None:
        key = self._settings.video_digest_api_key
        return key.get_secret_value() if key else None

    def configured(self) -> bool:
        cfg = self._settings.video_digest
        return bool(cfg.enabled and cfg.base_url and self._api_key())

    async def run(self) -> ImportStats:
        """One pass: page through the export, upsert each row."""
        cfg = self._settings.video_digest
        if not self.configured():
            log.info("video_digest.skipped", reason="not configured")
            return ImportStats()

        await ensure_show(self._store)
        stats = ImportStats()
        params: dict[str, Any] = {"limit": PAGE}

        async with httpx.AsyncClient(
            base_url=str(cfg.base_url).rstrip("/"),
            headers={"X-API-Key": self._api_key() or ""},
            timeout=cfg.timeout_s,
        ) as client:
            for _ in range(MAX_PAGES):
                response = await client.get("/videos", params=params)
                response.raise_for_status()
                body = response.json()
                videos = body.get("videos") or []
                for video in videos:
                    outcome = await _upsert(self._store, _episode_doc(video))
                    stats = stats.plus(fetched=1, **{outcome: 1})
                nxt = body.get("next")
                if not nxt:
                    break
                # httpx encodes these; the `+` in a `+00:00` offset must not
                # reach the wire raw, or it decodes as a space server-side.
                params = {"limit": PAGE, "since": nxt["since"], "since_id": nxt["since_id"]}

        log.info(
            "video_digest.imported",
            fetched=stats.fetched,
            created=stats.created,
            updated=stats.updated,
            unchanged=stats.unchanged,
        )
        return stats
