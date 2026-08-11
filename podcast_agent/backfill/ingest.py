"""Archive backfill ingestion (roadmap A1).

Walks each show's feed backwards one month-window at a time, tracked by a cursor
on the podcast document so a run is bounded, resumable and stoppable. Routine
ingestion only ever looks forward (see ``Ingestor._cutoff_for``); reaching into
the archive is exclusively this job's business, and it is deliberate, capped and
cost-estimated first.

Backfilled episodes carry ``origin: "backfill"``, which keeps them out of the
regular pipeline queues and out of the weekly digest entirely.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx

from ..config import Settings
from ..db import Doc, Store, update_doc
from ..ingest.feeds import (
    _description,
    _duration_seconds,
    _entry_link,
    _feed_transcripts,
    _pick_enclosure,
    _published_at,
    _stable_guid,
    _title,
    _transcripts_from_raw_xml,
)
from ..logging_setup import get_logger
from ..net import FetchPolicy, UrlGuard, UrlRejected, get_guarded
from ..podcasts import PodcastRecord, PodcastRegistry
from ..sanitize import html_to_text
from ..state import BACKFILL_ORIGIN, EpisodeStatus
from ..utils import episode_doc_id, iso, iso_now, podcast_doc_id

log = get_logger(__name__)


@dataclass(slots=True)
class BackfillStats:
    feeds_examined: int = 0
    shows_complete: int = 0
    shows_skipped: int = 0
    months_processed: int = 0
    entries_seen: int = 0
    episodes_created: int = 0
    #: True when the run stopped because backfill was paused, not because it
    #: finished — so the caller can say so rather than implying completion.
    stopped_early: bool = False
    episodes_existing: int = 0
    without_transcript: int = 0
    skipped_unsupported: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feeds_examined": self.feeds_examined,
            "shows_complete": self.shows_complete,
            "shows_skipped": self.shows_skipped,
            "months_processed": self.months_processed,
            "entries_seen": self.entries_seen,
            "episodes_created": self.episodes_created,
            "stopped_early": self.stopped_early,
            "episodes_existing": self.episodes_existing,
            "without_transcript": self.without_transcript,
            "skipped_unsupported": self.skipped_unsupported,
            "error_count": len(self.errors),
        }


def month_key(when: datetime) -> str:
    return f"{when.year:04d}-{when.month:02d}"


def previous_month(key: str) -> str:
    year, month = (int(part) for part in key.split("-"))
    return f"{year - 1:04d}-12" if month == 1 else f"{year:04d}-{month - 1:02d}"


def month_bounds(key: str) -> tuple[datetime, datetime]:
    """[start, end) of a YYYY-MM window, in UTC."""
    year, month = (int(part) for part in key.split("-"))
    start = datetime(year, month, 1, tzinfo=UTC)
    end = (
        datetime(year + 1, 1, 1, tzinfo=UTC)
        if month == 12
        else datetime(year, month + 1, 1, tzinfo=UTC)
    )
    return start, end


def floor_month(now: datetime, months: int) -> str:
    """Oldest month the configured window reaches back to."""
    return floor_from_anchor(month_key(now), months)


def floor_from_anchor(anchor: str, months: int) -> str:
    """Oldest month a walk anchored at ``anchor`` reaches back to.

    The anchor is the month the walk started from, recorded once. Deriving the
    floor from it rather than from *today* is what stops the window drifting:
    computed fresh each run, the floor moves forward every calendar month while
    the cursor moves backward, and the two can cross. That is not hypothetical —
    on 2026-07-31 five shows sat at cursor 2025-07 with floor 2025-07, meaning
    "2025-07 is next". At midnight the floor became 2025-08, every one of them
    was declared finished, and 2025-07 was never fetched for any of them.

    Deriving it also keeps the window editable: raising a show from 12 months to
    24 deepens the floor, where a stored floor would have frozen it.
    """
    key = anchor
    for _ in range(months):
        key = previous_month(key)
    return key


class BackfillIngestor:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        client: httpx.AsyncClient,
        guard: UrlGuard,
        registry: PodcastRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._client = client
        self._guard = guard
        self._registry = registry or PodcastRegistry(settings)

    def eligible_podcasts(self) -> list[PodcastRecord]:
        return [p for p in self._registry.enabled_podcasts() if p.backfill_mode != "skip"]

    async def run(
        self,
        *,
        dry_run: bool = False,
        now: datetime | None = None,
        should_stop: Callable[[], Awaitable[bool]] | None = None,
    ) -> BackfillStats:
        now = now or datetime.now(UTC)
        stats = BackfillStats()
        budget = self._settings.backfill.max_episodes_per_run

        for podcast in self._registry.enabled_podcasts():
            if podcast.backfill_mode == "skip":
                stats.shows_skipped += 1
                continue
            if budget <= 0:
                break
            # Between shows is a clean boundary: each show's cursor is already
            # written, so stopping here loses nothing.
            if should_stop is not None and await should_stop():
                stats.stopped_early = True
                log.info("backfill.paused", after_show=podcast.slug)
                break
            try:
                used = await self._backfill_show(podcast, stats, now, budget, dry_run)
                budget -= used
            except Exception as exc:
                stats.errors.append(f"{podcast.slug}: {exc}")
                log.warning(
                    "backfill.show_failed", podcast=podcast.slug, error=str(exc), exc_info=True
                )

        log.info("backfill.run_complete", dry_run=dry_run, **stats.as_dict())
        return stats

    async def _backfill_show(
        self,
        podcast: PodcastRecord,
        stats: BackfillStats,
        now: datetime,
        budget: int,
        dry_run: bool,
    ) -> int:
        pdoc = await self._store.get(podcast_doc_id(podcast.slug)) or {}
        # Per podcast: an evergreen show can be worth three years of archive
        # while a daily news show is not worth one. Falls back to the configured
        # default, so "all of them, twelve months" needs no per-podcast setting.
        months = podcast.backfill_months or self._settings.backfill.months

        # The month this show's walk started from, recorded once. See
        # floor_from_anchor: recomputing the floor from `now` every run lets the
        # window drift forward past months the cursor has not reached yet.
        anchor = pdoc.get("backfill_anchor") or month_key(now)
        floor = floor_from_anchor(anchor, months)

        # Start the month before the current one: the current month is routine
        # ingestion's territory, not the archive's.
        cursor = pdoc.get("backfill_cursor") or previous_month(anchor)
        if cursor < floor:
            stats.shows_complete += 1
            # Recorded here as well as after a month is processed. It used to be
            # written only inside `if months_done:`, so a show that arrived
            # already past its floor was counted complete in the run summary
            # while its document kept `backfill_complete: False` — and the
            # console reads the document, so it said "in progress" forever.
            if not dry_run and pdoc and not pdoc.get("backfill_complete"):
                await self._mark_progress(podcast.slug, cursor, anchor, complete=True)
                log.info("backfill.show_complete", podcast=podcast.slug, floor=floor)
            return 0

        if not dry_run and pdoc and not pdoc.get("backfill_anchor"):
            # First run for this show: pin the month the walk is measured from.
            # Skipped when there is no podcast document yet — the anchor is
            # progress metadata, not the record, and it is re-derived next run.
            await self._mark_progress(podcast.slug, cursor, anchor, complete=False)

        response = await get_guarded(
            self._client,
            podcast.feed_url,
            policy=FetchPolicy(self._guard, related_to=podcast.feed_url, allowlist=False),
        )
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        transcript_map = _transcripts_from_raw_xml(response.content)
        stats.feeds_examined += 1

        created = 0
        months_done = 0
        for _ in range(self._settings.backfill.months_per_run):
            if cursor < floor or created >= budget:
                break
            created += await self._ingest_month(
                podcast, parsed, transcript_map, cursor, stats, dry_run, budget - created
            )
            months_done += 1
            stats.months_processed += 1
            cursor = previous_month(cursor)

        if not dry_run and months_done:
            complete = cursor < floor
            await self._mark_progress(podcast.slug, cursor, anchor, complete=complete)
            if complete:
                stats.shows_complete += 1
                log.info("backfill.show_complete", podcast=podcast.slug, floor=floor)

        return created

    async def _mark_progress(self, slug: str, cursor: str, anchor: str, *, complete: bool) -> None:
        """Record where this show's walk has got to, and what it is measured from."""

        def _apply(doc: Doc) -> None:
            doc["backfill_cursor"] = cursor
            doc["backfill_anchor"] = anchor
            doc["backfill_complete"] = complete
            doc["backfill_updated_at"] = iso_now()

        await update_doc(self._store, podcast_doc_id(slug), _apply)

    async def _ingest_month(
        self,
        podcast: PodcastRecord,
        parsed: Any,
        transcript_map: dict[str, list[dict[str, str]]],
        cursor: str,
        stats: BackfillStats,
        dry_run: bool,
        budget: int,
    ) -> int:
        start, end = month_bounds(cursor)
        created = 0

        for entry in parsed.entries:
            published = _published_at(entry)
            if published is None or not (start <= published < end):
                continue
            stats.entries_seen += 1
            if created >= budget:
                break

            enclosure_url, enclosure_type, enclosure_len = _pick_enclosure(entry)
            guid = _stable_guid(entry, enclosure_url)
            if not guid or not enclosure_url:
                stats.skipped_unsupported += 1
                continue

            transcripts = transcript_map.get(guid) or transcript_map.get(enclosure_url or "")
            if transcripts is None:
                transcripts = _feed_transcripts(entry)
            allowed = [t for t in transcripts if self._permitted(podcast, t["url"])]

            # Recorded either way. Knowing an episode exists costs a small
            # document; spending on it is a separate decision, made downstream by
            # the podcast's own transcription setting. Discarding it here
            # conflated the two, and
            # left a weekly podcast showing one episode with no way to tell that
            # its back catalogue had been silently dropped.
            if not allowed:
                stats.without_transcript += 1

            if self._permitted(podcast, enclosure_url) is False:
                stats.skipped_unsupported += 1
                continue

            doc_id = episode_doc_id(podcast.slug, guid)
            if await self._store.get(doc_id) is not None:
                stats.episodes_existing += 1
                continue
            if dry_run:
                created += 1
                stats.episodes_created += 1
                continue

            doc: Doc = {
                "_id": doc_id,
                "type": "episode",
                # Keeps archive material out of the regular queues and the
                # weekly digest (roadmap A1).
                "origin": BACKFILL_ORIGIN,
                "archive_month": cursor,
                # The podcast's archive mode and transcription toggle are
                # deliberately *not* copied here. They were, and a copy taken at
                # ingest silently outranked whatever the console said later.
                # BackfillProcessor reads them from the registry instead.
                "podcast_slug": podcast.slug,
                "podcast_name": podcast.name,
                "guid": guid,
                "title": html_to_text(_title(entry), max_chars=500) or "(untitled)",
                "link": _entry_link(entry),
                "description_raw": html_to_text(
                    _description(entry),
                    max_chars=self._settings.pipeline.description_max_chars,
                ),
                "published_at": iso(published),
                "enclosure_url": enclosure_url,
                "enclosure_type": enclosure_type,
                "enclosure_bytes": enclosure_len,
                "duration_s": _duration_seconds(entry),
                "feed_transcripts": allowed,
                "status": EpisodeStatus.NEW.value,
                "tier0": None,
                "tier1": None,
                "transcript_source": "none",
                "digest_id": None,
                "attempts": {"transcript": 0, "tier0": 0, "tier1": 0},
                "last_error": None,
                "created_at": iso_now(),
                "updated_at": iso_now(),
            }
            if await self._store.create(doc):
                created += 1
                stats.episodes_created += 1
            else:
                stats.episodes_existing += 1

        log.debug(
            "backfill.month_done",
            podcast=podcast.slug,
            month=cursor,
            created=created,
            dry_run=dry_run,
        )
        return created

    def _permitted(self, podcast: PodcastRecord, url: str | None) -> bool:
        if not url:
            return False
        try:
            self._guard.check(url, related_to=podcast.feed_url)
        except UrlRejected:
            return False
        return True
