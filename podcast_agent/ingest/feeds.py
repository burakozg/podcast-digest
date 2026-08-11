"""Stage 1 — RSS ingestion (§4).

Idempotent and resumable: episode documents are keyed by
``sha256(podcast_slug + guid)`` and inserted with create-if-absent, so a crash
mid-run or two overlapping runs can never duplicate or lose an episode.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

# defusedxml, not xml.etree: feed XML is untrusted and stdlib ElementTree is
# vulnerable to entity-expansion (billion laughs) memory exhaustion (§10.2).
import defusedxml.ElementTree as DefusedET
import feedparser
import httpx

from ..cadence import cadence_from_dates
from ..config import Settings
from ..db import Doc, Store, typed_sort, update_doc
from ..logging_setup import get_logger
from ..net import FetchPolicy, UrlGuard, UrlRejected, get_guarded
from ..podcasts import PodcastRecord, PodcastRegistry
from ..sanitize import html_to_text
from ..state import ROUTINE_ORIGIN, EpisodeStatus
from ..utils import episode_doc_id, iso, iso_now, parse_iso, podcast_doc_id, utcnow

log = get_logger(__name__)

#: Podcasting 2.0 namespace. feedparser exposes prefixed keys, but namespace URI
#: spellings vary between publishers, so transcripts are read from the raw
#: element map rather than a single hard-coded prefix.
TRANSCRIPT_MIME_PREFERENCE = (
    "text/plain",
    "application/json",
    "text/vtt",
    "application/x-subrip",
    "application/srt",
    "text/html",
)

#: A feed that fails this many polls in a row is backed off to daily (§10.3).
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_BACKOFF = timedelta(hours=24)

#: Cap on episodes examined per feed per run — protects against a publisher
#: resetting all GUIDs and presenting 3000 "new" episodes at once.
MAX_ENTRIES_PER_FEED = 200


@dataclass(slots=True)
class IngestStats:
    feeds_polled: int = 0
    feeds_unchanged: int = 0
    feeds_failed: int = 0
    feeds_skipped_backoff: int = 0
    entries_seen: int = 0
    episodes_created: int = 0
    episodes_existing: int = 0
    entries_too_old: int = 0
    entries_unsupported: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feeds_polled": self.feeds_polled,
            "feeds_unchanged": self.feeds_unchanged,
            "feeds_failed": self.feeds_failed,
            "feeds_skipped_backoff": self.feeds_skipped_backoff,
            "entries_seen": self.entries_seen,
            "episodes_created": self.episodes_created,
            "episodes_existing": self.episodes_existing,
            "entries_too_old": self.entries_too_old,
            "entries_unsupported": self.entries_unsupported,
            "error_count": len(self.errors),
        }


class Ingestor:
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
        # The registry carries console overrides and console-added shows; falling
        # back to Settings keeps hand-built instances (tests) working.
        self._registry = registry or PodcastRegistry(settings)

    def use_registry(self, registry: PodcastRegistry) -> None:
        """Adopt the shared registry once it exists (seeding runs before it)."""
        self._registry = registry

    async def seed_podcast_docs(self) -> None:
        """Ensure a ``podcast`` doc exists per configured show.

        Config is the source of truth for feed URLs; the document only holds
        runtime state (etag, failure counters) (§6).

        Covers disabled shows too. The document is what console overrides attach
        to, so skipping disabled ones would make them impossible to re-enable.
        """
        for podcast in self._registry.all_podcasts():
            doc_id = podcast_doc_id(podcast.slug)
            existing = await self._store.get(doc_id)
            if existing is None:
                await self._store.create(
                    {
                        "_id": doc_id,
                        "type": "podcast",
                        "slug": podcast.slug,
                        "name": podcast.name,
                        "feed_url": podcast.feed_url,
                        "etag": None,
                        "last_modified": None,
                        "last_polled_at": None,
                        "last_error": None,
                        "consecutive_failures": 0,
                        "created_at": iso_now(),
                    }
                )
                log.info("ingest.podcast_seeded", podcast=podcast.slug)
                continue
            # Feed URL changed in config → reset conditional-GET state so the new
            # feed is read in full rather than 304'd against the old validators.
            if existing.get("feed_url") != podcast.feed_url:
                log.info(
                    "ingest.feed_url_changed",
                    podcast=podcast.slug,
                    old=existing.get("feed_url"),
                    new=podcast.feed_url,
                )

                def _apply(doc: Doc, p: PodcastRecord = podcast) -> None:
                    doc["feed_url"] = p.feed_url
                    doc["name"] = p.name
                    doc["etag"] = None
                    doc["last_modified"] = None
                    doc["consecutive_failures"] = 0
                    doc["last_error"] = None

                await update_doc(self._store, doc_id, _apply)

    async def run(self) -> IngestStats:
        stats = IngestStats()
        for podcast in self._registry.enabled_podcasts():
            try:
                await self._ingest_one(podcast, stats)
            except Exception as exc:  # one bad feed must not stop the rest
                stats.feeds_failed += 1
                stats.errors.append(f"{podcast.slug}: {exc}")
                log.warning(
                    "ingest.feed_failed", podcast=podcast.slug, error=str(exc), exc_info=True
                )
                await self._record_failure(podcast.slug, str(exc))
        log.info("ingest.run_complete", **stats.as_dict())
        return stats

    async def _ingest_one(self, podcast: PodcastRecord, stats: IngestStats) -> None:
        doc_id = podcast_doc_id(podcast.slug)
        pdoc = await self._store.get(doc_id) or {}

        if self._in_backoff(pdoc):
            stats.feeds_skipped_backoff += 1
            log.info(
                "ingest.feed_backoff_skip",
                podcast=podcast.slug,
                consecutive_failures=pdoc.get("consecutive_failures"),
            )
            return

        # Conditional GET saves bandwidth and is right once we have everything
        # we want from the feed. It is wrong before that: a 304 carries no body,
        # so a podcast that has never had its description and cadence captured
        # would wait for its next *episode* to get them — up to a month for a
        # monthly show, and forever for one that has stopped publishing.
        # Skipped once per podcast, then conditional again.
        needs_metadata = not pdoc.get("feed_metadata_at")
        # Same reasoning, different trigger: a 304 also means the entries are
        # never re-read, so widening the CDN allowlist changes nothing for any
        # feed that has not published since. Entries rejected under the old
        # rules stay rejected — which is how a podcast sat at zero episodes
        # across a fix that was already deployed and correct. One full fetch
        # per feed after the rules change, then conditional again.
        rules_changed = pdoc.get("intake_rules") != self._guard.fingerprint
        headers: dict[str, str] = {}
        if not needs_metadata and not rules_changed:
            if etag := pdoc.get("etag"):
                headers["If-None-Match"] = etag
            if last_modified := pdoc.get("last_modified"):
                headers["If-Modified-Since"] = last_modified
        elif rules_changed:
            log.info(
                "ingest.full_fetch_for_changed_rules",
                podcast=podcast.slug,
                detail="the fetch allowlist changed since this feed was last read in full",
            )
        else:
            log.info("ingest.full_fetch_for_metadata", podcast=podcast.slug)

        # The feed URL is owner-supplied, so the CDN allowlist does not apply to
        # it — but a feed that redirects still lands wherever it says, and that
        # target is chosen by whoever controls the feed's host. Every hop is
        # vetted; only the allowlist arm is relaxed.
        response = await get_guarded(
            self._client,
            podcast.feed_url,
            policy=FetchPolicy(self._guard, related_to=podcast.feed_url, allowlist=False),
            headers=headers,
        )
        stats.feeds_polled += 1

        if response.status_code == 304:
            stats.feeds_unchanged += 1
            await self._record_success(podcast.slug, response)
            log.debug("ingest.feed_unchanged", podcast=podcast.slug)
            return
        response.raise_for_status()

        parsed = feedparser.parse(response.content)
        if parsed.bozo and not parsed.entries:
            raise ValueError(f"unparseable feed: {getattr(parsed, 'bozo_exception', 'unknown')}")

        # feedparser collapses repeated <podcast:transcript> elements to the last
        # one, which silently discards fallback formats. Recover the full set from
        # the raw XML so acquisition can try text/plain before VTT before SRT.
        transcript_map = _transcripts_from_raw_xml(response.content)

        cutoff = await self._cutoff_for(podcast.slug)
        for entry in parsed.entries[:MAX_ENTRIES_PER_FEED]:
            stats.entries_seen += 1
            await self._ingest_entry(podcast, entry, cutoff, stats, transcript_map)

        await self._record_success(
            podcast.slug, response, metadata=_feed_metadata(parsed, transcript_map)
        )

    def _in_backoff(self, pdoc: Doc) -> bool:
        """Feed-level circuit breaker (§10.3)."""
        failures = int(pdoc.get("consecutive_failures") or 0)
        if failures < CIRCUIT_BREAKER_THRESHOLD:
            return False
        last_polled = pdoc.get("last_polled_at")
        if not last_polled:
            return False
        try:
            when = datetime.fromisoformat(last_polled)
        except ValueError:
            return False
        return utcnow() - when < CIRCUIT_BREAKER_BACKOFF

    async def _cutoff_for(self, slug: str) -> datetime | None:
        """Backfill guard (§10.4): routine polling only ever looks forward.

        For a fresh show the cutoff is ``initial_lookback_days``. For a show we
        already have history for it is the oldest episode we have already seen.

        That second case matters more than it looks. Leaving it uncapped meant
        that as soon as a show had one episode, every subsequent poll considered
        the whole feed page — up to MAX_ENTRIES_PER_FEED, which for a daily show
        is months of back catalogue, triaged silently at LLM cost. Reaching
        backwards is what the backfill job is for, and it is deliberate,
        rate-limited and cost-estimated; routine ingestion must not do it by
        accident.
        """
        lookback = self._settings.pipeline.initial_lookback_days
        # Oldest first; null published_at sorts first under CouchDB collation,
        # so take the first row that actually carries a date.
        existing = await self._store.find(
            {"type": "episode", "podcast_slug": slug},
            fields=["_id", "published_at"],
            sort=typed_sort("published_at", "asc"),
            limit=10,
        )
        if existing:
            dates = [parse_iso(d.get("published_at")) for d in existing]
            oldest = min((d for d in dates if d is not None), default=None)
            if oldest is not None:
                return oldest
            # History exists but is undated; fall back to the lookback window
            # rather than opening the whole feed.
        if lookback <= 0:
            return None
        return utcnow() - timedelta(days=lookback)

    async def _ingest_entry(
        self,
        podcast: PodcastRecord,
        entry: Any,
        cutoff: datetime | None,
        stats: IngestStats,
        transcript_map: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        enclosure_url, enclosure_type, enclosure_len = _pick_enclosure(entry)
        guid = _stable_guid(entry, enclosure_url)
        if not guid:
            stats.entries_unsupported += 1
            log.warning("ingest.entry_no_guid", podcast=podcast.slug, title=_title(entry)[:80])
            return
        if not enclosure_url:
            # v1 skips shows/entries with no audio enclosure (§2 non-goals).
            stats.entries_unsupported += 1
            log.info("ingest.entry_no_enclosure", podcast=podcast.slug, title=_title(entry)[:80])
            return

        published = _published_at(entry)
        if cutoff and published and published < cutoff:
            stats.entries_too_old += 1
            return

        doc_id = episode_doc_id(podcast.slug, guid)
        # Cheap existence probe keeps the hot path (nothing new) to one GET.
        if await self._store.get(doc_id) is not None:
            stats.episodes_existing += 1
            return

        safe_enclosure = self._checked(podcast, enclosure_url)
        if safe_enclosure is None:
            stats.entries_unsupported += 1
            return

        # Prefer the raw-XML set (all formats); fall back to feedparser's single one.
        transcripts = (transcript_map or {}).get(guid) or (transcript_map or {}).get(
            enclosure_url or ""
        )
        if transcripts is None:
            transcripts = _feed_transcripts(entry)
        allowed_transcripts = [
            t for t in transcripts if self._checked(podcast, t["url"]) is not None
        ]

        doc: Doc = {
            "_id": doc_id,
            "type": "episode",
            "podcast_slug": podcast.slug,
            "podcast_name": podcast.name,
            "guid": guid,
            "title": html_to_text(_title(entry), max_chars=500) or "(untitled)",
            "link": _entry_link(entry),
            "description_raw": html_to_text(
                _description(entry), max_chars=self._settings.pipeline.description_max_chars
            ),
            "published_at": iso(published) if published else None,
            "enclosure_url": safe_enclosure,
            "enclosure_type": enclosure_type,
            "enclosure_bytes": enclosure_len,
            "duration_s": _duration_seconds(entry),
            "feed_transcripts": allowed_transcripts,
            "status": EpisodeStatus.NEW.value,
            # Explicit, so selectors can match on equality. Mango cannot index
            # the absence of a field, and gets comparisons against a missing one
            # wrong — including negative ones.
            "origin": ROUTINE_ORIGIN,
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
            stats.episodes_created += 1
            log.info(
                "ingest.episode_created",
                episode_id=doc_id,
                podcast=podcast.slug,
                title=doc["title"][:100],
                published_at=doc["published_at"],
                feed_transcripts=len(allowed_transcripts),
            )
        else:
            # Lost a race with a concurrent run — the other writer created it.
            stats.episodes_existing += 1

    def _checked(self, podcast: PodcastRecord, url: str) -> str | None:
        try:
            return self._guard.check(url, related_to=podcast.feed_url)
        except UrlRejected as exc:
            log.warning("ingest.url_rejected", podcast=podcast.slug, error=str(exc))
            return None

    async def _record_success(
        self, slug: str, response: httpx.Response, *, metadata: dict[str, Any] | None = None
    ) -> None:
        def _apply(doc: Doc) -> None:
            doc["etag"] = response.headers.get("etag")
            doc["last_modified"] = response.headers.get("last-modified")
            doc["last_polled_at"] = iso_now()
            doc["last_error"] = None
            doc["consecutive_failures"] = 0
            # Which acceptance rules the stored validators are good for. Written
            # on every success, including a 304: the entries were read in full
            # under these rules at some point, and this poll confirms nothing
            # has changed since.
            doc["intake_rules"] = self._guard.fingerprint
            # A 304 carries no body, so there is nothing to re-read the feed's
            # own metadata from; the stored values are kept rather than blanked
            # on every unchanged poll.
            if metadata:
                doc.update(metadata)
                doc["feed_metadata_at"] = iso_now()

        await update_doc(self._store, podcast_doc_id(slug), _apply)

    async def _record_failure(self, slug: str, error: str) -> None:
        def _apply(doc: Doc) -> None:
            doc["last_polled_at"] = iso_now()
            doc["last_error"] = error[:500]
            doc["consecutive_failures"] = int(doc.get("consecutive_failures") or 0) + 1

        try:
            await update_doc(self._store, podcast_doc_id(slug), _apply)
        except Exception as exc:  # never mask the original feed error
            log.warning("ingest.failure_record_failed", podcast=slug, error=str(exc))


# --- feed entry parsing -----------------------------------------------------
# feedparser returns loosely-typed dicts; every accessor here tolerates missing
# and wrong-typed fields because feeds are untrusted input.


def _title(entry: Any) -> str:
    return str(entry.get("title") or "")


#: A console table cell, not an about page — long enough to say what the podcast
#: is, short enough that thirty rows stay scannable.
FEED_DESCRIPTION_CHARS = 240

#: Recent entries inspected when measuring cadence and transcript coverage.
#: Enough to be representative, cheap because the feed is already parsed.
PROBE_ENTRIES = 25


def _feed_metadata(parsed: Any, transcript_map: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """What the feed itself says about the podcast, captured at poll time.

    Cadence and transcript coverage are measured here rather than from the
    episodes we hold, because what we hold is bounded by the lookback window —
    a podcast added last week has two episodes and no measurable rhythm, while
    its feed plainly shows a weekly one.
    """
    entries = list(parsed.entries or [])[:PROBE_ENTRIES]
    dates = [d for d in (_published_at(e) for e in entries) if d is not None]
    cadence, cadence_detail = cadence_from_dates(dates)
    with_transcripts = sum(1 for e in entries if (e.get("id") or "") in transcript_map)
    return {
        "description": _feed_description(parsed.feed),
        "feed_cadence": cadence,
        "feed_cadence_detail": cadence_detail,
        # Measured, not declared: `has_feed_transcripts` in config.yaml is a
        # statement of intent, this is what the feed actually carries.
        "feed_entries_seen": len(entries),
        "feed_transcripts_seen": with_transcripts,
    }


def _feed_description(feed: Any) -> str:
    """The podcast's own description, as plain text.

    feedparser maps RSS ``channel/description`` and ``itunes:subtitle`` onto the
    same ``subtitle`` key, last one in the document winning, so there is no way
    from the mapped keys to prefer the tagline over the marketing copy — the
    keys below are tried in the order feedparser is most likely to have filled
    them, and whichever is non-empty is taken. Capping to a couple of lines
    makes the difference immaterial.

    This is untrusted feed content that ends up in the console, so it is reduced
    to text and capped here rather than trusted to be short or to be plain.
    """
    if not feed:
        return ""
    for key in ("subtitle", "summary", "content", "description"):
        value = feed.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, str) and value.strip():
            text = html_to_text(value, max_chars=FEED_DESCRIPTION_CHARS)
            if text:
                return text
    return ""


def _description(entry: Any) -> str:
    for key in ("content", "summary_detail", "summary", "subtitle", "description"):
        value = entry.get(key)
        if isinstance(value, list) and value:
            candidate = value[0].get("value") if isinstance(value[0], dict) else None
            if candidate:
                return str(candidate)
        elif isinstance(value, dict) and value.get("value"):
            return str(value["value"])
        elif isinstance(value, str) and value.strip():
            return value
    return ""


def _entry_link(entry: Any) -> str | None:
    link = entry.get("link")
    if isinstance(link, str) and link.startswith(("http://", "https://")):
        return link
    return None


def _stable_guid(entry: Any, enclosure_url: str | None) -> str | None:
    """GUID for the episode id, falling back to the enclosure URL (§4 stage 1)."""
    for key in ("id", "guid"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if enclosure_url:
        return enclosure_url
    # Last resort: a title+date pair. Weak, but better than dropping the episode.
    title = _title(entry).strip()
    published = entry.get("published") or entry.get("updated") or ""
    combined = f"{title}|{published}".strip("|")
    return combined or None


def _pick_enclosure(entry: Any) -> tuple[str | None, str | None, int | None]:
    """Best audio enclosure as (url, type, length_bytes)."""
    candidates = entry.get("enclosures") or []
    if not isinstance(candidates, list):
        return None, None, None
    audio = [
        e
        for e in candidates
        if isinstance(e, dict)
        and isinstance(e.get("href"), str)
        and str(e.get("type") or "").lower().startswith("audio")
    ]
    chosen = audio[0] if audio else None
    if chosen is None:
        # Some feeds omit or mistype the enclosure type; fall back to extension.
        chosen = next(
            (
                e
                for e in candidates
                if isinstance(e, dict)
                and isinstance(e.get("href"), str)
                and e["href"]
                .split("?")[0]
                .lower()
                .endswith((".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".mp4"))
            ),
            None,
        )
    if chosen is None:
        return None, None, None
    length = chosen.get("length")
    try:
        length_int = int(length) if length not in (None, "") else None
    except (TypeError, ValueError):
        length_int = None
    return chosen["href"], chosen.get("type"), length_int


def _rank_and_dedupe(found: list[dict[str, str]]) -> list[dict[str, str]]:
    """Order transcript candidates cheapest-to-normalise first, dropping dupes."""

    def rank(t: dict[str, str]) -> int:
        try:
            return TRANSCRIPT_MIME_PREFERENCE.index(t["type"])
        except ValueError:
            return len(TRANSCRIPT_MIME_PREFERENCE)

    found = sorted(found, key=rank)
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in found:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)
    return unique


def _feed_transcripts(entry: Any) -> list[dict[str, str]]:
    """Extract <podcast:transcript> from a feedparser entry (§4 stage 3).

    feedparser flattens namespaced elements to ``podcast_transcript`` and keeps
    only the last occurrence, so this is the fallback path — see
    :func:`_transcripts_from_raw_xml` for the complete set.
    """
    raw = entry.get("podcast_transcript")
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    found: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("href")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        found.append(
            {
                "url": url,
                "type": str(item.get("type") or "").lower(),
                "language": str(item.get("language") or ""),
            }
        )
    return _rank_and_dedupe(found)


#: Namespace URI spellings seen in the wild for the Podcasting 2.0 namespace.
_PODCAST_NS = (
    "https://podcastindex.org/namespace/1.0",
    "http://podcastindex.org/namespace/1.0",
    "https://podcastindex.org/namespace/1.0/",
)


def _transcripts_from_raw_xml(raw: bytes) -> dict[str, list[dict[str, str]]]:
    """Map guid (and enclosure URL) -> every <podcast:transcript> for that item.

    Parsed from the raw feed because feedparser discards all but the last
    transcript element, which would throw away usable fallback formats.
    Best-effort: a parse failure just means the feedparser value is used.
    """
    try:
        root = DefusedET.fromstring(raw)
    except Exception:
        # Includes ParseError and defusedxml's EntitiesForbidden / DTDForbidden.
        return {}

    mapping: dict[str, list[dict[str, str]]] = {}
    for item in root.iter("item"):
        found: list[dict[str, str]] = []
        for namespace in _PODCAST_NS:
            for node in item.findall(f"{{{namespace}}}transcript"):
                url = node.get("url")
                if not url or not url.startswith(("http://", "https://")):
                    continue
                found.append(
                    {
                        "url": url,
                        "type": (node.get("type") or "").strip().lower(),
                        "language": (node.get("language") or "").strip(),
                    }
                )
        if not found:
            continue
        ranked = _rank_and_dedupe(found)
        keys = [(item.findtext("guid") or "").strip()]
        enclosure = item.find("enclosure")
        if enclosure is not None:
            keys.append((enclosure.get("url") or "").strip())
        for key in keys:
            if key:
                mapping[key] = ranked
    return mapping


def _published_at(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC)
            except (TypeError, ValueError, OverflowError):
                continue
    return None


def _duration_seconds(entry: Any) -> int | None:
    """Parse itunes:duration, which appears as seconds, MM:SS or HH:MM:SS."""
    raw = entry.get("itunes_duration")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if ":" not in text:
            return max(0, int(float(text)))
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if len(parts) > 3:
        return None
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return max(0, int(total))
