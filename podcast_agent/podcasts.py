"""The live podcast list: config.yaml as seed, database as the managed layer.

§6 made config.yaml the source of truth for shows, which was right when the only
way to change one was to edit the file. Managing shows from the console needs
somewhere writable, and config.yaml is mounted read-only in the container — so
the database holds *overrides* and *additions*, and config remains the declared
baseline for everything nobody has touched.

The consequence worth knowing: a field edited in the console stops tracking
config.yaml for that show. Every record reports which fields are overridden, so
"why is this not what the file says?" is always answerable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config import PodcastConfig, Priority, Settings
from .db import Doc, Store, update_doc
from .logging_setup import get_logger
from .utils import iso_now, podcast_doc_id

log = get_logger(__name__)

#: Fields the console may override. Everything else stays config-only, because
#: it either cannot be expressed in a form (transcript_selector) or would let a
#: typo in a browser break ingestion for good (slug).
OVERRIDABLE = frozenset(
    {
        "name",
        "feed_url",
        "enabled",
        "priority",
        "always_escalate",
        "asr_enabled",
        "backfill_mode",
        "backfill_months",
    }
)

PodcastSource = Literal["config", "console"]


@dataclass(frozen=True, slots=True)
class PodcastRecord:
    """One show, after config and database have been merged."""

    slug: str
    name: str
    feed_url: str
    priority: Priority
    always_escalate: bool
    has_feed_transcripts: bool
    transcript_selector: str | None
    #: Substring rewrite applied to an episode link before scraping, or None.
    transcript_url_sub: tuple[str, str] | None
    enabled: bool
    backfill_mode: str
    #: Archive window for this show, or None to inherit backfill.months.
    backfill_months: int | None
    #: Whether the routine pipeline may transcribe this show's audio locally.
    asr_enabled: bool
    source: PodcastSource
    #: Field names whose value came from the console rather than config.yaml.
    overridden: frozenset[str] = frozenset()
    #: Runtime state, read-only here.
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def in_config(self) -> bool:
        return self.source == "config"


class PodcastRegistry:
    """Mutable, refreshable view of the show list.

    Held by the stages and refreshed at the start of each run, so a change made
    in the console takes effect on the next run rather than the next restart.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records: list[PodcastRecord] = [_from_config(p, {}) for p in settings.podcasts]

    async def refresh(self, store: Store) -> None:
        docs = {
            d["slug"]: d for d in await store.find({"type": "podcast"}, limit=1000) if d.get("slug")
        }
        records: list[PodcastRecord] = []
        seen: set[str] = set()

        for podcast in self._settings.podcasts:
            seen.add(podcast.slug)
            records.append(_from_config(podcast, docs.get(podcast.slug, {})))

        # Shows added in the console have no config entry at all.
        for slug, doc in docs.items():
            if slug in seen or doc.get("source") != "console":
                continue
            record = _from_doc(doc)
            if record is not None:
                records.append(record)

        self._records = records

    # --- read API (mirrors Settings so call sites barely change) ------------

    def all_podcasts(self) -> list[PodcastRecord]:
        return list(self._records)

    def enabled_podcasts(self) -> list[PodcastRecord]:
        return [r for r in self._records if r.enabled]

    def podcast_by_slug(self, slug: str) -> PodcastRecord | None:
        return next((r for r in self._records if r.slug == slug), None)

    def allows_asr(self, slug: str) -> bool:
        """Whether the routine pipeline may transcribe this show locally."""
        record = self.podcast_by_slug(slug)
        return bool(record and record.asr_enabled)


def _from_config(podcast: PodcastConfig, doc: Doc) -> PodcastRecord:
    overrides = {k: v for k, v in (doc.get("overrides") or {}).items() if k in OVERRIDABLE}

    def value(name: str) -> Any:
        return overrides.get(name, getattr(podcast, name))

    return PodcastRecord(
        slug=podcast.slug,
        name=value("name"),
        feed_url=value("feed_url"),
        priority=Priority(value("priority")),
        always_escalate=bool(value("always_escalate")),
        has_feed_transcripts=podcast.has_feed_transcripts,
        transcript_selector=podcast.transcript_selector,
        transcript_url_sub=podcast.transcript_url_sub,
        enabled=bool(value("enabled")),
        backfill_mode=str(value("backfill_mode")),
        backfill_months=_opt_int(value("backfill_months")),
        asr_enabled=bool(value("asr_enabled")),
        source="config",
        overridden=frozenset(overrides),
        state=_state(doc),
    )


def _opt_int(value: Any) -> int | None:
    """None stays None — it means "inherit", which is not the same as zero."""
    return int(value) if value else None


def _from_doc(doc: Doc) -> PodcastRecord | None:
    overrides = doc.get("overrides") or {}
    slug, feed_url = doc.get("slug"), overrides.get("feed_url") or doc.get("feed_url")
    if not slug or not feed_url:
        log.warning("podcasts.console_entry_incomplete", slug=slug)
        return None
    return PodcastRecord(
        slug=str(slug),
        name=str(overrides.get("name") or doc.get("name") or slug),
        feed_url=str(feed_url),
        priority=Priority(overrides.get("priority", "med")),
        always_escalate=bool(overrides.get("always_escalate", False)),
        has_feed_transcripts=bool(doc.get("has_feed_transcripts", False)),
        transcript_selector=None,
        transcript_url_sub=None,
        enabled=bool(overrides.get("enabled", True)),
        backfill_mode=str(overrides.get("backfill_mode", "full")),
        backfill_months=_opt_int(overrides.get("backfill_months")),
        asr_enabled=bool(overrides.get("asr_enabled", False)),
        source="console",
        overridden=frozenset(overrides),
        state=_state(doc),
    )


def _state(doc: Doc) -> dict[str, Any]:
    return {
        "last_polled_at": doc.get("last_polled_at"),
        "consecutive_failures": int(doc.get("consecutive_failures") or 0),
        "last_error": doc.get("last_error"),
        "backfill_cursor": doc.get("backfill_cursor"),
        "backfill_complete": bool(doc.get("backfill_complete")),
    }


# --- writes -----------------------------------------------------------------


async def set_overrides(store: Store, slug: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Apply console overrides to one show. Unknown fields are refused."""
    rejected = set(changes) - OVERRIDABLE
    if rejected:
        raise ValueError(f"not overridable: {', '.join(sorted(rejected))}")

    doc_id = podcast_doc_id(slug)
    if await store.get(doc_id) is None:
        raise KeyError(slug)

    def _apply(doc: Doc) -> None:
        overrides = dict(doc.get("overrides") or {})
        overrides.update(changes)
        doc["overrides"] = overrides
        doc["overrides_updated_at"] = iso_now()

    await update_doc(store, doc_id, _apply)
    log.info("podcasts.overrides_set", podcast=slug, changes=sorted(changes))
    return dict(changes)


async def clear_override(store: Store, slug: str, name: str) -> None:
    """Return one field to whatever config.yaml says."""

    def _apply(doc: Doc) -> None:
        overrides = dict(doc.get("overrides") or {})
        overrides.pop(name, None)
        doc["overrides"] = overrides
        doc["overrides_updated_at"] = iso_now()

    await update_doc(store, podcast_doc_id(slug), _apply)
    log.info("podcasts.override_cleared", podcast=slug, field=name)


async def add_console_podcast(
    store: Store, *, slug: str, name: str, feed_url: str, **overrides: Any
) -> Doc:
    """Register a show that exists only in the database.

    There is no counterpart that removes one. Shows are disabled, never deleted:
    a deleted show would take its provenance with it, leaving episodes in the
    database whose origin could not be explained.
    """
    doc_id = podcast_doc_id(slug)
    if await store.get(doc_id) is not None:
        raise ValueError(f"a podcast with slug {slug!r} already exists")

    doc: Doc = {
        "_id": doc_id,
        "type": "podcast",
        "slug": slug,
        "name": name,
        "feed_url": feed_url,
        "source": "console",
        "overrides": {"name": name, "feed_url": feed_url, "enabled": True, **overrides},
        "etag": None,
        "last_modified": None,
        "last_polled_at": None,
        "last_error": None,
        "consecutive_failures": 0,
        "created_at": iso_now(),
    }
    await store.create(doc)
    log.info("podcasts.added", podcast=slug, feed_url=feed_url)
    return doc
