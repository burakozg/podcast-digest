"""Podcast management endpoints for the console.

config.yaml stays the declared baseline; these write overrides and additions to
the database, because config.yaml is mounted read-only in the container and a
console that cannot persist anything is not a console.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import feedparser
import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..cadence import cadence_from_dates, describe_cadence
from ..db import Store
from ..ingest.feeds import (
    _feed_description,
    _pick_enclosure,
    _published_at,
    _transcripts_from_raw_xml,
)
from ..logging_setup import get_logger
from ..net import UrlGuard, UrlRejected, build_client
from ..podcasts import (
    OVERRIDABLE,
    PodcastRegistry,
    add_console_podcast,
    clear_override,
    set_overrides,
)
from ..sanitize import slugify
from ..state import ACTIVE_STATUSES
from ..utils import podcast_doc_id
from .auth import require_api_key

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/podcasts", dependencies=[Depends(require_api_key)])

#: Entries inspected when probing a feed. Enough to judge transcripts, cheap.
PROBE_ENTRIES = 25


class PodcastOverrides(BaseModel):
    """Fields the console may change. Absent means "leave as it is"."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    feed_url: str | None = None
    enabled: bool | None = None
    priority: Literal["high", "med", "low"] | None = None
    always_escalate: bool | None = None
    asr_enabled: bool | None = None
    backfill_mode: Literal["full", "tier0_only", "skip"] | None = None
    #: Archive window for this podcast. `None` in a PATCH means "unchanged"; to
    #: go back to inheriting the default, revert the override.
    backfill_months: Literal[12, 24, 36] | None = None

    @field_validator("feed_url")
    @classmethod
    def _http(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("feed_url must be an http(s) URL")
        return v

    def changes(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class NewPodcast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64)
    name: str = Field(min_length=1, max_length=200)
    feed_url: str
    priority: Literal["high", "med", "low"] = "med"
    always_escalate: bool = False
    #: Off by default. ASR is the expensive path and should be a decision.
    asr_enabled: bool = False
    backfill_mode: Literal["full", "tier0_only", "skip"] = "full"
    #: Inherits backfill.months unless set.
    backfill_months: Literal[12, 24, 36] | None = None

    @field_validator("feed_url")
    @classmethod
    def _http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("feed_url must be an http(s) URL")
        return v


def _registry(request: Request) -> PodcastRegistry:
    """The live registry the pipeline stages hold, so a write is seen at once."""
    registry: PodcastRegistry = request.app.state.registry
    return registry


def _store(request: Request) -> Store:
    return request.app.state.store  # type: ignore[no-any-return]


async def _episode_stats(store: Store) -> dict[str, dict[str, Any]]:
    """Episode counts and publication cadence per podcast, in one scan.

    Disabling a podcast stops *intake*; episodes already ingested keep moving
    through the pipeline. Reporting what is still queued makes that visible
    rather than surprising.

    Cadence is measured from the same rows, so it costs nothing beyond pulling
    one more field — and it describes the episodes actually held, which is the
    honest thing for the console to claim.
    """
    docs = await store.find(
        {"type": "episode"},
        fields=["podcast_slug", "status", "published_at"],
        limit=10_000,
    )
    stats: dict[str, dict[str, Any]] = {}
    active = {s.value for s in ACTIVE_STATUSES}
    for doc in docs:
        slug = doc.get("podcast_slug")
        if not slug:
            continue
        entry = stats.setdefault(slug, {"total": 0, "queued": 0, "published": []})
        entry["total"] += 1
        if doc.get("status") in active:
            entry["queued"] += 1
        entry["published"].append(doc.get("published_at"))

    for entry in stats.values():
        label, detail = describe_cadence(entry.pop("published"))
        entry["cadence"] = label
        entry["cadence_detail"] = detail
    return stats


async def _feed_metadata(store: Store) -> dict[str, dict[str, Any]]:
    """What each feed says about itself, captured by the last successful poll.

    Held on the podcast document rather than in config.yaml: these are the
    podcast's own facts, and they should follow the feed when it changes without
    anyone editing a file.
    """
    docs = await store.find(
        {"type": "podcast"},
        fields=[
            "slug",
            "description",
            "feed_cadence",
            "feed_cadence_detail",
            "feed_entries_seen",
            "feed_transcripts_seen",
            "feed_metadata_at",
        ],
        limit=1_000,
    )
    return {str(doc["slug"]): doc for doc in docs if doc.get("slug")}


def _feed_facts(meta: dict[str, Any], held: dict[str, Any]) -> dict[str, Any]:
    """Merge feed-measured facts with what can be derived from held episodes.

    The feed wins where it has an answer. What we hold is bounded by the lookback
    window, so a podcast added last week has two episodes and no measurable
    rhythm while its feed plainly shows a weekly one.
    """
    seen = int(meta.get("feed_entries_seen") or 0)
    with_transcripts = int(meta.get("feed_transcripts_seen") or 0)
    return {
        "description": str(meta.get("description") or ""),
        "cadence": meta.get("feed_cadence") or held.get("cadence"),
        "cadence_detail": meta.get("feed_cadence_detail") or held.get("cadence_detail"),
        "cadence_source": "feed" if meta.get("feed_cadence") else "episodes held",
        # None until polled — distinct from a measured zero, which means the feed
        # was read and carries no transcripts at all.
        "transcripts_seen": with_transcripts if seen else None,
        "transcripts_of": seen or None,
        "polled_for_metadata": bool(meta.get("feed_metadata_at")),
    }


@router.get("", summary="Every configured podcast, merged with console overrides")
async def list_podcasts(request: Request) -> dict[str, Any]:
    registry = _registry(request)
    settings = request.app.state.settings
    store = _store(request)
    await registry.refresh(store)
    stats = await _episode_stats(store)
    feed_meta = await _feed_metadata(store)

    return {
        "count": len(registry.all_podcasts()),
        "overridable": sorted(OVERRIDABLE),
        "podcasts": [
            {
                "slug": r.slug,
                "name": r.name,
                "feed_url": r.feed_url,
                "priority": r.priority.value,
                "enabled": r.enabled,
                "always_escalate": r.always_escalate,
                "asr_enabled": r.asr_enabled,
                "backfill_mode": r.backfill_mode,
                # The effective window, plus whether it was chosen or inherited.
                "backfill_months": r.backfill_months or settings.backfill.months,
                "backfill_months_overridden": r.backfill_months is not None,
                "has_feed_transcripts": r.has_feed_transcripts,
                # Everything below comes from the feed itself at poll time, so
                # it is absent until the podcast has been polled once.
                **_feed_facts(feed_meta.get(r.slug) or {}, stats.get(r.slug) or {}),
                # Where each value came from, so a surprise is explainable.
                "source": r.source,
                "overridden": sorted(r.overridden),
                # Nothing is ever removed; a podcast is disabled and keeps its
                # history. Episodes are never deleted at all.
                "episodes": stats.get(r.slug, {}).get("total", 0),
                # Whether "summarise" is actually reachable for this podcast.
                # The archive summarises from a transcript; with none published
                # and local transcription off there is nothing to summarise from,
                # so it is indexed and scored whatever its archive mode says.
                "archive_indexes_only": bool(
                    r.backfill_mode != "skip"
                    and not r.asr_enabled
                    and (feed_meta.get(r.slug) or {}).get("feed_metadata_at")
                    and not (feed_meta.get(r.slug) or {}).get("feed_transcripts_seen")
                ),
                "queued": stats.get(r.slug, {}).get("queued", 0),
                **r.state,
            }
            for r in sorted(registry.all_podcasts(), key=lambda r: r.slug)
        ],
    }


@router.post("/probe", summary="Check a feed URL before adding it")
async def probe_feed(
    feed_url: str = Query(description="RSS feed URL to inspect"),
) -> dict[str, Any]:
    """Fetch a candidate feed and report what the pipeline would find.

    Adding a show that turns out to be a homepage, or has no audio enclosures,
    otherwise fails silently at the next ingest — this makes it visible first.
    """
    if not feed_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="feed_url must be http(s)"
        )
    try:
        async with build_client(timeout=25.0) as client:
            response = await client.get(feed_url)
            response.raise_for_status()
            raw = response.content
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"could not fetch: {exc}"
        ) from exc

    parsed = feedparser.parse(raw)
    entries = parsed.entries[:PROBE_ENTRIES]
    with_audio = sum(1 for e in entries if _pick_enclosure(e)[0])
    transcripts = _transcripts_from_raw_xml(raw)

    title = str((parsed.feed or {}).get("title") or "").strip()
    if not entries:
        detail = "parsed, but it contains no items — is this really a podcast feed?"
    elif not with_audio:
        detail = "items found but none carry an audio enclosure; v1 skips such shows"
    else:
        detail = "looks like a usable podcast feed"

    return {
        "ok": bool(entries and with_audio),
        "title": title,
        "description": _feed_description(parsed.feed),
        # Measured from the feed itself, so the rhythm is visible before
        # committing to poll it.
        "cadence": cadence_from_dates(
            [d for d in (_published_at(e) for e in entries) if d is not None]
        )[0],
        "suggested_slug": _slugify(title) if title else "",
        "entries_seen": len(entries),
        "with_audio": with_audio,
        "with_transcripts": sum(1 for e in entries if (e.get("id") or "") in transcripts),
        "latest": str(entries[0].get("published") or "") if entries else "",
        "detail": detail,
    }


def _slugify(title: str) -> str:
    return slugify(title, max_len=48)


@router.post("", status_code=status.HTTP_201_CREATED, summary="Add a show")
async def create_podcast(request: Request, body: NewPodcast) -> dict[str, Any]:
    store = _store(request)
    settings = request.app.state.settings

    if any(p.slug == body.slug for p in settings.podcasts):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{body.slug!r} is defined in config.yaml; edit it there or override it",
        )
    # Exactly the guard ingestion applies to a feed URL: scheme only. The
    # domain allowlist governs where a feed's audio and transcripts may come
    # from, not where the feed itself lives, so a self-hosted LAN feed is fine.
    try:
        UrlGuard(settings.security).check(body.feed_url, related_to=body.feed_url)
    except UrlRejected as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        await add_console_podcast(
            store,
            slug=body.slug,
            name=body.name,
            feed_url=body.feed_url,
            priority=body.priority,
            always_escalate=body.always_escalate,
            asr_enabled=body.asr_enabled,
            backfill_mode=body.backfill_mode,
            # Omitted when unset: storing an explicit null would mark the field
            # as overridden when it is in fact inheriting the default.
            **({"backfill_months": body.backfill_months} if body.backfill_months else {}),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    await _registry(request).refresh(store)
    return {"slug": body.slug, "created": True}


@router.patch("/{slug}", summary="Override settings for one show")
async def update_podcast(
    request: Request,
    slug: str,
    body: Annotated[PodcastOverrides, Body()],
) -> dict[str, Any]:
    changes = body.changes()
    if not changes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no fields to change")
    try:
        await set_overrides(_store(request), slug, changes)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no such podcast: {slug}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    await _registry(request).refresh(_store(request))
    return {"slug": slug, "changed": changes}


@router.delete("/{slug}/overrides/{fieldname}", summary="Revert one field to the config.yaml value")
async def reset_override(request: Request, slug: str, fieldname: str) -> dict[str, Any]:
    if fieldname not in OVERRIDABLE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"not overridable: {fieldname}"
        )
    if await _store(request).get(podcast_doc_id(slug)) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such podcast")
    await clear_override(_store(request), slug, fieldname)
    await _registry(request).refresh(_store(request))
    return {"slug": slug, "reverted": fieldname}
