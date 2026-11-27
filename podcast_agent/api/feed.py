"""The narrated digests as a private podcast feed.

A podcast player cannot sign in to the reverse proxy's login page, so the proxy
lets this one path through unauthenticated and the secret in the URL is the
whole gate. Nothing else belongs under this prefix: whatever is added here is
reachable without a login.

Built on request from the digest documents and the files beside them, so there is
no `feed.xml` to regenerate and none to go stale.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from ..config import Settings
from ..db import Doc, Store, typed_sort
from ..digest import cover as artwork
from ..digest.generate import BASIS_LABELS, summary_view
from ..digest.mp3 import estimated_file_duration_s
from ..digest.read import DigestUnreadable, digest_period_key, digest_runs, resolve_within
from ..digest.show_notes import episode_title, lead_theme, show_notes

#: The proxy's login exception names this prefix too — change them together.
PREFIX = "/podcast-feed"

_ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"
_CONTENT = "http://purl.org/rss/1.0/modules/content/"
ET.register_namespace("itunes", _ITUNES)
ET.register_namespace("content", _CONTENT)

#: Players cache artwork by its address: rename the file when the picture changes.
_COVER = Path(__file__).parent / "static" / "podcast-cover.jpg"

#: Episode documents read at once while building show notes.
_READS_AT_ONCE = 8

#: Four years of weeks.
_MAX_DIGESTS = 210

_TITLE = "Podcast Digest"
_DESCRIPTION = "The weekly digest, read aloud."

router = APIRouter(prefix=PREFIX, include_in_schema=False)

#: For the console, so under `/api/` and behind the login: never under PREFIX.
console_router = APIRouter(prefix="/api/v1")


@dataclass(frozen=True)
class Narration:
    period_key: str
    path: Path
    run: Doc
    digest: Doc

    @property
    def recorded(self) -> dict[str, Any]:
        """What narration noted about *this* file, if anything."""
        for holder in (self.run, self.digest):
            found = holder.get("narration") or {}
            if Path(str(found.get("audio_path") or "")).name == self.path.name:
                return dict(found)
        return {}

    @property
    def theme(self) -> str:
        return lead_theme(self.digest.get("synthesis"))

    @property
    def cover_name(self) -> str:
        """Carries a version, so a changed picture is fetched again."""
        return f"{self.path.stem}-{artwork.version(self.period_key, self.theme)}.jpg"


@dataclass(frozen=True)
class Entry:
    narration: Narration
    title: str
    notes: str
    duration_s: float | None


def _authorise(request: Request, token: str) -> Settings:
    """404 for every refusal, so a wrong guess cannot tell a feed is there."""
    settings: Settings = request.app.state.settings
    secret = settings.feed_token
    if secret is None or not secrets.compare_digest(
        token.encode(), secret.get_secret_value().encode()
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return settings


async def narrations(store: Store, digest_dir: Path) -> list[Narration]:
    """One entry per week that has audio on disk, newest week first.

    The newest run *with audio* wins: a week just regenerated keeps serving its
    previous reading until the new one has been narrated. MP3 only, because that
    is what podcast players can be relied on to play.
    """
    docs = await store.find(
        {"type": "digest"}, sort=typed_sort("generated_at", "desc"), limit=_MAX_DIGESTS
    )
    found: list[Narration] = []
    for doc in sorted(docs, key=digest_period_key, reverse=True):
        for run in reversed(digest_runs(doc)):
            relative = run.get("file_path")
            if not relative:
                continue
            try:
                path = resolve_within(digest_dir, str(Path(relative).with_suffix(".mp3")))
            except DigestUnreadable:
                continue
            if path.is_file():
                found.append(Narration(digest_period_key(doc), path, run, doc))
                break
    return found


def _published(narration: Narration) -> datetime:
    stamp = narration.run.get("generated_at")
    if stamp:
        return datetime.fromisoformat(str(stamp))
    return datetime.fromtimestamp(narration.path.stat().st_mtime).astimezone()


async def describe(settings: Settings, store: Store, items: list[Narration]) -> list[Entry]:
    """Title, notes and length for each narrated week."""
    gate = asyncio.Semaphore(_READS_AT_ONCE)

    async def read(episode_id: str) -> Doc | None:
        async with gate:
            return await store.get(episode_id)

    entries: list[Entry] = []
    for narration in items:
        ids = [str(i) for i in narration.run.get("episode_ids") or []]
        docs = await asyncio.gather(*(read(i) for i in ids))
        views = [
            summary_view(settings, doc, BASIS_LABELS)
            for doc in docs
            if doc is not None and (doc.get("tier1") or {}).get("summary_md")
        ]
        threshold = settings.pipeline.top_pick_threshold
        picks = sorted(
            (v for v in views if v["score"] >= threshold),
            key=lambda v: (-v["score"], v["podcast_name"]),
        )
        synthesis = narration.digest.get("synthesis")
        recorded = narration.recorded
        entries.append(
            Entry(
                narration=narration,
                title=episode_title(narration.period_key, synthesis),
                notes=show_notes(
                    narration.period_key,
                    synthesis,
                    recorded.get("chapters") or [],
                    picks,
                    len(ids),
                ),
                duration_s=recorded.get("duration_s") or estimated_file_duration_s(narration.path),
            )
        )
    return entries


def render_feed(entries: list[Entry], base_url: str) -> bytes:
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    def add(parent: ET.Element, tag: str, text: str) -> None:
        ET.SubElement(parent, tag).text = text

    cover = f"{base_url}/{_COVER.name}"
    add(channel, "title", _TITLE)
    add(channel, "link", base_url)
    add(channel, "description", _DESCRIPTION)
    add(channel, "language", "en")
    image = ET.SubElement(channel, "image")
    add(image, "url", cover)
    add(image, "title", _TITLE)
    add(image, "link", base_url)
    ET.SubElement(channel, f"{{{_ITUNES}}}image", href=cover)
    add(channel, f"{{{_ITUNES}}}author", _TITLE)
    add(channel, f"{{{_ITUNES}}}explicit", "false")
    # Asks directories not to list it, should the URL ever reach one.
    add(channel, f"{{{_ITUNES}}}block", "Yes")

    for entry in entries:
        narration = entry.narration
        item = ET.SubElement(channel, "item")
        add(item, "title", entry.title)
        add(item, "description", entry.notes)
        add(item, f"{{{_CONTENT}}}encoded", entry.notes)
        add(item, "pubDate", format_datetime(_published(narration)))
        ET.SubElement(item, f"{{{_ITUNES}}}image", href=f"{base_url}/cover/{narration.cover_name}")
        if entry.duration_s:
            add(item, f"{{{_ITUNES}}}duration", str(round(entry.duration_s)))
        # The file's name, not the week: a regenerated week lands as `-r2` and a
        # player has to see that as new audio rather than one it already holds.
        guid = ET.SubElement(item, "guid", isPermaLink="false")
        guid.text = narration.path.stem
        ET.SubElement(
            item,
            "enclosure",
            url=f"{base_url}/audio/{narration.path.name}",
            length=str(narration.path.stat().st_size),
            type="audio/mpeg",
        )
    return bytes(ET.tostring(rss, encoding="utf-8", xml_declaration=True))


@console_router.get("/feed", summary="Where to follow the private podcast feed")
async def feed_address(request: Request, response: Response) -> dict[str, str | None]:
    """The one response that carries a secret, so that it can be found again.

    ``path`` rather than a whole URL: the console is already at the address the
    player should use. None while the feed is off.
    """
    secret = request.app.state.settings.feed_token
    response.headers["Cache-Control"] = "no-store"
    if secret is None:
        return {"path": None}
    return {"path": f"{PREFIX}/{secret.get_secret_value()}/feed.xml"}


# HEAD as well as GET: players ask about a file before they download it.
@router.api_route("/{token}/feed.xml", methods=["GET", "HEAD"])
async def feed(request: Request, token: str) -> Response:
    settings = _authorise(request, token)
    store: Store = request.app.state.store
    entries = await describe(settings, store, await narrations(store, settings.output.digest_dir))
    # Absolute URLs built from the address the player used to get here.
    base_url = f"{str(request.base_url).rstrip('/')}{PREFIX}/{token}"
    return Response(render_feed(entries, base_url), media_type="application/rss+xml")


@router.api_route(f"/{{token}}/{_COVER.name}", methods=["GET", "HEAD"])
async def cover(request: Request, token: str) -> FileResponse:
    _authorise(request, token)
    return FileResponse(_COVER, media_type="image/jpeg")


@router.api_route("/{token}/cover/{name}", methods=["GET", "HEAD"])
async def episode_cover(request: Request, token: str, name: str) -> Response:
    settings = _authorise(request, token)
    for narration in await narrations(request.app.state.store, settings.output.digest_dir):
        if narration.cover_name == name:
            # Off the event loop: drawing takes long enough to stall other requests.
            image = await asyncio.to_thread(artwork.render, narration.period_key, narration.theme)
            return Response(image, media_type="image/jpeg")
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.api_route("/{token}/audio/{name}", methods=["GET", "HEAD"])
async def audio(request: Request, token: str, name: str) -> FileResponse:
    settings = _authorise(request, token)
    # Matched against the feed's own entries; `name` never becomes a path.
    for narration in await narrations(request.app.state.store, settings.output.digest_dir):
        if narration.path.name == name:
            # Handles Range requests, which is what seeking and resuming use.
            return FileResponse(narration.path, media_type="audio/mpeg")
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
