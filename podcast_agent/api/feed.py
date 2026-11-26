"""The narrated digests as a private podcast feed.

A podcast player cannot sign in to the reverse proxy's login page, so the proxy
lets this one path through unauthenticated and the secret in the URL is the
whole gate. Nothing else belongs under this prefix: whatever is added here is
reachable without a login.

Built on request from the digest documents and the files beside them, so there is
no `feed.xml` to regenerate and none to go stale.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from ..config import Settings
from ..db import Doc, Store, typed_sort
from ..digest.read import DigestUnreadable, digest_period_key, digest_runs, resolve_within

#: The proxy's login exception names this prefix too — change them together.
PREFIX = "/podcast-feed"

_ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"
ET.register_namespace("itunes", _ITUNES)

#: Four years of weeks.
_MAX_DIGESTS = 210

_TITLE = "Podcast Digest"
_DESCRIPTION = "The weekly digest, read aloud."

router = APIRouter(prefix=PREFIX, include_in_schema=False)


@dataclass(frozen=True)
class Narration:
    period_key: str
    path: Path
    run: Doc


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
                found.append(Narration(digest_period_key(doc), path, run))
                break
    return found


def _published(narration: Narration) -> datetime:
    stamp = narration.run.get("generated_at")
    if stamp:
        return datetime.fromisoformat(str(stamp))
    return datetime.fromtimestamp(narration.path.stat().st_mtime).astimezone()


def render_feed(items: list[Narration], base_url: str) -> bytes:
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    def add(parent: ET.Element, tag: str, text: str) -> None:
        ET.SubElement(parent, tag).text = text

    add(channel, "title", _TITLE)
    add(channel, "link", base_url)
    add(channel, "description", _DESCRIPTION)
    add(channel, "language", "en")
    add(channel, f"{{{_ITUNES}}}author", _TITLE)
    add(channel, f"{{{_ITUNES}}}explicit", "false")
    # Asks directories not to list it, should the URL ever reach one.
    add(channel, f"{{{_ITUNES}}}block", "Yes")

    for narration in items:
        year, week = narration.period_key.split("-W")
        item = ET.SubElement(channel, "item")
        add(item, "title", f"Week {int(week)}, {year}")
        count = len(narration.run.get("episode_ids") or [])
        add(item, "description", f"{count} episodes from week {int(week)} of {year}.")
        add(item, "pubDate", format_datetime(_published(narration)))
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


# HEAD as well as GET: players ask about a file before they download it.
@router.api_route("/{token}/feed.xml", methods=["GET", "HEAD"])
async def feed(request: Request, token: str) -> Response:
    settings = _authorise(request, token)
    items = await narrations(request.app.state.store, settings.output.digest_dir)
    # Absolute URLs built from the address the player used to get here.
    base_url = f"{str(request.base_url).rstrip('/')}{PREFIX}/{token}"
    return Response(render_feed(items, base_url), media_type="application/rss+xml")


@router.api_route("/{token}/audio/{name}", methods=["GET", "HEAD"])
async def audio(request: Request, token: str, name: str) -> FileResponse:
    settings = _authorise(request, token)
    # Matched against the feed's own entries; `name` never becomes a path.
    for narration in await narrations(request.app.state.store, settings.output.digest_dir):
        if narration.path.name == name:
            # Handles Range requests, which is what seeking and resuming use.
            return FileResponse(narration.path, media_type="audio/mpeg")
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
