#!/usr/bin/env python
"""Walk each podcast's audio redirect chain and check every hop against the guard.

Audio is rarely served from the URL a feed publishes. It is reached through one
or more analytics prefixers, and *every* host in that chain has to satisfy
``security.cdn_allowlist`` — not just the first, since the fetcher re-checks each
hop before following it.

The failure this exists to prevent is silent: a rejected enclosure discards the
episode, so a missing allowlist entry looks like a podcast that has stopped
publishing. Two of these were found only because someone opened an episode and
read the error, by which point the show had been quietly empty for a while.

Run it after changing the allowlist, after adding a podcast, and after any change
to how redirects are followed:

    PODAGENT_CONFIG_FILE=config.local.yaml uv run python scripts/check-enclosure-chains.py

Needs the network and a populated database; it is deliberately not a test.
Exits non-zero if any chain would be refused.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from podcast_agent.config import load_settings
from podcast_agent.net import MAX_REDIRECTS, USER_AGENT, UrlGuard, UrlRejected

#: One byte. Enough to learn where a redirect points without downloading audio.
PEEK = {"Range": "bytes=0-0"}


async def newest_enclosure_per_show(settings: Any) -> dict[str, str]:
    """The most recent enclosure for each podcast — one representative chain."""
    password = os.environ.get("PODAGENT_COUCHDB_PASSWORD") or ""
    if not password and settings.couchdb_password:
        password = settings.couchdb_password.get_secret_value()
    async with httpx.AsyncClient(
        base_url=settings.couchdb.url.rstrip("/"),
        auth=(settings.couchdb.user, password),
        timeout=30,
    ) as couch:
        response = await couch.post(
            f"/{settings.couchdb.db}/_find",
            json={
                "selector": {"type": "episode", "enclosure_url": {"$exists": True}},
                "fields": ["podcast_slug", "enclosure_url", "published_at"],
                "limit": 5000,
            },
        )
        response.raise_for_status()
        docs = response.json()["docs"]
    newest: dict[str, str] = {}
    for doc in sorted(docs, key=lambda d: d.get("published_at") or "", reverse=True):
        newest.setdefault(doc["podcast_slug"], doc["enclosure_url"])
    return newest


async def walk(
    client: httpx.AsyncClient, guard: UrlGuard, url: str, feed_url: str | None
) -> tuple[list[str], str | None]:
    """Every host the fetch would touch, and the one that refuses it, if any."""
    hosts: list[str] = []
    for _ in range(MAX_REDIRECTS + 1):
        host = urlparse(url).hostname or "?"
        try:
            guard.check(url, related_to=feed_url)
        except UrlRejected:
            hosts.append(f"[{host}]")
            return hosts, host
        hosts.append(host)
        try:
            response = await client.request("GET", url, headers=PEEK)
        except httpx.HTTPError as exc:
            return hosts, f"{type(exc).__name__}"
        if response.status_code not in (301, 302, 303, 307, 308):
            return hosts, None
        location = response.headers.get("location")
        if not location:
            return hosts, None
        url = urljoin(url, location)
    return hosts, f"more than {MAX_REDIRECTS} redirects"


async def main() -> int:
    settings = load_settings()
    guard = UrlGuard(settings.security)
    feeds = {p.slug: p.feed_url for p in settings.podcasts}
    enclosures = await newest_enclosure_per_show(settings)
    refused: list[tuple[str, str]] = []

    async with httpx.AsyncClient(
        timeout=20, follow_redirects=False, headers={"User-Agent": USER_AGENT}
    ) as client:
        for slug, enclosure in sorted(enclosures.items()):
            hosts, problem = await walk(client, guard, enclosure, feeds.get(slug))
            print(f"{'ok  ' if problem is None else 'FAIL'} {slug:34} {' -> '.join(hosts)}")
            if problem:
                refused.append((slug, problem))

    print()
    if not refused:
        print(f"All {len(feeds)} chains pass the guard.")
        return 0
    print(f"{len(refused)} show(s) would fetch nothing:")
    for slug, problem in refused:
        print(f"  {slug}: {problem}")
    print("\nAdd the bracketed host to security.cdn_allowlist in config.yaml.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
