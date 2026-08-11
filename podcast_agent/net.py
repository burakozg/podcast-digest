"""Outbound HTTP with the guardrails from §10.2.

Every outbound request in the system goes through this module: explicit timeouts,
a redirect cap, an http(s)-only scheme check, a registrable-domain allowlist, and
hard byte caps on anything downloaded. Feed content is untrusted, so the URLs it
supplies are treated as attacker-chosen.

**Redirects are walked here, not by httpx.** Vetting only the URL a fetch starts
at leaves the guard checking one host and the transport connecting to another:
enclosure chains are routinely four deep through analytics prefixers, and any
host in such a chain can answer with a ``Location`` pointing at a LAN address.
Every request therefore goes out with ``follow_redirects=False`` and each hop is
re-checked before it is followed.

Names are also resolved and their addresses rejected if private, so an
allowlisted domain cannot reach a LAN service by pointing its A record inward.
Accepted residual: the address is checked and then connected to separately, so a
resolver that answers differently between the two would not be caught (TOCTOU).
Closing that needs a connection-pinning transport, which is out of scope here.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urljoin, urlparse

import httpx
import tldextract

from .config import SecurityConfig
from .logging_setup import get_logger

log = get_logger(__name__)

USER_AGENT: Final = "podcast-digest-agent/1.0 (+self-hosted; contact: local admin)"

#: Redirect cap (§10.2).
MAX_REDIRECTS: Final = 5

#: Statuses that mean "go somewhere else". Deliberately *not* every 3xx:
#: 304 Not Modified is the answer to a conditional feed poll and carries no
#: Location, so treating the whole range as redirects broke the cheap no-op
#: path that most ingest runs take.
REDIRECT_CODES: Final = frozenset({301, 302, 303, 307, 308})

#: Content types accepted for an audio enclosure download.
AUDIO_CONTENT_TYPES: Final = ("audio/", "application/octet-stream")

#: Mid-transfer failures worth resuming instead of restarting. These all mean
#: the connection died with the response incomplete — podcast CDNs do this
#: routinely on large enclosures — and say nothing about the bytes already
#: written, which remain valid.
RESUMABLE_TRANSFER_ERRORS: Final = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ReadTimeout,
)

#: How many times a dropped transfer is resumed before the download is failed.
#: Without this an interrupted 27 MB enclosure was re-fetched from byte zero on
#: every attempt, three times, and still failed.
MAX_RESUME_ATTEMPTS: Final = 3

#: Content types accepted for transcript/page fetches.
TEXT_CONTENT_TYPES: Final = (
    "text/",
    "application/json",
    "application/x-subrip",
    "application/srt",
    "application/xml",
    "application/octet-stream",
)

#: Offline suffix-list extractor — never fetches the public suffix list at
#: runtime (the container has no reason to, and a network call here would be a
#: surprising dependency). Uses the snapshot bundled with tldextract.
_extract = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)


async def _resolve(host: str) -> list[str]:
    """Addresses ``host`` resolves to.

    Its own function so tests can replace it: the suite must never touch a real
    resolver, and a hostname that happens to exist would otherwise make the
    address check depend on someone else's DNS.
    """
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [str(info[4][0]) for info in infos]


def _is_reachable_publicly(address: str) -> bool:
    """Whether ``address`` is a public unicast address we are willing to fetch."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        # Unparseable is not "safe": refuse rather than guess.
        return False
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_reserved
        or parsed.is_unspecified
        or parsed.is_multicast
    )


class UrlRejected(Exception):
    """URL failed a security guard. Caller should log and skip, not retry."""


class DownloadTooLarge(Exception):
    """Response exceeded its configured byte cap; the transfer was aborted."""


class DownloadInterrupted(Exception):
    """The transfer was cut short and could not be resumed.

    Distinct from a plain transport error: it means we did make progress and
    resumption was attempted, so a caller retrying immediately will most likely
    hit the same wall.
    """


def registrable_domain(url: str) -> str:
    """eTLD+1 of ``url``, or '' when it cannot be determined."""
    host = urlparse(url).hostname or ""
    result = _extract(host)
    return result.registered_domain.lower() if result.registered_domain else ""


def build_client(*, timeout: float = 30.0, follow_redirects: bool = True) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10.0),
        follow_redirects=follow_redirects,
        max_redirects=MAX_REDIRECTS,
        headers={"User-Agent": USER_AGENT},
    )


class UrlGuard:
    """Decides whether a feed-supplied URL may be fetched."""

    def __init__(self, cfg: SecurityConfig) -> None:
        self._enforce = cfg.enforce_domain_allowlist
        self._allowlist = {d.lower().lstrip(".") for d in cfg.cdn_allowlist}

    @property
    def fingerprint(self) -> str:
        """Short digest of the rules this guard enforces.

        Stored beside a feed's cached validators so a change to the allowlist
        can invalidate them. Without it, widening the allowlist has no effect on
        any feed whose content has not changed: the poll answers 304, the body
        is never re-read, and entries rejected under the old rules stay rejected
        forever. That is not hypothetical — it kept a podcast at zero episodes
        across a fix that was already deployed and correct.
        """
        material = repr((self._enforce, sorted(self._allowlist)))
        return hashlib.sha256(material.encode()).hexdigest()[:12]

    def check(self, url: str, *, related_to: str | None = None, allowlist: bool = True) -> str:
        """Return ``url`` if permitted, else raise :class:`UrlRejected`.

        ``related_to`` is the feed URL the target came from: sharing its
        registrable domain is sufficient, otherwise the CDN allowlist applies.

        ``allowlist=False`` keeps the scheme and host checks but drops the CDN
        allowlist, for targets the *owner* chose rather than a feed — the feed
        URL itself, which may legitimately redirect to a publisher on another
        domain. It never widens what an untrusted URL may reach.

        Fails closed when the registrable domain cannot be determined — which is
        the case for bare IPs, ``localhost`` and special-use TLDs. A self-hosted
        feed on such a host still works via the same-host shortcut below; a
        *cross-host* jump to one needs ``enforce_domain_allowlist: false``.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise UrlRejected(f"scheme {parsed.scheme!r} not allowed: {url!r}")
        if not parsed.hostname:
            raise UrlRejected(f"no host in URL: {url!r}")
        if not self._enforce:
            return url

        # Same host as the feed: trivially safe, and the only way a feed hosted on
        # a LAN name or IP can reference its own audio.
        if related_to and parsed.hostname == (urlparse(related_to).hostname or None):
            return url

        if not allowlist:
            return url

        target_domain = registrable_domain(url)
        if not target_domain:
            raise UrlRejected(
                f"cannot determine registrable domain for {url!r} "
                "(bare IP, localhost or special-use TLD)"
            )
        if related_to and target_domain == registrable_domain(related_to):
            return url
        if target_domain in self._allowlist:
            return url
        raise UrlRejected(
            f"{target_domain!r} is neither the feed's own domain nor on the CDN "
            f"allowlist (url={url!r})"
        )

    async def check_address(self, url: str, *, related_to: str | None = None) -> None:
        """Refuse ``url`` when its host resolves anywhere private.

        Separate from :meth:`check` and asynchronous because it costs a DNS
        lookup: ``check`` runs at ingest time on every enclosure of every
        episode to decide what to *store*, where no connection is made and
        blocking the loop for a resolution would be absurd. This runs where a
        connection is actually about to happen.

        Two exemptions, both deliberate. A host equal to the feed's own is
        allowed to be a LAN address — self-hosting a feed beside the agent is
        documented and supported — and the whole check follows
        ``enforce_domain_allowlist``, so the existing escape hatch turns off all
        of the network guards together rather than half of them.
        """
        if not self._enforce:
            return
        host = urlparse(url).hostname
        if not host:
            raise UrlRejected(f"no host in URL: {url!r}")
        if related_to and host == (urlparse(related_to).hostname or None):
            return
        try:
            addresses = await _resolve(host)
        except OSError as exc:
            raise UrlRejected(f"cannot resolve {host!r}: {exc}") from exc
        # *Every* answer must be public: a name that returns one public and one
        # private address would otherwise be a coin toss decided by the resolver.
        for address in addresses:
            if not _is_reachable_publicly(address):
                raise UrlRejected(
                    f"{host!r} resolves to non-public address {address} — "
                    "refusing to fetch a LAN or loopback target"
                )

    def permits(self, url: str, *, related_to: str | None = None) -> bool:
        try:
            self.check(url, related_to=related_to)
        except UrlRejected:
            return False
        return True


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    """Who vets each hop of a fetch, and how strictly.

    Passed down into the transfer functions so the guard travels with the
    request. Without it a caller can check a URL and then follow a redirect
    somewhere else entirely, which is precisely the hole this closes.
    """

    guard: UrlGuard
    #: The feed the target came from, for the same-domain and same-host arms.
    related_to: str | None = None
    #: False for owner-supplied targets (the feed URL itself): the CDN
    #: allowlist does not apply, every other check still does.
    allowlist: bool = True

    async def vet(self, url: str) -> None:
        self.guard.check(url, related_to=self.related_to, allowlist=self.allowlist)
        await self.guard.check_address(url, related_to=self.related_to)


async def _vet(url: str, policy: FetchPolicy | None) -> None:
    """Check one hop. Without a policy, only the scheme and host are enforced."""
    if policy is not None:
        await policy.vet(url)
        return
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UrlRejected(f"scheme {parsed.scheme!r} not allowed: {url!r}")
    if not parsed.hostname:
        raise UrlRejected(f"no host in URL: {url!r}")


@asynccontextmanager
async def _guarded_stream(
    client: httpx.AsyncClient,
    url: str,
    *,
    policy: FetchPolicy | None,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[httpx.Response]:
    """Stream ``url``, vetting every redirect hop before following it.

    Yields the first non-redirect response. Each hop is resolved against the URL
    that produced it, so a relative ``Location`` cannot be used to smuggle a
    host past the check.
    """
    current = url
    await _vet(current, policy)
    for _ in range(MAX_REDIRECTS + 1):
        try:
            async with client.stream(
                "GET", current, headers=headers or {}, follow_redirects=False
            ) as response:
                if response.status_code not in REDIRECT_CODES:
                    yield response
                    return
                location = response.headers.get("location")
                if not location:
                    raise UrlRejected(
                        f"{current!r}: {response.status_code} redirect without a Location header"
                    )
                target = urljoin(current, location)
        except httpx.InvalidURL as exc:
            # httpx parses the Location to populate `next_request` even when it
            # is not following redirects, so a malformed target fails there
            # before this code sees it. It is still a refused URL, and callers
            # already handle UrlRejected.
            raise UrlRejected(f"{current!r}: unusable redirect target ({exc})") from exc
        # Outside the context manager: the redirect's own body is of no
        # interest and its connection is released before the next request.
        await _vet(target, policy)
        log.debug("net.redirect_followed", frm=current, to=target)
        current = target
    raise UrlRejected(f"{url!r}: more than {MAX_REDIRECTS} redirects")


async def get_guarded(
    client: httpx.AsyncClient,
    url: str,
    *,
    policy: FetchPolicy | None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """GET with the redirect chain vetted hop by hop, body fully read."""
    async with _guarded_stream(client, url, policy=policy, headers=headers) as response:
        await response.aread()
        return response


def _content_type_ok(content_type: str | None, allowed: Iterable[str]) -> bool:
    if not content_type:
        # Some hosts omit the header. Byte caps still apply, and the payload is
        # only ever handed to a parser (never executed), so allow it through.
        return True
    main = content_type.split(";", 1)[0].strip().lower()
    return any(main.startswith(prefix) for prefix in allowed)


async def fetch_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int,
    allowed_content_types: Iterable[str] = TEXT_CONTENT_TYPES,
    policy: FetchPolicy | None = None,
) -> str:
    """GET a text resource, refusing anything over ``max_bytes``."""
    async with _guarded_stream(client, url, policy=policy) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        if not _content_type_ok(content_type, allowed_content_types):
            raise UrlRejected(f"unexpected content-type {content_type!r} for {url!r}")
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise DownloadTooLarge(f"{url}: declared {declared} bytes > cap {max_bytes}")
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise DownloadTooLarge(f"{url}: exceeded cap {max_bytes} bytes")
            chunks.append(chunk)
    encoding = response.charset_encoding or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace")


async def _stream_into(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    offset: int,
    max_bytes: int,
    allowed_content_types: Iterable[str],
    policy: FetchPolicy | None,
) -> int:
    """Stream ``url`` into ``dest`` starting at ``offset``; return the new total.

    ``offset`` > 0 asks the server to continue from there. A server that does not
    support ranges answers 200 with the whole body instead of 206, in which case
    the bytes on disk are stale and the file is rewritten from the start.

    The redirect chain is walked afresh from ``url`` on every call, never
    resumed against a remembered final hop: CDN redirect targets are commonly
    signed and short-lived, so the URL that served the first half is often
    already invalid by the time a resume is attempted.
    """
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    async with _guarded_stream(client, url, policy=policy, headers=headers) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        if not _content_type_ok(content_type, allowed_content_types):
            raise UrlRejected(f"unexpected content-type {content_type!r} for {url!r}")

        resuming = bool(offset) and response.status_code == 206
        total = offset if resuming else 0

        # On a 206 content-length covers only the remainder, so the cap has to be
        # checked against what the file will end up holding, not the response.
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and total + int(declared) > max_bytes:
            raise DownloadTooLarge(
                f"{url}: declared {total + int(declared)} bytes > cap {max_bytes}"
            )

        with dest.open("ab" if resuming else "wb") as fh:
            async for chunk in response.aiter_bytes(chunk_size=256 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise DownloadTooLarge(f"{url}: exceeded cap {max_bytes} bytes")
                fh.write(chunk)
    return total


async def download_to_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    max_bytes: int,
    allowed_content_types: Iterable[str] = AUDIO_CONTENT_TYPES,
    max_resume_attempts: int = MAX_RESUME_ATTEMPTS,
    policy: FetchPolicy | None = None,
) -> int:
    """Stream ``url`` to ``dest``, never buffering the body in memory (§10.2).

    Returns the byte count. A transfer the server cuts short is resumed with a
    ``Range`` request rather than restarted — large enclosures were failing after
    three full re-downloads of the same 27 MB. On final failure the partial file
    is removed so a truncated download can never be handed to the ASR backend.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        for attempt in range(max_resume_attempts + 1):
            try:
                total = await _stream_into(
                    client,
                    url,
                    dest,
                    offset=total,
                    max_bytes=max_bytes,
                    allowed_content_types=allowed_content_types,
                    policy=policy,
                )
                break
            except RESUMABLE_TRANSFER_ERRORS as exc:
                written = dest.stat().st_size if dest.exists() else 0
                # No progress this round means resuming is just a slower way to
                # fail: the server is refusing at the same point, or ignoring the
                # range and re-sending a body it cannot finish either.
                if written <= total or attempt >= max_resume_attempts:
                    raise DownloadInterrupted(
                        f"{url}: transfer cut short after {written} bytes "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc
                total = written
                log.info(
                    "net.download_resuming",
                    url=url,
                    offset=total,
                    attempt=attempt + 1,
                    error=type(exc).__name__,
                )
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    log.debug("net.download_complete", url=url, bytes=total, dest=str(dest))
    return total
