"""Outbound download behaviour (§10.2).

The byte caps and the guard have tests elsewhere, through the stages that use
them. What is here is resumption, which those tests cannot reach: it only shows
up when a server hangs up mid-body, which is exactly what podcast CDNs were
doing to 27 MB enclosures — three full re-downloads and then a failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from podcast_agent import net
from podcast_agent.config import SecurityConfig
from podcast_agent.net import (
    AUDIO_CONTENT_TYPES,
    MAX_REDIRECTS,
    DownloadInterrupted,
    DownloadTooLarge,
    FetchPolicy,
    UrlGuard,
    UrlRejected,
    download_to_file,
    fetch_text,
)

#: Bigger than one read chunk on purpose. `aiter_bytes` buffers up to its chunk
#: size, so anything less than that is still inside httpx when the connection
#: dies and never reaches the file — a resume therefore restarts from the last
#: whole chunk, not the last byte. Harmless on a 27 MB enclosure, invisible on a
#: 4 KB fixture.
CHUNK = 256 * 1024
BODY = bytes(i % 251 for i in range(4 * CHUNK))
URL = "https://cdn.example.com/ep.mp3"
AUDIO = {"content-type": "audio/mpeg"}
TEXT = {"content-type": "text/plain"}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _Cutter:
    """Serves ``BODY`` but hangs up after ``cut`` bytes, ``fails`` times.

    The stream has to be genuinely interrupted rather than merely short: what
    the real CDNs did was send a `Content-Length` and then close the connection
    early, which surfaces as ``RemoteProtocolError`` part-way through the body.
    """

    def __init__(
        self, cut: int, fails: int, *, honour_range: bool = True, declare: bool = True
    ) -> None:
        self.cut = cut
        self.fails = fails
        self.honour_range = honour_range
        self.declare = declare
        self.ranges: list[str | None] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        offset = 0
        header = request.headers.get("range")
        self.ranges.append(header)
        resuming = bool(header) and self.honour_range
        if resuming:
            offset = int(str(header).removeprefix("bytes=").rstrip("-"))

        remaining = BODY[offset:]
        headers = dict(AUDIO)
        if self.declare or resuming:
            headers["content-length"] = str(len(remaining))
        status = 206 if resuming else 200
        if self.fails <= 0:
            return httpx.Response(status, headers=headers, content=remaining)

        self.fails -= 1
        cut = self.cut

        async def _dies() -> Any:
            if cut:
                yield remaining[:cut]
            raise httpx.RemoteProtocolError(
                f"peer closed connection without sending complete message body "
                f"(received {cut} bytes, expected {len(remaining)})"
            )

        return httpx.Response(status, headers=headers, content=_dies())


class TestResume:
    async def test_a_cut_transfer_resumes_instead_of_restarting(self, tmp_path: Path) -> None:
        cutter = _Cutter(cut=CHUNK + 10, fails=1)
        async with _client(cutter) as client:
            size = await download_to_file(client, URL, tmp_path / "a.audio", max_bytes=10**7)
        assert size == len(BODY)
        # Reassembled in the right order, not merely the right length.
        assert (tmp_path / "a.audio").read_bytes() == BODY
        # The second request asked to continue rather than start over.
        assert cutter.ranges[0] is None
        assert cutter.ranges[1] == f"bytes={CHUNK}-"

    async def test_several_cuts_in_a_row_still_complete(self, tmp_path: Path) -> None:
        cutter = _Cutter(cut=CHUNK, fails=3)
        async with _client(cutter) as client:
            size = await download_to_file(client, URL, tmp_path / "a.audio", max_bytes=10**7)
        assert size == len(BODY)
        assert (tmp_path / "a.audio").read_bytes() == BODY

    async def test_it_gives_up_rather_than_retrying_forever(self, tmp_path: Path) -> None:
        cutter = _Cutter(cut=CHUNK, fails=99)
        async with _client(cutter) as client:
            with pytest.raises(DownloadInterrupted):
                await download_to_file(client, URL, tmp_path / "a.audio", max_bytes=10**7)
        assert len(cutter.ranges) == 4  # the first try plus MAX_RESUME_ATTEMPTS

    async def test_a_truncated_file_never_survives(self, tmp_path: Path) -> None:
        """The invariant ASR depends on: a partial file must not reach Whisper."""
        dest = tmp_path / "a.audio"
        async with _client(_Cutter(cut=CHUNK, fails=99)) as client:
            with pytest.raises(DownloadInterrupted):
                await download_to_file(client, URL, dest, max_bytes=10**7)
        assert not dest.exists()

    async def test_a_server_ignoring_range_starts_over_cleanly(self, tmp_path: Path) -> None:
        """A 200 in reply to a Range request means the bytes on disk are stale."""
        cutter = _Cutter(cut=CHUNK, fails=1, honour_range=False)
        async with _client(cutter) as client:
            size = await download_to_file(client, URL, tmp_path / "a.audio", max_bytes=10**7)
        assert size == len(BODY)
        assert (tmp_path / "a.audio").read_bytes() == BODY

    async def test_no_progress_means_no_further_attempts(self, tmp_path: Path) -> None:
        """Resuming from a server that sends nothing is a slower way to fail."""
        cutter = _Cutter(cut=0, fails=99)
        async with _client(cutter) as client:
            with pytest.raises(DownloadInterrupted):
                await download_to_file(client, URL, tmp_path / "a.audio", max_bytes=10**7)
        assert len(cutter.ranges) == 1


class TestCapsStillApply:
    async def test_the_cap_counts_bytes_already_on_disk(self, tmp_path: Path) -> None:
        """On a 206, content-length covers only the remainder.

        Checking it against the cap in isolation would let a resumed download
        exceed the limit by whatever was already written.
        """
        # The first response declares nothing, so only the resumed one can trip
        # the cap — and only if the bytes already written are counted with it.
        cutter = _Cutter(cut=CHUNK, fails=1, declare=False)
        async with _client(cutter) as client:
            with pytest.raises(DownloadTooLarge):
                await download_to_file(client, URL, tmp_path / "a.audio", max_bytes=len(BODY) - 1)

    async def test_an_oversized_declaration_is_refused_before_the_body(
        self, tmp_path: Path
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers=AUDIO, content=BODY)

        async with _client(handler) as client:
            with pytest.raises(DownloadTooLarge):
                await download_to_file(client, URL, tmp_path / "a.audio", max_bytes=100)

    async def test_the_content_type_is_still_checked(self, tmp_path: Path) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/html"}, content=BODY)

        async with _client(handler) as client:
            with pytest.raises(Exception, match="content-type"):
                await download_to_file(
                    client,
                    URL,
                    tmp_path / "a.audio",
                    max_bytes=10**7,
                    allowed_content_types=AUDIO_CONTENT_TYPES,
                )


class _Chain:
    """Serves a redirect chain, then a body.

    ``hops`` maps a URL to where it redirects; anything not in it answers 200.
    """

    def __init__(
        self,
        hops: dict[str, str],
        *,
        status: int = 302,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.hops = hops
        self.status = status
        self.headers = AUDIO if headers is None else headers
        self.seen: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen.append(url)
        if url in self.hops:
            return httpx.Response(self.status, headers={"location": self.hops[url]})
        return httpx.Response(200, content=b"x" * 1000, headers=self.headers)


def _guard(*allow: str, enforce: bool = True) -> UrlGuard:
    return UrlGuard(SecurityConfig(enforce_domain_allowlist=enforce, cdn_allowlist=list(allow)))


FEED = "https://podcast.example/feed.xml"


class TestEveryRedirectHopIsChecked:
    """Vetting only the starting URL checks one host and connects to another.

    Enclosure chains run four deep through analytics prefixers, and every host
    in one can answer with a Location of its choosing. httpx followed those
    hops internally, so the guard never saw them.
    """

    async def _download(self, chain: _Chain, tmp_path: Path, policy: FetchPolicy) -> int:
        return await download_to_file(
            _client(chain),
            "https://cdn-host.net/ep.mp3",
            tmp_path / "out.audio",
            max_bytes=10_000,
            policy=policy,
        )

    async def test_a_chain_of_allowed_hosts_completes(self, tmp_path: Path) -> None:
        chain = _Chain({"https://cdn-host.net/ep.mp3": "https://second.net/real.mp3"})
        policy = FetchPolicy(_guard("cdn-host.net", "second.net"), related_to=FEED)
        assert await self._download(chain, tmp_path, policy) == 1000

    async def test_a_hop_to_an_unlisted_domain_is_refused(self, tmp_path: Path) -> None:
        chain = _Chain({"https://cdn-host.net/ep.mp3": "https://elsewhere.net/real.mp3"})
        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        with pytest.raises(UrlRejected, match=r"elsewhere\.net"):
            await self._download(chain, tmp_path, policy)

    async def test_the_unlisted_hop_is_never_requested(self, tmp_path: Path) -> None:
        """Refused before the connection, not after reading the response."""
        chain = _Chain({"https://cdn-host.net/ep.mp3": "https://elsewhere.net/real.mp3"})
        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        with pytest.raises(UrlRejected):
            await self._download(chain, tmp_path, policy)
        assert chain.seen == ["https://cdn-host.net/ep.mp3"]

    async def test_a_hop_to_a_bare_lan_address_is_refused(self, tmp_path: Path) -> None:
        """The shape that matters: an allowlisted CDN pointing inward."""
        chain = _Chain({"https://cdn-host.net/ep.mp3": "http://192.168.1.1/admin"})
        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        with pytest.raises(UrlRejected):
            await self._download(chain, tmp_path, policy)

    async def test_a_hop_to_a_non_http_scheme_is_refused(self, tmp_path: Path) -> None:
        chain = _Chain({"https://cdn-host.net/ep.mp3": "file:///etc/passwd"})
        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        with pytest.raises(UrlRejected, match="scheme"):
            await self._download(chain, tmp_path, policy)

    async def test_an_unparseable_location_is_refused_cleanly(self, tmp_path: Path) -> None:
        """`javascript:alert(1)` fails inside httpx, which parses the Location
        to populate `next_request` even when told not to follow it. Callers
        handle UrlRejected, so it must not surface as a transport error."""
        chain = _Chain({"https://cdn-host.net/ep.mp3": "javascript:alert(1)"})
        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        with pytest.raises(UrlRejected):
            await self._download(chain, tmp_path, policy)

    async def test_a_relative_location_resolves_against_the_hop_that_sent_it(
        self, tmp_path: Path
    ) -> None:
        """A relative Location must not be able to smuggle a host past the check."""
        chain = _Chain({"https://cdn-host.net/ep.mp3": "/moved/real.mp3"})
        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        assert await self._download(chain, tmp_path, policy) == 1000
        assert chain.seen[-1] == "https://cdn-host.net/moved/real.mp3"

    async def test_too_many_hops_is_refused(self, tmp_path: Path) -> None:
        hops = {
            f"https://cdn-host.net/{i}": f"https://cdn-host.net/{i + 1}"
            for i in range(MAX_REDIRECTS + 2)
        }
        chain = _Chain(hops)
        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        with pytest.raises(UrlRejected, match="more than"):
            await download_to_file(
                _client(chain),
                "https://cdn-host.net/0",
                tmp_path / "out.audio",
                max_bytes=10_000,
                policy=policy,
            )

    async def test_a_redirect_without_a_location_is_refused(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302)

        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        with pytest.raises(UrlRejected, match="without a Location"):
            await download_to_file(
                _client(handler),
                "https://cdn-host.net/ep.mp3",
                tmp_path / "out.audio",
                max_bytes=10_000,
                policy=policy,
            )

    async def test_the_owner_supplied_relaxation_still_checks_the_scheme(
        self, tmp_path: Path
    ) -> None:
        """`allowlist=False` drops one arm, not the guard."""
        chain = _Chain({FEED: "ftp://elsewhere.example/x"}, headers=TEXT)
        policy = FetchPolicy(_guard(), related_to=FEED, allowlist=False)
        with pytest.raises(UrlRejected, match="scheme"):
            await fetch_text(_client(chain), FEED, max_bytes=10_000, policy=policy)

    async def test_an_owner_supplied_feed_may_redirect_off_its_own_domain(self) -> None:
        """Feeds legitimately move to a publisher on another domain."""
        chain = _Chain({FEED: "https://publisher.example/rss"}, headers=TEXT)
        policy = FetchPolicy(_guard(), related_to=FEED, allowlist=False)
        assert await fetch_text(_client(chain), FEED, max_bytes=10_000, policy=policy)


class TestResumptionRewalksTheChain:
    async def test_a_resume_starts_from_the_original_url(self, tmp_path: Path) -> None:
        """CDN redirect targets are signed and short-lived, so the URL that
        served the first half is usually already invalid on a resume."""
        cutter = _Cutter(CHUNK, fails=1)
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            seen.append(url)
            if url == "https://cdn-host.net/start.mp3":
                return httpx.Response(302, headers={"location": URL})
            return cutter(request)

        policy = FetchPolicy(_guard("cdn-host.net", "example.com"), related_to=FEED)
        await download_to_file(
            _client(handler),
            "https://cdn-host.net/start.mp3",
            tmp_path / "out.audio",
            max_bytes=len(BODY) * 2,
            policy=policy,
        )
        # Twice through the redirect, once per attempt — not a resume aimed at
        # the expired target.
        assert seen.count("https://cdn-host.net/start.mp3") == 2


class TestPrivateAddressesAreRefused:
    """An allowlisted name whose A record points at the LAN.

    The guard reasons about names; this is the arm that reasons about where a
    name actually goes.
    """

    async def _fetch(self, monkeypatch, addresses: list[str], **policy_kw: Any) -> str:
        async def _resolve(host: str) -> list[str]:
            return addresses

        monkeypatch.setattr(net, "_resolve", _resolve)
        handler = _Chain({}, headers=TEXT)
        policy = FetchPolicy(_guard("cdn-host.net"), **policy_kw)
        return await fetch_text(
            _client(handler), "https://cdn-host.net/t.txt", max_bytes=10_000, policy=policy
        )

    async def test_a_public_address_passes(self, monkeypatch) -> None:
        assert await self._fetch(monkeypatch, ["93.184.216.34"], related_to=FEED)

    async def test_a_private_address_is_refused(self, monkeypatch) -> None:
        with pytest.raises(UrlRejected, match="non-public"):
            await self._fetch(monkeypatch, ["10.0.0.5"], related_to=FEED)

    async def test_loopback_is_refused(self, monkeypatch) -> None:
        with pytest.raises(UrlRejected, match="non-public"):
            await self._fetch(monkeypatch, ["127.0.0.1"], related_to=FEED)

    async def test_link_local_is_refused(self, monkeypatch) -> None:
        """169.254.169.254 is the cloud metadata address."""
        with pytest.raises(UrlRejected, match="non-public"):
            await self._fetch(monkeypatch, ["169.254.169.254"], related_to=FEED)

    async def test_one_private_answer_among_several_is_enough(self, monkeypatch) -> None:
        """Otherwise which address wins is decided by the resolver's ordering."""
        with pytest.raises(UrlRejected, match="non-public"):
            await self._fetch(monkeypatch, ["93.184.216.34", "192.168.1.10"], related_to=FEED)

    async def test_a_feed_on_the_same_host_may_be_on_the_lan(self, monkeypatch) -> None:
        """Self-hosting a feed beside the agent is documented and supported."""

        async def _resolve(host: str) -> list[str]:
            return ["192.168.1.50"]

        monkeypatch.setattr(net, "_resolve", _resolve)
        chain = _Chain({}, headers=TEXT)
        policy = FetchPolicy(_guard(), related_to="http://nas.lan/feed.xml", allowlist=False)
        assert await fetch_text(
            _client(chain), "http://nas.lan/feed.xml", max_bytes=10_000, policy=policy
        )

    async def test_the_escape_hatch_turns_the_whole_guard_off(self, monkeypatch) -> None:
        """`enforce_domain_allowlist: false` disables the network guards
        together, rather than leaving half of them on."""

        async def _resolve(host: str) -> list[str]:
            return ["10.0.0.5"]

        monkeypatch.setattr(net, "_resolve", _resolve)
        chain = _Chain({}, headers=TEXT)
        policy = FetchPolicy(_guard(enforce=False), related_to=FEED)
        assert await fetch_text(
            _client(chain), "https://anywhere.invalid/t.txt", max_bytes=10_000, policy=policy
        )

    async def test_a_resolver_failure_is_a_rejection(self, monkeypatch) -> None:
        async def _resolve(host: str) -> list[str]:
            raise OSError("no such host")

        monkeypatch.setattr(net, "_resolve", _resolve)
        chain = _Chain({}, headers=TEXT)
        policy = FetchPolicy(_guard("cdn-host.net"), related_to=FEED)
        with pytest.raises(UrlRejected, match="cannot resolve"):
            await fetch_text(
                _client(chain), "https://cdn-host.net/t.txt", max_bytes=10_000, policy=policy
            )
