"""Stage 3 — transcript acquisition (§4).

Ordered strategy, first success wins:

1. feed-provided ``<podcast:transcript>`` (txt / vtt / srt / Podcasting 2.0 JSON)
2. show-notes scrape, only for shows with an explicit ``transcript_selector``
   (generic scraping is never attempted)
3. audio download + ASR

Concurrency is capped here rather than in the runner so the limits hold no matter
who calls: one ASR job at a time, two concurrent audio downloads (§4).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ..config import Settings
from ..db import Doc, Store
from ..logging_setup import get_logger
from ..models import TranscriptResult
from ..net import (
    DownloadInterrupted,
    DownloadTooLarge,
    FetchPolicy,
    UrlGuard,
    UrlRejected,
    download_to_file,
    fetch_text,
)
from ..podcasts import PodcastRecord, PodcastRegistry
from ..sanitize import html_to_text, slugify
from ..utils import asr_run_doc_id, iso_now
from .asr import ASRBackend, ASRResult, ASRUnavailable
from .normalize import normalize_transcript

log = get_logger(__name__)

#: Anything shorter than this is a stub ("transcript coming soon"), a paywall
#: notice, or a failed render — not a usable transcript. Try the next strategy.
MIN_TRANSCRIPT_CHARS = 400


def transcript_page_url(link: str | None, sub: tuple[str, str] | None) -> str | None:
    """The page a show's transcript is actually on, given its episode link.

    Publishers commonly link to show notes and keep the transcript on a sibling
    page. Applied once and only to the tail of the URL: rewriting an early
    segment could move the request to another host, and a substitution from a
    config file must not be able to do that. A pattern that does not appear
    leaves the link alone, so a stale rule degrades to scraping the notes page
    rather than fetching something unintended.
    """
    if not link or not sub:
        return link
    old, new = sub
    if not old or old not in link:
        return link
    head, _, tail = link.rpartition(old)
    rewritten = f"{head}{new}{tail}"
    # Same host, or it is not a sibling page and the rule is wrong.
    if urlparse(rewritten).netloc != urlparse(link).netloc:
        log.warning("transcript.url_sub_ignored", link=link, rewritten=rewritten)
        return link
    return rewritten


class TranscriptUnavailable(Exception):
    """Every acquisition strategy failed for this episode.

    ``retryable`` separates "this did not work *this time*" — a download that
    timed out, a scrape that 503'd — from "there is nothing here to try": the
    podcast publishes no transcript, no scrape selector is configured, and local
    transcription is off. The second cannot succeed on a later run, so retrying
    it three times only produces three identical warnings and delays the
    episode's description-only summary.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class TranscriptAcquirer:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        client: httpx.AsyncClient,
        guard: UrlGuard,
        asr_backend: ASRBackend,
        registry: PodcastRegistry | None = None,
    ) -> None:
        self._registry = registry or PodcastRegistry(settings)
        self._settings = settings
        self._store = store
        self._client = client
        self._guard = guard
        self._asr = asr_backend
        self._asr_sem = asyncio.Semaphore(settings.asr.asr_concurrency)
        self._download_sem = asyncio.Semaphore(settings.asr.download_concurrency)
        self._max_text_bytes = settings.security.max_text_download_mb * 1024 * 1024

    def _policy(self, podcast: PodcastRecord | None) -> FetchPolicy:
        """The guard, travelling with the request so every hop is checked.

        Everything a podcast points at is feed-supplied and therefore untrusted:
        the full allowlist applies to each hop, not merely to the URL the fetch
        started at. An unknown podcast gets no ``related_to``, which is the
        stricter reading — the CDN allowlist alone must justify the fetch.
        """
        return FetchPolicy(self._guard, related_to=podcast.feed_url if podcast else None)

    async def acquire(self, episode: Doc, *, allow_asr: bool = True) -> TranscriptResult:
        """Acquire a transcript, optionally without falling back to ASR.

        ``allow_asr=False`` is how archive backfill stays affordable: an episode
        that would need transcription is abandoned rather than queued, because
        the untranscribed archive runs to thousands of audio hours (roadmap A1).
        """
        episode_id = episode["_id"]
        podcast = self._registry.podcast_by_slug(episode["podcast_slug"])
        failures: list[str] = []

        # True once a strategy has done something that could have worked and did
        # not — a fetch, a scrape, a transcription. A strategy that was simply
        # not available (nothing published, no selector, transcription off) is
        # not an attempt, and will be just as unavailable on the next run.
        attempted = False

        if result := await self._try_feed_transcripts(episode, podcast, failures):
            return result
        attempted |= bool(episode.get("feed_transcripts"))

        if result := await self._try_scrape(episode, podcast, failures):
            return result
        attempted |= bool(podcast and podcast.transcript_selector and episode.get("link"))

        if not allow_asr:
            failures.append("local transcription is off for this episode")
        else:
            attempted = True
            if result := await self._try_asr(episode, podcast, failures):
                return result

        detail = "; ".join(failures) or "no strategy applicable"
        raise TranscriptUnavailable(
            f"{episode_id}: no transcript from any strategy ({detail})",
            retryable=attempted,
        )

    async def _record_asr_run(self, episode: Doc, result: ASRResult, *, chars: int) -> None:
        """Keep a durable row for a transcription, beside the model-call rows.

        Deliberately its own document type rather than an `llm_call`: that shape
        is token-and-cost shaped, and an ASR run has neither. Folding it in would
        mean five null columns per row and, worse, a `calls` count and an average
        latency that mixed a twelve-second triage with a forty-minute
        transcription — destroying the two numbers most worth watching.

        Never raises: telemetry losing a row must not fail an episode that has
        just been transcribed successfully.
        """
        cfg = self._settings.asr
        elapsed = float(result.elapsed_s or 0.0)
        audio = float(result.duration_s or 0)
        try:
            await self._store.create(
                {
                    "_id": asr_run_doc_id(),
                    "type": "asr_run",
                    "episode_id": episode["_id"],
                    "podcast_slug": episode.get("podcast_slug"),
                    "model": cfg.model,
                    "device": cfg.device,
                    "compute_type": cfg.compute_type,
                    "backend": cfg.backend,
                    "audio_duration_s": int(audio) or None,
                    "elapsed_s": round(elapsed, 1) or None,
                    "realtime_factor": round(audio / elapsed, 2) if elapsed and audio else None,
                    "chars": chars,
                    "language": result.language,
                    "ts": iso_now(),
                }
            )
        except Exception as exc:
            log.warning("asr.run_not_recorded", episode_id=episode["_id"], error=str(exc))

    # --- strategy 1: feed transcript ----------------------------------------

    async def _try_feed_transcripts(
        self, episode: Doc, podcast: PodcastRecord | None, failures: list[str]
    ) -> TranscriptResult | None:
        candidates = episode.get("feed_transcripts") or []
        for candidate in candidates:
            url = candidate.get("url")
            if not url:
                continue
            try:
                raw = await fetch_text(
                    self._client,
                    url,
                    max_bytes=self._max_text_bytes,
                    policy=self._policy(podcast),
                )
                text = normalize_transcript(raw, candidate.get("type", ""), url)
                if len(text) < MIN_TRANSCRIPT_CHARS:
                    failures.append(f"feed transcript too short ({len(text)} chars) at {url}")
                    continue
                log.info(
                    "transcript.from_feed",
                    episode_id=episode["_id"],
                    url=url,
                    declared_type=candidate.get("type"),
                    chars=len(text),
                )
                return TranscriptResult(text=text, source="feed")
            except (UrlRejected, DownloadTooLarge) as exc:
                failures.append(f"feed transcript rejected: {exc}")
            except httpx.HTTPError as exc:
                failures.append(f"feed transcript fetch failed ({type(exc).__name__}): {exc}")
        return None

    # --- strategy 2: configured scrape --------------------------------------

    async def _try_scrape(
        self, episode: Doc, podcast: PodcastRecord | None, failures: list[str]
    ) -> TranscriptResult | None:
        if podcast is None or not podcast.transcript_selector:
            return None
        page_url = transcript_page_url(episode.get("link"), podcast.transcript_url_sub)
        if not page_url:
            failures.append("scrape skipped: episode has no link")
            return None
        try:
            html = await fetch_text(
                self._client,
                page_url,
                max_bytes=self._max_text_bytes,
                policy=self._policy(podcast),
            )
        except (UrlRejected, DownloadTooLarge, httpx.HTTPError) as exc:
            failures.append(f"scrape fetch failed: {exc}")
            return None

        # lxml is used for speed; the parser never executes anything from the page.
        soup = BeautifulSoup(html, "lxml")
        nodes = soup.select(podcast.transcript_selector)
        if not nodes:
            failures.append(f"scrape selector {podcast.transcript_selector!r} matched nothing")
            return None
        text = html_to_text("\n\n".join(str(node) for node in nodes))
        if len(text) < MIN_TRANSCRIPT_CHARS:
            failures.append(f"scraped transcript too short ({len(text)} chars)")
            return None
        log.info(
            "transcript.from_scrape",
            episode_id=episode["_id"],
            url=page_url,
            selector=podcast.transcript_selector,
            rewritten=page_url != episode.get("link"),
            chars=len(text),
        )
        return TranscriptResult(text=text, source="scrape")

    # --- strategy 3: ASR ----------------------------------------------------

    async def _try_asr(
        self, episode: Doc, podcast: PodcastRecord | None, failures: list[str]
    ) -> TranscriptResult | None:
        enclosure = episode.get("enclosure_url")
        if not enclosure:
            failures.append("ASR skipped: no audio enclosure")
            return None

        max_bytes = self._settings.asr.max_audio_mb * 1024 * 1024
        declared = episode.get("enclosure_bytes")
        if isinstance(declared, int) and declared > max_bytes:
            failures.append(f"ASR skipped: enclosure declares {declared} bytes > cap {max_bytes}")
            return None

        audio_path = self._audio_path(episode)
        try:
            self._guard.check(enclosure, related_to=podcast.feed_url if podcast else None)
        except UrlRejected as exc:
            failures.append(f"ASR skipped: {exc}")
            return None

        try:
            async with self._download_sem:
                log.info("transcript.audio_download_start", episode_id=episode["_id"])
                size = await download_to_file(
                    self._client,
                    enclosure,
                    audio_path,
                    max_bytes=max_bytes,
                    policy=self._policy(podcast),
                )
            log.info(
                "transcript.audio_downloaded",
                episode_id=episode["_id"],
                bytes=size,
                path=str(audio_path),
            )

            async with self._asr_sem:
                log.info("transcript.asr_start", episode_id=episode["_id"], backend=self._asr.name)
                result = await self._asr.transcribe(
                    audio_path, language=self._settings.asr.language
                )
            text = normalize_transcript(result.text)
            if len(text) < MIN_TRANSCRIPT_CHARS:
                failures.append(f"ASR produced only {len(text)} chars")
                return None
            log.info(
                "transcript.from_asr",
                episode_id=episode["_id"],
                chars=len(text),
                detected_language=result.language,
                audio_duration_s=result.duration_s,
            )
            await self._record_asr_run(episode, result, chars=len(text))
            return TranscriptResult(
                text=text,
                source="asr",
                detected_language=result.language,
                duration_s=result.duration_s,
            )
        except (DownloadTooLarge, UrlRejected) as exc:
            failures.append(f"audio download rejected: {exc}")
            return None
        except DownloadInterrupted as exc:
            # Resumption was already tried; another attempt now would restart the
            # same losing transfer. The episode's own retry schedule handles it.
            failures.append(f"audio download interrupted: {exc}")
            return None
        except httpx.HTTPError as exc:
            failures.append(f"audio download failed ({type(exc).__name__}): {exc}")
            return None
        except ASRUnavailable:
            # Backend problem, not an episode problem — let the caller retry later.
            raise
        finally:
            if not self._settings.asr.keep_audio:
                audio_path.unlink(missing_ok=True)
            else:
                log.debug("transcript.audio_kept", path=str(audio_path))

    def _audio_path(self, episode: Doc) -> Path:
        """Quarantined download location (§10.2): never inside the digest output."""
        quarantine = self._settings.output.work_dir / "audio"
        # Episode ids are hashes, so the name is safe; the slug is for humans.
        stem = f"{slugify(episode['podcast_slug'])}-{episode['_id'].split(':')[-1][:16]}"
        return quarantine / f"{stem}.audio"
