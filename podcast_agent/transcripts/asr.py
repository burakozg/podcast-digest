"""ASR backends behind a narrow interface (§3, §10.4).

Two implementations, chosen by ``asr.backend``:

``local``
    faster-whisper in-process. Right on a machine with the CPU for it.

``remote``
    ``POST {asr.remote_url}/v1/audio/transcriptions`` — the OpenAI audio API
    shape, which speaches, faster-whisper-server, whisper.cpp's server and
    LocalAI all speak. Which machine does the work is then one URL, so a laptop
    today and a real server later cost a config change and a restart.

The interface was declared for exactly this before it had a second
implementation, and it paid off: production runs on a NAS whose realtime factor
is **0.11** — ten hours of CPU for a 68-minute episode — so the work has to
happen elsewhere, and the pipeline did not have to change to move it.

When the remote endpoint is unreachable it raises :class:`ASRUnavailable`, which
the transcript stage treats as "not this episode's fault": the work stays queued
and no retry budget is spent, so a laptop that sleeps overnight costs nothing but
a delay.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..config import ASRConfig
from ..logging_setup import get_logger

log = get_logger(__name__)

#: How often to report progress while decoding. Long enough that a short file
#: produces none at all, short enough that an hour-long one never looks hung.
PROGRESS_EVERY_SECONDS = 30.0


@dataclass(slots=True)
class ASRResult:
    text: str
    language: str | None = None
    duration_s: int | None = None
    #: Wall time spent decoding, measured inside the backend so it excludes
    #: waiting for the concurrency semaphore. Paired with `duration_s` it gives
    #: the realtime factor, which is the number that says whether this model and
    #: this machine are a sane pairing.
    elapsed_s: float | None = None


class ASRUnavailable(Exception):
    """The backend cannot run at all (missing dependency, unreachable endpoint)."""


class ASRBackend(Protocol):
    @property
    def name(self) -> str: ...

    async def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult: ...

    async def close(self) -> None: ...


class LocalFasterWhisperBackend:
    """faster-whisper in-process, executed off the event loop.

    The model is loaded lazily on first use (it costs hundreds of MB of RAM, and
    a deployment where nothing ever escalates should never pay that) and then
    kept for the process lifetime. Transcription runs in a worker thread; the
    caller is responsible for the concurrency limit.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        self._cfg = cfg
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return f"local:faster-whisper:{self._cfg.model}"

    async def _ensure_model(self) -> Any:
        # Double-checked locking: the fast path avoids the lock, and the slow path
        # re-reads the attribute after awaiting, since another caller may have
        # finished loading while this one waited.
        cached = self._model
        if cached is not None:
            return cached
        async with self._load_lock:
            cached = self._model
            if cached is not None:
                return cached
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - depends on install extra
                raise ASRUnavailable(
                    "faster-whisper is not installed; install the 'asr' extra "
                    "(pip install '.[asr]') or set asr.backend to 'remote'"
                ) from exc
            log.info(
                "asr.loading_model",
                model=self._cfg.model,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
            )
            self._model = await asyncio.to_thread(
                WhisperModel,
                self._cfg.model,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
            )
            log.info("asr.model_loaded", model=self._cfg.model)
            return self._model

    async def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult:
        model = await self._ensure_model()
        lang = language or self._cfg.language
        return await asyncio.to_thread(self._transcribe_sync, model, audio_path, lang)

    def _transcribe_sync(self, model: Any, audio_path: Path, language: str | None) -> ASRResult:
        started = time.monotonic()
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self._cfg.beam_size,
            vad_filter=True,
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        # `segments` is a generator: consuming it is what performs the work.
        # An hour of audio takes ten to twenty minutes on CPU, during which the
        # only prior evidence of life was the line logged before it started —
        # indistinguishable from a hung process. Progress is reported as it goes,
        # throttled by wall time so a short file stays quiet and a long one does
        # not flood the log.
        parts: list[str] = []
        last_report = started
        for segment in segments:
            if text := segment.text.strip():
                parts.append(text)
            now = time.monotonic()
            if now - last_report >= PROGRESS_EVERY_SECONDS:
                last_report = now
                elapsed = now - started
                position = float(getattr(segment, "end", 0.0) or 0.0)
                log.info(
                    "asr.progress",
                    audio_position_s=int(position),
                    audio_duration_s=int(duration) or None,
                    percent=round(100 * position / duration, 1) if duration else None,
                    elapsed_s=int(elapsed),
                    # How many seconds of audio per second of compute. The number
                    # that says whether this model and machine are a sane pairing.
                    realtime_factor=round(position / elapsed, 2) if elapsed else None,
                )

        elapsed = time.monotonic() - started
        log.info(
            "asr.complete",
            audio_duration_s=int(duration) or None,
            elapsed_s=int(elapsed),
            realtime_factor=round(duration / elapsed, 2) if elapsed and duration else None,
            chars=sum(len(p) for p in parts),
        )
        return ASRResult(
            text=" ".join(parts),
            language=getattr(info, "language", None),
            duration_s=int(duration) if duration else None,
            elapsed_s=elapsed,
        )

    async def close(self) -> None:
        self._model = None


class RemoteASRBackend:
    """Transcription on another machine, over the OpenAI audio API shape.

    ``POST {remote_url}/v1/audio/transcriptions`` with the audio as multipart
    form data — what speaches, faster-whisper-server, whisper.cpp's server and
    LocalAI all expose. The protocol is the point: moving transcription from a
    laptop to a real server later is a change of ``asr.remote_url``, nothing
    more.

    Why this exists: on the NAS that runs this in production, faster-whisper
    manages a realtime factor of **0.11** — a 68-minute episode costs about ten
    hours of CPU, so a week of episodes can never drain. The same model on an
    M2 runs near 10x realtime. The work has to happen elsewhere.

    **Every failure here is reported as :class:`ASRUnavailable`, deliberately.**
    Two reasons. The narrow one: :meth:`acquire._try_asr` catches
    ``httpx.HTTPError`` and files it as "audio download failed", which spends
    the episode's retry budget — so a sleeping laptop would quietly burn three
    attempts per episode and mark them failed. Letting an httpx error escape
    this class is therefore a bug, and the blanket translation is what prevents
    it. The broad one: this endpoint is operator-configured infrastructure, so
    its failures are operator problems, and the pipeline's designed answer to
    those is to leave the work queued rather than blame the episode.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        self._cfg = cfg
        self._base = (cfg.remote_url or "").rstrip("/")

    @property
    def name(self) -> str:
        return f"remote:{self._base}"

    async def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult:
        if not self._base:
            raise ASRUnavailable("asr.remote_url is not set")

        url = f"{self._base}/v1/audio/transcriptions"
        data: dict[str, str] = {
            "model": self._cfg.model,
            # verbose_json carries duration and detected language; servers that
            # do not implement it fall back to a plain {"text": ...}, which is
            # parsed just the same below.
            "response_format": "verbose_json",
        }
        if language:
            data["language"] = language

        started = time.monotonic()
        try:
            # The file is passed as a handle, not bytes: episodes run to
            # hundreds of megabytes and httpx streams from disk this way. This
            # process has already been OOM-killed once for holding audio in
            # memory; do not read() it here.
            with audio_path.open("rb") as handle:
                async with httpx.AsyncClient(timeout=self._cfg.remote_timeout_s) as client:
                    response = await client.post(
                        url,
                        data=data,
                        files={"file": (audio_path.name, handle, "application/octet-stream")},
                    )
        except httpx.HTTPError as exc:
            raise ASRUnavailable(f"{self.name} unreachable: {type(exc).__name__}: {exc}") from exc
        except OSError as exc:
            raise ASRUnavailable(f"could not read {audio_path}: {exc}") from exc

        elapsed = time.monotonic() - started

        if response.status_code >= 400:
            body = response.text[:300]
            raise ASRUnavailable(f"{self.name} returned HTTP {response.status_code}: {body}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ASRUnavailable(f"{self.name} returned non-JSON: {response.text[:200]}") from exc

        text = (payload or {}).get("text")
        if not isinstance(text, str):
            raise ASRUnavailable(f"{self.name} returned no text field: {str(payload)[:200]}")

        duration = payload.get("duration")
        log.info(
            "asr.complete",
            backend=self.name,
            model=self._cfg.model,
            chars=len(text),
            elapsed_s=round(elapsed, 1),
        )
        return ASRResult(
            text=text,
            language=payload.get("language") or None,
            duration_s=int(duration) if isinstance(duration, int | float) else None,
            elapsed_s=elapsed,
        )

    async def close(self) -> None:
        return None


def build_asr_backend(cfg: ASRConfig) -> ASRBackend:
    return LocalFasterWhisperBackend(cfg) if cfg.backend == "local" else RemoteASRBackend(cfg)
