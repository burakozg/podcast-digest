"""ASR backends behind a narrow interface (§3, §10.4).

v1 implements ``local`` (faster-whisper in-process). The ``remote`` backend exists
as a declared interface only, so pointing ASR at a beefier box later is a config
change plus one class, with no pipeline changes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
    """Placeholder for a remote Whisper endpoint (§14 — deferred in v1)."""

    def __init__(self, cfg: ASRConfig) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return f"remote:{self._cfg.remote_url}"

    async def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult:
        raise ASRUnavailable(
            "the remote ASR backend is declared but not implemented in v1; "
            "set asr.backend to 'local'"
        )

    async def close(self) -> None:
        return None


def build_asr_backend(cfg: ASRConfig) -> ASRBackend:
    return LocalFasterWhisperBackend(cfg) if cfg.backend == "local" else RemoteASRBackend(cfg)
