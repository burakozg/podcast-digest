"""Text-to-speech behind a narrow interface — the mirror of :mod:`.transcripts.asr`.

One implementation today: ``POST {tts.base_url}/v1/audio/speech``, the OpenAI
speech API shape, which Kokoro-FastAPI, speaches, LocalAI and openedai-speech all
expose. The protocol is the point, exactly as it is for ASR: which machine does
the synthesis — and which model it runs — is a URL, so moving from a laptop to a
real server later is a config change and a restart.

The interface is declared here before it has a second implementation because the
first one did the same and paid for itself. A hosted engine (ElevenLabs and
friends) is a second class in this file, not a change anywhere else.

When the endpoint is unreachable this raises :class:`SpeechUnavailable`, which the
narration job treats as "come back later": nothing is written, nothing is marked
failed, and the next hourly run picks it up. That matters more here than for ASR,
because the machine expected to answer is a laptop that sleeps.
"""

from __future__ import annotations

import time
from typing import Protocol

import httpx

from .config import TTSConfig
from .logging_setup import get_logger

log = get_logger(__name__)

#: Content types a speech endpoint may answer with. Checked only to catch an
#: endpoint answering with a JSON error body at HTTP 200, which is a real
#: behaviour of some proxies; the exact subtype is not interesting.
_AUDIO_PREFIX = "audio/"


class SpeechUnavailable(Exception):
    """The backend cannot run at all (unset URL, unreachable endpoint)."""


class SpeechBackend(Protocol):
    @property
    def name(self) -> str: ...

    async def synthesize(self, text: str) -> bytes: ...

    async def close(self) -> None: ...


class OpenAISpeechBackend:
    """Synthesis on another machine, over the OpenAI speech API shape.

    **Every failure here is reported as :class:`SpeechUnavailable`,
    deliberately** — the same rule :class:`~.transcripts.asr.RemoteASRBackend`
    follows, for the same reason. This endpoint is operator-configured
    infrastructure, so its failures are operator problems, and the designed
    answer to those is to leave the work undone rather than record a failure
    against a digest that is perfectly fine.

    Two smaller decisions also carried over from that class:

    Its own short-lived client, not the shared :func:`~.net.build_client` one,
    whose 30-second timeout would cut off a chunk that takes a minute to render.

    It does **not** go through :class:`~.net.UrlGuard`. That guard exists to stop
    *feed-supplied* URLs reaching private addresses; this URL is operator-supplied
    and is *expected* to be a LAN address, which the guard would reject outright.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._base = (cfg.base_url or "").rstrip("/")

    @property
    def name(self) -> str:
        return f"speech:{self._base}"

    async def synthesize(self, text: str) -> bytes:
        if not self._base:
            raise SpeechUnavailable("tts.base_url is not set")

        url = f"{self._base}/v1/audio/speech"
        payload = {
            "model": self._cfg.model,
            "input": text,
            "voice": self._cfg.voice,
            "response_format": self._cfg.response_format,
            "speed": self._cfg.speed,
        }

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._cfg.timeout_s) as client:
                response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise SpeechUnavailable(
                f"{self.name} unreachable: {type(exc).__name__}: {exc}"
            ) from exc

        elapsed = time.monotonic() - started

        if response.status_code >= 400:
            raise SpeechUnavailable(
                f"{self.name} returned HTTP {response.status_code}: {response.text[:300]}"
            )

        audio = response.content
        if not audio:
            raise SpeechUnavailable(f"{self.name} returned an empty body")

        # A JSON error at HTTP 200 would otherwise be written into the file and
        # only discovered by pressing play a week later.
        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith(_AUDIO_PREFIX):
            raise SpeechUnavailable(
                f"{self.name} returned {content_type or 'no content type'}, not audio: "
                f"{response.text[:200]}"
            )

        log.info(
            "tts.chunk",
            backend=self.name,
            model=self._cfg.model,
            voice=self._cfg.voice,
            chars=len(text),
            bytes=len(audio),
            elapsed_s=round(elapsed, 1),
        )
        return audio

    async def close(self) -> None:
        return None


def build_speech_backend(cfg: TTSConfig) -> SpeechBackend:
    return OpenAISpeechBackend(cfg)
