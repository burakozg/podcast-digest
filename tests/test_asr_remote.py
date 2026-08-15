"""The remote ASR backend (§14): transcription on another machine.

Production runs on a NAS with a realtime factor of 0.11 — a 68-minute episode
costs about ten hours of CPU — so the work is pushed to a machine that can
actually do it. What these tests protect is not the happy path so much as the
failure path: a laptop that sleeps must cost a delay, never an episode.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from helpers import make_settings

from podcast_agent.transcripts.asr import (
    ASRUnavailable,
    LocalFasterWhisperBackend,
    RemoteASRBackend,
    build_asr_backend,
)

REMOTE = "http://mac.lan:8000"
ENDPOINT = f"{REMOTE}/v1/audio/transcriptions"


def _cfg(tmp_path: Path, **over):
    asr = {"backend": "remote", "remote_url": REMOTE, "model": "small.en", **over}
    return make_settings(tmp_path, asr=asr).asr


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "episode.audio"
    path.write_bytes(b"not really audio, but it is bytes on disk")
    return path


class TestBackendSelection:
    def test_backend_config_picks_the_implementation(self, tmp_path: Path) -> None:
        assert isinstance(build_asr_backend(_cfg(tmp_path)), RemoteASRBackend)
        local = make_settings(tmp_path, asr={"backend": "local", "model": "small.en"}).asr
        assert isinstance(build_asr_backend(local), LocalFasterWhisperBackend)

    def test_the_destination_is_just_a_url(self, tmp_path: Path) -> None:
        """Moving from a laptop to a server must be one config value."""
        server = _cfg(tmp_path, remote_url="https://asr.example.internal")
        assert RemoteASRBackend(server).name == "remote:https://asr.example.internal"


class TestTranscribe:
    @respx.mock
    async def test_posts_the_audio_and_returns_the_text(self, tmp_path: Path, audio: Path) -> None:
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"text": "  hello world  ", "language": "en", "duration": 42.5}
            )
        )
        result = await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio, language="en")

        assert route.called
        request = route.calls.last.request
        body = request.content
        assert b"small.en" in body, "the configured model must be sent"
        assert b"episode.audio" in body, "the audio must be uploaded as a file part"
        assert result.text == "  hello world  "
        assert result.language == "en"
        assert result.duration_s == 42
        assert result.elapsed_s is not None

    @respx.mock
    async def test_accepts_a_server_that_only_returns_text(
        self, tmp_path: Path, audio: Path
    ) -> None:
        """Not every server implements verbose_json; plain {"text": ...} is enough."""
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"text": "plain"}))
        result = await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio)
        assert result.text == "plain"
        assert result.duration_s is None


class TestFailuresAreAlwaysASRUnavailable:
    """Every failure must be ASRUnavailable, or episodes pay for the operator.

    `acquire._try_asr` catches `httpx.HTTPError` and files it as a per-episode
    download failure, which spends the retry budget. An httpx error escaping
    this backend would therefore make a sleeping laptop mark episodes failed.
    """

    @respx.mock
    async def test_unreachable_host(self, tmp_path: Path, audio: Path) -> None:
        respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(ASRUnavailable) as exc:
            await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio)
        assert "unreachable" in str(exc.value)

    @respx.mock
    async def test_timeout(self, tmp_path: Path, audio: Path) -> None:
        respx.post(ENDPOINT).mock(side_effect=httpx.ReadTimeout("too slow"))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio)

    @respx.mock
    async def test_server_error(self, tmp_path: Path, audio: Path) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(503, text="model loading"))
        with pytest.raises(ASRUnavailable) as exc:
            await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio)
        assert "503" in str(exc.value)

    @respx.mock
    async def test_misconfiguration_is_an_operator_problem_not_an_episode_one(
        self, tmp_path: Path, audio: Path
    ) -> None:
        """A wrong model name is a 4xx, and must not permanently fail episodes."""
        respx.post(ENDPOINT).mock(return_value=httpx.Response(404, text="unknown model"))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio)

    @respx.mock
    async def test_garbage_response(self, tmp_path: Path, audio: Path) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text="<html>nope</html>"))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio)

    @respx.mock
    async def test_json_without_text(self, tmp_path: Path, audio: Path) -> None:
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"error": "busy"}))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio)

    async def test_missing_audio_file(self, tmp_path: Path) -> None:
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg(tmp_path)).transcribe(tmp_path / "gone.audio")

    async def test_unset_url(self, tmp_path: Path, audio: Path) -> None:
        cfg = make_settings(tmp_path, asr={"backend": "local", "model": "small.en"}).asr
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(cfg).transcribe(audio)
