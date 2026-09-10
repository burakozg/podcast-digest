"""The remote ASR backend (§14): transcription on another machine.

Production runs on a NAS with a realtime factor of 0.11 — a 68-minute episode
costs about ten hours of CPU — so the work is pushed to a machine that can
actually do it. What these tests protect is not the happy path so much as the
failure path: a laptop that sleeps must cost a delay, never an episode.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from helpers import make_settings

from podcast_agent.transcripts import asr as asr_module
from podcast_agent.transcripts.asr import (
    ASRUnavailable,
    LocalFasterWhisperBackend,
    RemoteASRBackend,
    build_asr_backend,
)

REMOTE = "http://mac.lan:8000"
ENDPOINT = f"{REMOTE}/v1/audio/transcriptions"


async def _async(value: Any) -> Any:
    """A ready-made awaitable, so a patched helper can be a one-line lambda."""
    return value


async def _missing_binary(*args: str, timeout_s: float) -> tuple[int, bytes, bytes]:
    raise FileNotFoundError(args[0])


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

    @respx.mock
    async def test_a_non_object_json_body_does_not_escape_as_attributeerror(
        self, tmp_path: Path, audio: Path
    ) -> None:
        """A 200 carrying a JSON list reaches `.get` unless the payload is typed."""
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=["nope"]))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg(tmp_path)).transcribe(audio)


class TestChunkingLongAudio:
    """Splitting a long episode into bounded requests (`remote_chunk_minutes`).

    The ASR host holds the decoded audio for one request, so its peak memory
    tracks the longest *request*, not the longest episode. Measured against
    speaches on an 8 GB host: ~55 MB of RSS per minute of audio, so a
    127-minute episode reached 7.4 GB and was OOM-killed mid-request — which
    the pipeline reads as an outage and defers the whole transcript stage.
    """

    @staticmethod
    def _fake_split(recorder: dict[str, object], count: int):
        async def _split(audio_path: Path, dest: Path, *, chunk_seconds: int, timeout_s: float):
            dest.mkdir(parents=True, exist_ok=True)
            recorder["dest"] = dest
            recorder["chunk_seconds"] = chunk_seconds
            made = []
            for i in range(count):
                part = dest / f"chunk_{i:04d}.wav"
                part.write_bytes(b"chunk bytes")
                made.append(part)
            return made

        return _split

    @respx.mock
    async def test_long_audio_is_split_and_the_text_rejoined(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(asr_module, "probe_duration_s", lambda p, **k: _async(7643.0))
        rec: dict[str, object] = {}
        monkeypatch.setattr(asr_module, "split_audio", self._fake_split(rec, 3))
        bodies = iter(["one", "two", "three"])
        respx.post(ENDPOINT).mock(
            side_effect=lambda request: httpx.Response(200, json={"text": next(bodies)})
        )

        result = await RemoteASRBackend(_cfg(tmp_path, remote_chunk_minutes=20)).transcribe(
            audio, language="en"
        )

        assert result.text == "one two three"
        assert rec["chunk_seconds"] == 1200
        # The source's own runtime, not the sum of what the chunks reported —
        # it is the one figure measured before any splitting happened.
        assert result.duration_s == 7643

    @respx.mock
    async def test_chunks_are_sent_one_at_a_time(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        """Sequential is the whole point: concurrent chunks put the memory back."""
        monkeypatch.setattr(asr_module, "probe_duration_s", lambda p, **k: _async(7643.0))
        monkeypatch.setattr(asr_module, "split_audio", self._fake_split({}, 4))
        in_flight = 0
        peak = 0

        async def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return httpx.Response(200, json={"text": "part"})

        respx.post(ENDPOINT).mock(side_effect=_handler)
        await RemoteASRBackend(_cfg(tmp_path, remote_chunk_minutes=20)).transcribe(audio)
        assert peak == 1

    @respx.mock
    async def test_audio_within_one_chunk_is_sent_whole(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(asr_module, "probe_duration_s", lambda p, **k: _async(600.0))

        async def _never(*a, **k):
            raise AssertionError("must not split audio shorter than one chunk")

        monkeypatch.setattr(asr_module, "split_audio", _never)
        route = respx.post(ENDPOINT).mock(
            return_value=httpx.Response(200, json={"text": "whole", "duration": 600})
        )
        result = await RemoteASRBackend(_cfg(tmp_path, remote_chunk_minutes=20)).transcribe(audio)
        assert result.text == "whole"
        assert route.call_count == 1

    @respx.mock
    async def test_chunking_is_off_by_default_and_probes_nothing(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        """Upgrading must not start spawning a subprocess per episode."""
        cfg = _cfg(tmp_path)
        assert cfg.remote_chunk_minutes == 0

        async def _never(*a, **k):
            raise AssertionError("must not probe when chunking is off")

        monkeypatch.setattr(asr_module, "probe_duration_s", _never)
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"text": "whole"}))
        assert (await RemoteASRBackend(cfg).transcribe(audio)).text == "whole"

    @respx.mock
    async def test_an_unprobeable_file_falls_back_to_one_request(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        """ "We could not tell how long it is" must not fail the episode."""
        monkeypatch.setattr(asr_module, "probe_duration_s", lambda p, **k: _async(None))

        async def _never(*a, **k):
            raise AssertionError("must not split on an unknown duration")

        monkeypatch.setattr(asr_module, "split_audio", _never)
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"text": "whole"}))
        result = await RemoteASRBackend(_cfg(tmp_path, remote_chunk_minutes=20)).transcribe(audio)
        assert result.text == "whole"

    @respx.mock
    async def test_the_chunk_files_do_not_survive_the_call(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(asr_module, "probe_duration_s", lambda p, **k: _async(7643.0))
        rec: dict[str, object] = {}
        monkeypatch.setattr(asr_module, "split_audio", self._fake_split(rec, 2))
        respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"text": "part"}))
        await RemoteASRBackend(_cfg(tmp_path, remote_chunk_minutes=20)).transcribe(audio)
        assert not Path(str(rec["dest"])).exists()

    @respx.mock
    async def test_chunk_files_are_removed_even_when_a_chunk_fails(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        """A two-hour episode leaves ~230 MB of WAV behind if this leaks."""
        monkeypatch.setattr(asr_module, "probe_duration_s", lambda p, **k: _async(7643.0))
        rec: dict[str, object] = {}
        monkeypatch.setattr(asr_module, "split_audio", self._fake_split(rec, 3))
        respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("died mid-episode"))
        with pytest.raises(ASRUnavailable):
            await RemoteASRBackend(_cfg(tmp_path, remote_chunk_minutes=20)).transcribe(audio)
        assert not Path(str(rec["dest"])).exists()

    @respx.mock
    async def test_a_failed_split_is_an_operator_problem(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(asr_module, "probe_duration_s", lambda p, **k: _async(7643.0))

        async def _boom(*a, **k):
            raise ASRUnavailable("ffmpeg could not split episode.audio: broken")

        monkeypatch.setattr(asr_module, "split_audio", _boom)
        with pytest.raises(ASRUnavailable, match="ffmpeg"):
            await RemoteASRBackend(_cfg(tmp_path, remote_chunk_minutes=20)).transcribe(audio)


class TestProbingIsBestEffort:
    async def test_a_missing_ffprobe_returns_none_rather_than_raising(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        """The only decision resting on the probe is whether to split.

        So an ffprobe that is absent (or a file it cannot read) must degrade to
        the plain single request, not fail an episode — and must not raise
        something other than ASRUnavailable out of this module either.
        """
        monkeypatch.setattr(asr_module, "_run", _missing_binary)
        assert await asr_module.probe_duration_s(audio) is None

    async def test_unparseable_ffprobe_output_returns_none(
        self, tmp_path: Path, audio: Path, monkeypatch
    ) -> None:
        async def _garbage(*args: str, timeout_s: float):
            return 0, b"N/A\n", b""

        monkeypatch.setattr(asr_module, "_run", _garbage)
        assert await asr_module.probe_duration_s(audio) is None
