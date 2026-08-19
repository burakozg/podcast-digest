"""Reading the weekly digest aloud (§5): the backend, the script, and the job."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from helpers import make_episode, make_settings

from podcast_agent.config import TTSConfig
from podcast_agent.db import MemoryStore
from podcast_agent.digest.narrate import (
    DigestNarrator,
    NothingToNarrate,
    chunk_script,
)
from podcast_agent.main import build_app
from podcast_agent.speech import OpenAISpeechBackend, SpeechUnavailable, build_speech_backend
from podcast_agent.utils import digest_doc_id, episode_doc_id, iso_now

SPEECH_URL = "http://mac.lan:8880/v1/audio/speech"


def _cfg(**overrides: Any) -> TTSConfig:
    return TTSConfig(**{"enabled": True, "base_url": "http://mac.lan:8880", **overrides})


# --- the backend -------------------------------------------------------------


class TestBackendSelection:
    def test_the_openai_shape_is_what_gets_built(self) -> None:
        backend = build_speech_backend(_cfg())
        assert isinstance(backend, OpenAISpeechBackend)
        assert backend.name == "speech:http://mac.lan:8880"

    def test_a_trailing_slash_does_not_double_up(self) -> None:
        backend = build_speech_backend(_cfg(base_url="http://mac.lan:8880/"))
        assert backend.name == "speech:http://mac.lan:8880"


class TestSynthesize:
    @respx.mock
    async def test_the_request_carries_every_voice_setting(self) -> None:
        route = respx.post(SPEECH_URL).mock(
            return_value=httpx.Response(
                200, content=b"ID3audio", headers={"content-type": "audio/mpeg"}
            )
        )
        backend = build_speech_backend(_cfg(voice="bf_emma", speed=1.25, model="kokoro-v1"))
        assert await backend.synthesize("Hello.") == b"ID3audio"

        body = json.loads(route.calls.last.request.content)
        assert body == {
            "model": "kokoro-v1",
            "input": "Hello.",
            "voice": "bf_emma",
            "response_format": "mp3",
            "speed": 1.25,
        }


class TestFailuresAreAlwaysSpeechUnavailable:
    """Every failure is reported as one exception type, deliberately.

    The narration job's answer to an unreachable endpoint is to leave the digest
    un-narrated and try again on the next hourly fire — the machine expected to
    answer is a laptop that sleeps. Letting an httpx error escape this class
    would turn a shut lid into a stack trace.
    """

    async def test_an_unset_url_is_not_a_crash(self) -> None:
        backend = build_speech_backend(TTSConfig())
        with pytest.raises(SpeechUnavailable, match="base_url"):
            await backend.synthesize("Hello.")

    @respx.mock
    async def test_unreachable(self) -> None:
        respx.post(SPEECH_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(SpeechUnavailable, match="unreachable"):
            await build_speech_backend(_cfg()).synthesize("Hello.")

    @respx.mock
    async def test_timeout(self) -> None:
        respx.post(SPEECH_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(SpeechUnavailable, match="unreachable"):
            await build_speech_backend(_cfg()).synthesize("Hello.")

    @respx.mock
    async def test_server_error(self) -> None:
        respx.post(SPEECH_URL).mock(return_value=httpx.Response(503, text="model loading"))
        with pytest.raises(SpeechUnavailable, match="HTTP 503"):
            await build_speech_backend(_cfg()).synthesize("Hello.")

    @respx.mock
    async def test_an_empty_body_is_not_silently_written(self) -> None:
        respx.post(SPEECH_URL).mock(return_value=httpx.Response(200, content=b""))
        with pytest.raises(SpeechUnavailable, match="empty"):
            await build_speech_backend(_cfg()).synthesize("Hello.")

    @respx.mock
    async def test_json_at_http_200_is_caught(self) -> None:
        """Some proxies answer errors with 200 and a JSON body. Written into an
        .mp3 that is only discovered by pressing play a week later."""
        respx.post(SPEECH_URL).mock(
            return_value=httpx.Response(200, json={"error": "no such voice"})
        )
        with pytest.raises(SpeechUnavailable, match="not audio"):
            await build_speech_backend(_cfg()).synthesize("Hello.")


# --- chunking ----------------------------------------------------------------


class TestChunking:
    def test_a_short_script_is_one_request(self) -> None:
        assert chunk_script("One paragraph.\n\nAnd another.", 3000) == [
            "One paragraph.\n\nAnd another."
        ]

    def test_every_chunk_stays_under_the_cap(self) -> None:
        script = "\n\n".join(f"Paragraph number {i} says something." * 3 for i in range(60))
        chunks = chunk_script(script, 500)
        assert len(chunks) > 1
        assert all(len(c) <= 500 for c in chunks)

    def test_nothing_is_lost_in_the_split(self) -> None:
        script = "\n\n".join(f"Sentence {i}. More words here." for i in range(40))
        rejoined = " ".join(chunk_script(script, 200)).split()
        assert rejoined == script.split()

    def test_an_oversized_paragraph_splits_on_sentences(self) -> None:
        script = " ".join(f"Sentence number {i} is here." for i in range(80))
        chunks = chunk_script(script, 300)
        assert all(len(c) <= 300 for c in chunks)
        # Split at sentence ends, so no chunk begins mid-sentence.
        assert all(c[0].isupper() for c in chunks)

    def test_one_enormous_sentence_is_cut_rather_than_dropped(self) -> None:
        script = "word " * 500
        chunks = chunk_script(script, 100)
        assert all(len(c) <= 100 for c in chunks)
        assert "".join(chunks).replace(" ", "") == script.replace(" ", "")

    def test_a_zero_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            chunk_script("Hello.", 0)


# --- the job -----------------------------------------------------------------


class FakeSpeech:
    """Duck-typed, like the ASR fakes: the Protocol is structural."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._fail = fail

    @property
    def name(self) -> str:
        return "fake-speech"

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        if self._fail:
            raise self._fail
        return f"[{len(self.calls)}]".encode()

    async def close(self) -> None:
        return None


class TestNarration:
    def _settings(self, tmp_path: Path) -> Any:
        return make_settings(tmp_path, tts={"enabled": True, "base_url": "http://mac.lan:8880"})

    def _seed(
        self,
        store: MemoryStore,
        tmp_path: Path,
        week: str = "2026-W31",
        *,
        episodes: int = 2,
        body: str | None = None,
    ) -> Path:
        """A digest document plus the Markdown file it names."""
        settings = self._settings(tmp_path)
        relative = f"{week[:4]}/podcast-digest-{week}.md"
        note = Path(settings.output.digest_dir) / relative
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            body
            if body is not None
            else f"---\nweek: {week}\n---\n\n# Podcast Digest — Week {week}\n\n*stats line*\n",
            encoding="utf-8",
        )

        episode_ids = []
        for index in range(episodes):
            guid = f"{week}-{index}"
            store.seed(
                make_episode(
                    guid=guid,
                    title=f"Ransomware crews are hiring {index}",
                    tier1={
                        # First episode a top pick, the rest not.
                        "relevance_score": 9 if index == 0 else 5,
                        "summary_basis": "transcript",
                        "why_it_matters": f"Reason number {index} to care.",
                        "summary_md": (
                            f"Affiliate churn hit **40%** in week {index}. "
                            "Read [the report](https://example.com/report)."
                        ),
                        "key_takeaways": [f"Takeaway {index} alpha", f"Takeaway {index} beta"],
                        "entities": ["LockBit"],
                        "matched_interests": [],
                    },
                )
            )
            episode_ids.append(episode_doc_id("test-show", guid))

        store.seed(
            {
                "_id": digest_doc_id(week),
                "type": "digest",
                "period": {"from": f"{week[:4]}-07-24T00:00:00+00:00", "to": f"{week[:4]}-07-31"},
                "file_path": relative,
                "episode_ids": episode_ids,
                "runs": [
                    {
                        "file_path": relative,
                        "period": {},
                        "episode_ids": episode_ids,
                        "stats": {"scanned": 41, "summarized": episodes},
                        "generated_at": iso_now(),
                    }
                ],
                "synthesis": {
                    "themes": [
                        {
                            "title": "Affiliates are professionalising",
                            "summary": "Crews now run **formal** pipelines.",
                            "shows": ["Risky Business"],
                        }
                    ],
                    "disagreements": ["They split on attribution."],
                    "whats_new": [],
                },
                "marking_complete": True,
                "generated_at": iso_now(),
            }
        )
        return note

    def _narrator(self, tmp_path: Path, store: MemoryStore, speech: FakeSpeech) -> DigestNarrator:
        return DigestNarrator(self._settings(tmp_path), store, speech)

    async def test_the_audio_lands_beside_the_markdown(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        note = self._seed(store, tmp_path)
        speech = FakeSpeech()
        result = await self._narrator(tmp_path, store, speech).narrate()

        assert result.audio_path == note.with_suffix(".mp3")
        assert result.audio_path.exists()
        assert result.audio_path.read_bytes() == b"".join(
            f"[{i}]".encode() for i in range(1, len(speech.calls) + 1)
        )
        assert result.bytes_written == result.audio_path.stat().st_size

    async def test_the_script_is_speakable(self, tmp_path: Path, store: MemoryStore) -> None:
        self._seed(store, tmp_path)
        speech = FakeSpeech()
        await self._narrator(tmp_path, store, speech).narrate()
        script = "\n\n".join(speech.calls)

        # The substance is all there.
        assert "Affiliates are professionalising" in script
        assert "Ransomware crews are hiring 0" in script
        assert "Reason number 0 to care." in script
        assert "Affiliate churn hit 40% in week 0." in script
        assert "Takeaway 0 alpha" in script
        assert "They split on attribution." in script

        # None of what a synthesiser would read as punctuation or spell out.
        assert "**" not in script
        assert "[[" not in script
        assert "http" not in script
        assert "/10" not in script
        assert "Everything else scanned" not in script

    async def test_link_text_survives_but_the_url_does_not(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        self._seed(store, tmp_path)
        speech = FakeSpeech()
        await self._narrator(tmp_path, store, speech).narrate()
        script = "\n\n".join(speech.calls)
        assert "Read the report." in script
        assert "example.com" not in script

    async def test_top_picks_and_also_relevant_are_separated(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        self._seed(store, tmp_path, episodes=2)
        speech = FakeSpeech()
        await self._narrator(tmp_path, store, speech).narrate()
        script = "\n\n".join(speech.calls)
        assert script.index("Top picks.") < script.index("Ransomware crews are hiring 0")
        assert script.index("Also relevant.") < script.index("Ransomware crews are hiring 1")
        # Only the top pick gets its full summary read; the rest are takeaways.
        assert "Affiliate churn hit 40% in week 1." not in script

    async def test_the_note_gets_a_player(self, tmp_path: Path, store: MemoryStore) -> None:
        note = self._seed(store, tmp_path)
        result = await self._narrator(tmp_path, store, FakeSpeech()).narrate()
        text = note.read_text(encoding="utf-8")
        assert result.embedded is True
        assert "![[podcast-digest-2026-W31.mp3]]" in text
        # Under the heading, above the body — not appended to the end.
        assert text.index("# Podcast Digest") < text.index("![[")
        assert text.index("![[") < text.index("*stats line*")

    async def test_a_rerun_does_not_stack_up_players(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        note = self._seed(store, tmp_path)
        narrator = self._narrator(tmp_path, store, FakeSpeech())
        await narrator.narrate()
        await narrator.narrate(force=True)
        assert note.read_text(encoding="utf-8").count("![[") == 1

    async def test_a_note_with_no_heading_still_gets_its_audio(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """The vault belongs to the reader. An edited note is not a failure."""
        self._seed(store, tmp_path, body="just some text the user rewrote\n")
        result = await self._narrator(tmp_path, store, FakeSpeech()).narrate()
        assert result.embedded is False
        assert result.audio_path is not None
        assert result.audio_path.exists()

    async def test_a_second_call_is_free(self, tmp_path: Path, store: MemoryStore) -> None:
        self._seed(store, tmp_path)
        speech = FakeSpeech()
        narrator = self._narrator(tmp_path, store, speech)
        await narrator.narrate()
        first = len(speech.calls)

        result = await narrator.narrate()
        assert result.skipped is True
        assert len(speech.calls) == first, "an idempotent job must not re-synthesise"

    async def test_force_redoes_it(self, tmp_path: Path, store: MemoryStore) -> None:
        self._seed(store, tmp_path)
        speech = FakeSpeech()
        narrator = self._narrator(tmp_path, store, speech)
        await narrator.narrate()
        first = len(speech.calls)
        result = await narrator.narrate(force=True)
        assert result.skipped is False
        assert len(speech.calls) > first

    async def test_a_failure_leaves_nothing_to_play(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        note = self._seed(store, tmp_path)
        speech = FakeSpeech(fail=SpeechUnavailable("laptop asleep"))
        with pytest.raises(SpeechUnavailable):
            await self._narrator(tmp_path, store, speech).narrate()

        assert not note.with_suffix(".mp3").exists()
        assert list(note.parent.glob(".*.tmp")) == [], "the temp file must be cleaned up"

    async def test_the_narration_is_recorded_on_the_run(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        self._seed(store, tmp_path)
        await self._narrator(tmp_path, store, FakeSpeech()).narrate()
        doc = await store.get(digest_doc_id("2026-W31"))
        assert doc is not None
        narration = doc["runs"][0]["narration"]
        assert narration["voice"] == "af_heart"
        assert narration["chunks"] >= 1
        assert narration["bytes"] > 0

    async def test_a_week_with_no_summaries_says_so(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        self._seed(store, tmp_path, episodes=0)
        with pytest.raises(NothingToNarrate, match="no summarised episodes"):
            await self._narrator(tmp_path, store, FakeSpeech()).narrate()

    async def test_an_unknown_week_is_refused(self, tmp_path: Path, store: MemoryStore) -> None:
        self._seed(store, tmp_path)
        with pytest.raises(NothingToNarrate, match="no digest for"):
            await self._narrator(tmp_path, store, FakeSpeech()).narrate("1999-W01")

    async def test_nothing_generated_yet(self, tmp_path: Path, store: MemoryStore) -> None:
        with pytest.raises(NothingToNarrate, match="no digests"):
            await self._narrator(tmp_path, store, FakeSpeech()).narrate()


class TestHistoryIsLeftAlone:
    """The scheduled job runs hourly. It must never work backwards.

    Narrating a year of archives would be hours of synthesis for files that have
    already been read — so the unnamed call reaches exactly one week, the newest,
    and everything older waits for a person to ask.
    """

    async def test_only_the_newest_week_is_narrated(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        job = TestNarration()
        notes = [job._seed(store, tmp_path, week) for week in ("2026-W29", "2026-W30", "2026-W31")]
        speech = FakeSpeech()

        result = await job._narrator(tmp_path, store, speech).narrate()

        assert result.period_key == "2026-W31"
        assert notes[2].with_suffix(".mp3").exists()
        assert not notes[0].with_suffix(".mp3").exists()
        assert not notes[1].with_suffix(".mp3").exists()

    async def test_a_named_week_is_the_only_way_back(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        job = TestNarration()
        notes = [job._seed(store, tmp_path, week) for week in ("2026-W29", "2026-W31")]
        narrator = job._narrator(tmp_path, store, FakeSpeech())

        await narrator.narrate("2026-W29")

        assert notes[0].with_suffix(".mp3").exists()
        assert not notes[1].with_suffix(".mp3").exists()

    async def test_regenerating_an_old_week_does_not_make_it_the_newest(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """`generated_at` orders by when it ran; the archive orders by week.

        Re-running week 29 today gives it the freshest timestamp, and picking the
        newest by that is exactly the walk backwards this must not do.
        """
        job = TestNarration()
        old_note = job._seed(store, tmp_path, "2026-W29")
        new_note = job._seed(store, tmp_path, "2026-W31")
        doc = await store.get(digest_doc_id("2026-W29"))
        assert doc is not None
        store.seed({**doc, "generated_at": "2099-01-01T00:00:00+00:00"})

        result = await job._narrator(tmp_path, store, FakeSpeech()).narrate()

        assert result.period_key == "2026-W31"
        assert new_note.with_suffix(".mp3").exists()
        assert not old_note.with_suffix(".mp3").exists()


class TestTheJobIsOnlyRegisteredWhenItCanRun:
    """A disabled feature must cost nothing — not even a wakeup an hour."""

    def _jobs(self, tmp_path: Path, store: MemoryStore, **tts: Any) -> set[str]:
        from helpers import FakeLLM

        settings = make_settings(tmp_path, tts=tts) if tts else make_settings(tmp_path)
        with TestClient(build_app(settings, store=store, llm=FakeLLM())) as client:
            return {j.id for j in client.app.state.scheduler.get_jobs()}

    def test_registered_when_speech_is_on(self, tmp_path: Path, store: MemoryStore) -> None:
        jobs = self._jobs(tmp_path, store, enabled=True, base_url="http://mac.lan:8880")
        assert "digest_narrate" in jobs

    def test_absent_when_speech_is_off(self, tmp_path: Path, store: MemoryStore) -> None:
        jobs = self._jobs(tmp_path, store)
        assert "digest_narrate" not in jobs
        # The rest of the schedule is untouched by the feature being off.
        assert {"ingest", "pipeline", "digest_weekly"} <= jobs
