"""Transcript stage: status handling, retry budget and storage around acquisition.

Failure policy (§4): after ``pipeline.max_retries`` attempts the episode moves to
TRANSCRIPT_FAILED, which is not a dead end — Tier-1 then runs on the description
alone and the digest labels the entry honestly as ``description_only``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from ..config import Settings
from ..db import Doc, Store, save_transcript
from ..episodes import attempt_count, bump_attempt, transition
from ..logging_setup import get_logger
from ..state import EpisodeStatus
from ..utils import iso_now
from .acquire import TranscriptAcquirer, TranscriptUnavailable
from .asr import ASRUnavailable

log = get_logger(__name__)

#: How many times an episode may be *in flight* when the process dies before it
#: is retired.
#:
#: Separate from ``pipeline.max_retries``, and deliberately its own counter,
#: because a crash does not identify a culprit. Only one episode is transcribed
#: at a time, but audio for others downloads alongside it, so whichever episode
#: happens to be in flight when the container hits its memory limit wears the
#: blame — one-minute episodes were retired for a ceiling they could not
#: possibly have reached. A crash is therefore counted apart from a genuine
#: failure, and only ever bounds the loop; it is not evidence about the episode.
CRASH_BUDGET: Final = 3


class TranscriptStage:
    def __init__(self, settings: Settings, store: Store, acquirer: TranscriptAcquirer) -> None:
        self._settings = settings
        self._store = store
        self._acquirer = acquirer

    async def process(self, episode: Doc, *, allow_asr: bool = True) -> EpisodeStatus:
        episode_id = episode["_id"]
        attempts_before = attempt_count(episode, "transcript")
        crashes_before = attempt_count(episode, "transcript_crash")

        # Crash budget spent — retire it without touching the audio again.
        if crashes_before >= CRASH_BUDGET:
            return await self._give_up(
                episode_id,
                crashes_before,
                f"the process died {crashes_before} times with this episode in flight; "
                "retired without a verdict on the episode itself — retry it once the "
                "cause (usually the container's memory limit) is addressed",
            )

        # Mark the episode in flight BEFORE doing anything that can kill the process.
        #
        # Nothing is recorded when the process is killed rather than raising:
        # SIGKILL — an OOM during ASR, say — unwinds no handler. A counter
        # bumped only in an `except` clause stayed at zero through ~40 real
        # attempts, `max_retries` never engaged, and because the episode sorted
        # first it was retried on every run forever. Transcripts run before
        # summarising, so nothing downstream executed either: one episode froze
        # the whole pipeline for five days.
        #
        # So the marker goes down first and is *released* on every clean exit
        # below — success, failure, or outage. Whatever is left after that was
        # a crash, and only a crash.
        def _claim(doc: Doc) -> None:
            bump_attempt(doc, "transcript_crash")

        await transition(self._store, episode_id, EpisodeStatus.AWAITING_TRANSCRIPT, mutate=_claim)

        def _release(doc: Doc) -> None:
            doc.setdefault("attempts", {})["transcript_crash"] = crashes_before

        try:
            result = await self._acquirer.acquire(episode, allow_asr=allow_asr)
        except ASRUnavailable as exc:
            # The backend itself is broken (missing dependency, dead endpoint).
            # Don't burn the episode's retry budget for an operator problem.
            await transition(
                self._store, episode_id, EpisodeStatus.AWAITING_TRANSCRIPT, mutate=_release
            )
            log.error("transcript.backend_unavailable", episode_id=episode_id, error=str(exc))
            raise
        except TranscriptUnavailable as exc:
            return await self._handle_failure(
                episode, attempts_before, str(exc), retryable=exc.retryable, release=_release
            )

        size = await save_transcript(self._store, episode_id, result.text)

        def _apply(doc: Doc) -> None:
            _release(doc)
            bump_attempt(doc, "transcript")
            doc["transcript_source"] = result.source
            doc["transcript_chars"] = len(result.text)
            doc["transcript_bytes_gz"] = size
            doc["transcript_at"] = iso_now()
            if result.detected_language:
                doc["transcript_language"] = result.detected_language
            # ASR reports true audio length; feeds often lie or omit it.
            if result.duration_s and not doc.get("duration_s"):
                doc["duration_s"] = result.duration_s

        await transition(self._store, episode_id, EpisodeStatus.TRANSCRIBED, mutate=_apply)
        log.info(
            "transcript.stored",
            episode_id=episode_id,
            source=result.source,
            chars=len(result.text),
            gz_bytes=size,
        )
        return EpisodeStatus.TRANSCRIBED

    async def _give_up(self, episode_id: str, attempts: int, error: str) -> EpisodeStatus:
        """Retire an episode whose retry budget went with a crashed process."""

        def _apply(doc: Doc) -> None:
            doc["last_error"] = {
                "stage": "transcript",
                "type": "AttemptsExhausted",
                "message": error[:1000],
                "at": iso_now(),
            }

        await transition(self._store, episode_id, EpisodeStatus.TRANSCRIPT_FAILED, mutate=_apply)
        log.warning(
            "transcript.failed_permanently",
            episode_id=episode_id,
            attempts=attempts,
            retryable=False,
            error=error[:400],
        )
        return EpisodeStatus.TRANSCRIPT_FAILED

    async def _handle_failure(
        self,
        episode: Doc,
        attempts_before: int,
        error: str,
        *,
        retryable: bool = True,
        release: Callable[[Doc], None] | None = None,
    ) -> EpisodeStatus:
        episode_id = episode["_id"]
        attempts = attempts_before + 1
        # A failure with nothing left to try does not get retries. Three passes
        # at "this podcast publishes no transcript and transcription is off"
        # produce three identical warnings and delay the description-only
        # summary the episode is going to get anyway.
        exhausted = not retryable or attempts >= self._settings.pipeline.max_retries

        def _apply(doc: Doc) -> None:
            # This episode reported for itself, so the in-flight marker goes.
            if release is not None:
                release(doc)
            bump_attempt(doc, "transcript")
            doc["last_error"] = {
                "stage": "transcript",
                "type": "TranscriptUnavailable",
                "message": error[:1000],
                "at": iso_now(),
            }

        if exhausted:
            await transition(
                self._store, episode_id, EpisodeStatus.TRANSCRIPT_FAILED, mutate=_apply
            )
            log.warning(
                "transcript.failed_permanently",
                episode_id=episode_id,
                attempts=attempts,
                retryable=retryable,
                error=error[:400],
            )
            return EpisodeStatus.TRANSCRIPT_FAILED

        # Stay queued; the next scheduler run retries with backoff-by-schedule.
        await transition(self._store, episode_id, EpisodeStatus.AWAITING_TRANSCRIPT, mutate=_apply)
        log.warning(
            "transcript.attempt_failed",
            episode_id=episode_id,
            attempts=attempts,
            max_retries=self._settings.pipeline.max_retries,
            error=error[:400],
        )
        return EpisodeStatus.AWAITING_TRANSCRIPT
