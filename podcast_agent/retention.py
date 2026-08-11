"""Retention cleanup (§10.6).

Transcripts are dropped after ``transcript_days`` (summaries are kept forever),
telemetry after ``llm_call_days``, and stray audio files from interrupted ASR runs
are swept up. Only attachments and telemetry are deleted — episode documents
remain as the permanent audit trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .config import Settings
from .db import TRANSCRIPT_ATTACHMENT, Doc, Store, update_doc
from .logging_setup import bind_run, clear_run_context, get_logger
from .utils import iso, iso_now, new_run_id, utcnow

log = get_logger(__name__)

#: Audio files older than this are leftovers from an interrupted run.
ORPHAN_AUDIO_AGE = timedelta(hours=12)


@dataclass(slots=True)
class RetentionStats:
    transcripts_deleted: int = 0
    transcript_bytes_freed: int = 0
    llm_calls_deleted: int = 0
    asr_runs_deleted: int = 0
    runs_deleted: int = 0
    logs_deleted: int = 0
    orphan_audio_deleted: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "transcripts_deleted": self.transcripts_deleted,
            "transcript_bytes_freed": self.transcript_bytes_freed,
            "llm_calls_deleted": self.llm_calls_deleted,
            "asr_runs_deleted": self.asr_runs_deleted,
            "runs_deleted": self.runs_deleted,
            "logs_deleted": self.logs_deleted,
            "orphan_audio_deleted": self.orphan_audio_deleted,
            "error_count": len(self.errors),
        }


class RetentionJob:
    def __init__(self, settings: Settings, store: Store) -> None:
        self._settings = settings
        self._store = store

    async def run(self) -> dict[str, Any]:
        run_id = new_run_id()
        bind_run(run_id, job="retention")
        started = time.perf_counter()
        stats = RetentionStats()
        try:
            await self._expire_transcripts(stats)
            await self._expire_llm_calls(stats)
            await self._expire_asr_runs(stats)
            await self._expire_runs(stats)
            await self._expire_logs(stats)
            self._sweep_orphan_audio(stats)
            summary = {
                **stats.as_dict(),
                "wall_ms": int((time.perf_counter() - started) * 1000),
            }
            log.info("run.retention_summary", **summary)
            return summary
        finally:
            clear_run_context()

    async def _expire_transcripts(self, stats: RetentionStats) -> None:
        days = self._settings.retention.transcript_days
        if days == 0:
            # Kept indefinitely. Deliberately a no-op rather than a cutoff far in
            # the past: a transcript deleted here is gone, and everything built
            # over the corpus is limited by how far back it still reaches.
            log.debug("retention.transcripts_kept_indefinitely")
            return
        cutoff = iso(utcnow() - timedelta(days=days))
        # Only episodes that still hold a transcript are candidates.
        candidates = await self._store.find(
            {
                "type": "episode",
                "transcript_at": {"$lt": cutoff, "$exists": True},
            },
            limit=500,
        )
        for episode in candidates:
            attachments = episode.get("_attachments") or {}
            if TRANSCRIPT_ATTACHMENT not in attachments:
                continue
            episode_id = episode["_id"]
            size = int(attachments[TRANSCRIPT_ATTACHMENT].get("length") or 0)
            try:
                await self._store.delete_attachment(episode_id, TRANSCRIPT_ATTACHMENT)

                def _apply(doc: Doc) -> None:
                    doc["transcript_expired_at"] = iso_now()

                await update_doc(self._store, episode_id, _apply)
                stats.transcripts_deleted += 1
                stats.transcript_bytes_freed += size
                log.debug("retention.transcript_deleted", episode_id=episode_id, bytes=size)
            except Exception as exc:
                stats.errors.append(f"{episode_id}: {exc}")
                log.warning(
                    "retention.transcript_delete_failed", episode_id=episode_id, error=str(exc)
                )

    async def _expire_llm_calls(self, stats: RetentionStats) -> None:
        cutoff = iso(utcnow() - timedelta(days=self._settings.retention.llm_call_days))
        while True:
            batch = await self._store.find({"type": "llm_call", "ts": {"$lt": cutoff}}, limit=500)
            if not batch:
                return
            for doc in batch:
                try:
                    await self._store.delete(doc["_id"], doc["_rev"])
                    stats.llm_calls_deleted += 1
                except Exception as exc:
                    stats.errors.append(f"{doc['_id']}: {exc}")
            if len(batch) < 500:
                return

    async def _expire_asr_runs(self, stats: RetentionStats) -> None:
        """Transcription records age out on the same clock as model calls.

        They answer the same question — what did the machine spend — so keeping
        them for different lengths of time would make the two halves of that
        answer disagree about how far back it goes.
        """
        cutoff = iso(utcnow() - timedelta(days=self._settings.retention.llm_call_days))
        while True:
            batch = await self._store.find({"type": "asr_run", "ts": {"$lt": cutoff}}, limit=500)
            if not batch:
                return
            for doc in batch:
                try:
                    await self._store.delete(doc["_id"], doc["_rev"])
                    stats.asr_runs_deleted += 1
                except Exception as exc:
                    stats.errors.append(f"{doc['_id']}: {exc}")
            if len(batch) < 500:
                return

    async def _expire_runs(self, stats: RetentionStats) -> None:
        """Job-run records age out on the same clock as model calls.

        One document per job firing is a few hundred a day; unpruned they would
        eventually outnumber the episodes they describe.
        """
        cutoff = iso(utcnow() - timedelta(days=self._settings.retention.run_days))
        while True:
            batch = await self._store.find({"type": "run", "at": {"$lt": cutoff}}, limit=500)
            if not batch:
                return
            for doc in batch:
                try:
                    await self._store.delete(doc["_id"], doc["_rev"])
                    stats.runs_deleted += 1
                except Exception as exc:
                    stats.errors.append(f"{doc['_id']}: {exc}")
            if len(batch) < 500:
                return

    async def _expire_logs(self, stats: RetentionStats) -> None:
        """Stored warnings age out faster than anything else.

        They answer "what has been going wrong lately". A year of them is not a
        record worth keeping, it is a table nobody reads.
        """
        cutoff = iso(utcnow() - timedelta(days=self._settings.retention.log_days))
        while True:
            batch = await self._store.find({"type": "log", "at": {"$lt": cutoff}}, limit=500)
            if not batch:
                return
            for doc in batch:
                try:
                    await self._store.delete(doc["_id"], doc["_rev"])
                    stats.logs_deleted += 1
                except Exception as exc:
                    stats.errors.append(f"{doc['_id']}: {exc}")
            if len(batch) < 500:
                return

    def _sweep_orphan_audio(self, stats: RetentionStats) -> None:
        """Delete audio left behind when a run died mid-transcription."""
        audio_dir = self._settings.output.work_dir / "audio"
        if not audio_dir.is_dir():
            return
        threshold = time.time() - ORPHAN_AUDIO_AGE.total_seconds()
        for path in audio_dir.iterdir():
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < threshold:
                    path.unlink()
                    stats.orphan_audio_deleted += 1
                    log.info("retention.orphan_audio_deleted", path=str(path))
            except OSError as exc:
                stats.errors.append(f"{path}: {exc}")
