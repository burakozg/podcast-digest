"""Reading a generated digest aloud, into a file beside it.

The audio lands next to the Markdown in the vault — ``podcast-digest-2026-W31.mp3``
beside ``podcast-digest-2026-W31.md`` — and the note gets an embed so Obsidian
shows a player. Nothing is served over HTTP; the vault already syncs.

Two properties do most of the work here.

**Idempotent.** If the audio exists, this returns without making a single
request. That is what lets the scheduled job run hourly instead of once a week:
the machine that synthesises is a laptop that sleeps, and a weekly fire that
found it shut would wait a week to retry.

**Newest-only by default.** ``period_key=None`` means the most recent digest and
nothing else, so an hourly job can never work backwards through a year of
archives. Older weeks are reachable only when a person asks for one by name.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..db import Doc, Store, typed_sort, update_doc
from ..logging_setup import get_logger
from ..speech import SpeechBackend
from ..utils import digest_doc_id, iso_now
from .generate import BASIS_LABELS, _build_env, summary_view
from .read import DigestUnreadable, digest_period_key, digest_runs, resolve_within
from .synthesis import as_view

log = get_logger(__name__)

#: How many digest documents to page when looking for "the newest week". The
#: list is ordered by ISO week key rather than generation time (regenerating an
#: old week must not make it the newest), so a handful has to be read to sort.
_NEWEST_SCAN = 20

#: Where the embed goes: immediately after the note's first H1. Matching the
#: heading rather than a line number survives the frontmatter changing shape.
_H1 = re.compile(r"^#\s+.*$", re.MULTILINE)

#: Paragraphs first, sentences only when a single paragraph is over the cap.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class NothingToNarrate(Exception):
    """No digest matched, or the one that did has no summarised episodes."""


@dataclass(slots=True)
class NarrationResult:
    period_key: str
    audio_path: Path | None
    skipped: bool = False
    chunks: int = 0
    bytes_written: int = 0
    elapsed_s: float = 0.0
    embedded: bool = False
    episodes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "period_key": self.period_key,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "skipped": self.skipped,
            "chunks": self.chunks,
            "bytes": self.bytes_written,
            "elapsed_s": round(self.elapsed_s, 1),
            "embedded": self.embedded,
            "episodes": len(self.episodes),
        }


def chunk_script(text: str, max_chars: int) -> list[str]:
    """Split a script into pieces the speech endpoint will accept.

    OpenAI's speech API caps ``input`` at 4096 characters and the servers that
    copy its shape follow suit, so the split is ours to make rather than the
    server's. Paragraph boundaries first, because a blank line is the pause the
    listener already expects; sentences only when one paragraph is oversized,
    and a hard slice only when a single sentence is.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in text.split("\n\n"):
        piece = paragraph.strip()
        if not piece:
            continue
        for part in _split_piece(piece, max_chars):
            if current and len(current) + 2 + len(part) > max_chars:
                flush()
            current = f"{current}\n\n{part}" if current else part
    flush()
    return chunks


def _split_piece(piece: str, max_chars: int) -> list[str]:
    """One paragraph, cut down to pieces no longer than ``max_chars``."""
    if len(piece) <= max_chars:
        return [piece]
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(piece):
        if len(sentence) > max_chars:
            if current:
                out.append(current)
                current = ""
            # A single sentence over the cap: nothing left but to cut it. Rare
            # enough to be worth saying out loud in the log.
            log.warning("tts.sentence_over_cap", chars=len(sentence), cap=max_chars)
            out.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        out.append(current)
    return out


class DigestNarrator:
    def __init__(self, settings: Settings, store: Store, backend: SpeechBackend) -> None:
        self._settings = settings
        self._store = store
        self._backend = backend
        self._env = _build_env()

    async def narrate(
        self, period_key: str | None = None, *, force: bool = False
    ) -> NarrationResult:
        doc = await self._resolve(period_key)
        key = digest_period_key(doc)
        run = digest_runs(doc)[-1]

        relative = run.get("file_path")
        if not isinstance(relative, str) or not relative:
            raise NothingToNarrate(f"{key} has no file to narrate")
        audio_path = self._audio_path(relative)

        if audio_path.exists() and not force:
            log.info("tts.skipped", period_key=key, audio=str(audio_path))
            return NarrationResult(period_key=key, audio_path=audio_path, skipped=True)

        episodes = await self._episodes(run)
        if not episodes:
            raise NothingToNarrate(f"{key} has no summarised episodes")

        script = self._script(doc, run, episodes)
        chunks = chunk_script(script, self._settings.tts.max_chars_per_request)
        log.info("tts.start", period_key=key, chars=len(script), chunks=len(chunks))

        started = time.monotonic()
        written = await self._synthesize_to(audio_path, chunks)
        elapsed = time.monotonic() - started

        result = NarrationResult(
            period_key=key,
            audio_path=audio_path,
            chunks=len(chunks),
            bytes_written=written,
            elapsed_s=elapsed,
            episodes=[e["episode_id"] for e in episodes],
        )
        result.embedded = self._embed(relative, audio_path)
        await self._record(doc["_id"], run, result)
        log.info("tts.complete", **result.as_dict())
        return result

    # --- resolution ---------------------------------------------------------

    async def _resolve(self, period_key: str | None) -> Doc:
        if period_key:
            doc = await self._store.get(digest_doc_id(period_key))
            if doc is None:
                raise NothingToNarrate(f"no digest for {period_key}")
            return doc
        docs = await self._store.find(
            {"type": "digest"}, sort=typed_sort("generated_at", "desc"), limit=_NEWEST_SCAN
        )
        if not docs:
            raise NothingToNarrate("no digests have been generated yet")
        # By period, not by generation time — regenerating an old week gives it
        # the newest `generated_at`, and narrating *that* is exactly the walk
        # backwards this job must never do.
        return max(docs, key=digest_period_key)

    def _audio_path(self, relative: str) -> Path:
        """The audio file for a digest, beside it and named after it.

        Derived from the run's own `file_path`, so a regenerated week that
        landed as `-r2.md` gets `-r2.mp3` rather than colliding with the first.

        `resolve_within` because `file_path` comes off a document, and a
        document is data: without the guard, a crafted one would place a write
        anywhere the service can reach.
        """
        suffix = f".{self._settings.tts.response_format}"
        target = Path(relative).with_suffix(suffix)
        return resolve_within(self._settings.output.digest_dir, str(target))

    async def _episodes(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        """The run's summarised episodes, as template views.

        Built from the ids the run recorded rather than by re-collecting the
        period, so the audio says what the file says. A week re-summarised since
        would otherwise be narrated with material its own Markdown never
        mentioned.
        """
        views: list[dict[str, Any]] = []
        for episode_id in run.get("episode_ids") or []:
            doc = await self._store.get(str(episode_id))
            if doc is None or not (doc.get("tier1") or {}).get("summary_md"):
                continue
            views.append(summary_view(self._settings, doc, BASIS_LABELS))
        views.sort(key=lambda v: (-int(v["score"]), str(v["podcast_name"])))
        return views

    def _script(self, doc: Doc, run: dict[str, Any], episodes: list[dict[str, Any]]) -> str:
        threshold = self._settings.pipeline.top_pick_threshold
        period_key = digest_period_key(doc)
        stats = run.get("stats") or {}
        return str(
            self._env.get_template("digest.speech.txt.j2").render(
                week_number=int(period_key.split("-W")[-1]),
                year=int(period_key.split("-W")[0]),
                stats={
                    "scanned": int(stats.get("scanned") or len(episodes)),
                    "summarized": len(episodes),
                },
                # Read back off the document rather than recomputed: it is a
                # model call, and the file already quotes this exact version.
                synthesis=doc.get("synthesis") or as_view(None),
                top_picks=[e for e in episodes if int(e["score"]) >= threshold],
                also_relevant=[e for e in episodes if int(e["score"]) < threshold],
            )
        )

    # --- synthesis ----------------------------------------------------------

    async def _synthesize_to(self, audio_path: Path, chunks: list[str]) -> int:
        """Render every chunk into ``audio_path``, atomically.

        Appended to a temp file rather than accumulated in memory: this process
        has already been OOM-killed once for holding audio in RAM (see
        `RemoteASRBackend.transcribe`), and half an hour of speech is not the
        place to relearn it. The temp file is removed on any failure, so a
        `SpeechUnavailable` halfway through leaves nothing to play by mistake.
        """
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = audio_path.with_name(f".{audio_path.name}.tmp")
        written = 0
        try:
            with tmp.open("wb") as handle:
                for index, chunk in enumerate(chunks, start=1):
                    audio = await self._backend.synthesize(chunk)
                    handle.write(audio)
                    written += len(audio)
                    log.debug("tts.progress", chunk=index, of=len(chunks), bytes=written)
            os.replace(tmp, audio_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return written

    # --- the note -----------------------------------------------------------

    def _embed(self, relative: str, audio_path: Path) -> bool:
        """Put an Obsidian embed under the note's H1. Best effort, by design.

        The vault belongs to the reader, who may have edited, moved or pruned
        the note. None of that is a reason to fail a narration that already
        succeeded, so every unhappy path here logs and returns False.
        """
        embed = f"![[{audio_path.name}]]"
        try:
            note = resolve_within(self._settings.output.digest_dir, relative)
            text = note.read_text(encoding="utf-8")
        except (DigestUnreadable, OSError) as exc:
            log.warning("tts.embed_skipped", reason=str(exc))
            return False
        if embed in text:
            return False
        match = _H1.search(text)
        if match is None:
            log.warning("tts.embed_skipped", reason="no heading to anchor to", note=str(note))
            return False
        end = match.end()
        updated = f"{text[:end]}\n\n{embed}{text[end:]}"
        try:
            note.write_text(updated, encoding="utf-8")
        except OSError as exc:
            log.warning("tts.embed_skipped", reason=str(exc))
            return False
        return True

    async def _record(self, digest_id: str, run: dict[str, Any], result: NarrationResult) -> None:
        """Note the narration on the run it belongs to.

        Matched by `file_path` rather than by index: `digest_runs` synthesises a
        run for documents written before runs were recorded, and that one has no
        slot in the stored list to write back into.
        """
        target = run.get("file_path")
        narration = {
            "audio_path": str(result.audio_path),
            "bytes": result.bytes_written,
            "chunks": result.chunks,
            "voice": self._settings.tts.voice,
            "model": self._settings.tts.model,
            "elapsed_s": round(result.elapsed_s, 1),
            "at": iso_now(),
        }

        def _apply(doc: Doc) -> None:
            for stored in doc.get("runs") or []:
                if stored.get("file_path") == target:
                    stored["narration"] = narration
                    return
            doc["narration"] = narration

        try:
            await update_doc(self._store, digest_id, _apply)
        except Exception as exc:  # telemetry must never fail the job
            log.warning("tts.not_recorded", digest_id=digest_id, error=str(exc))
