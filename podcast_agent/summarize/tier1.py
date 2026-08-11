"""Stage 4 — Tier-1 summarise and score (§4).

Single call when the input fits the budget; map-reduce over paragraph-boundary
chunks when it does not. ``summary_basis`` is set here from what the pipeline
actually had available, never from model output (see models.py).
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..db import Doc, Store, load_transcript
from ..episodes import bump_attempt, call_meta_dict, filter_interest_keys, transition
from ..llm.base import StructuredLLM
from ..llm.prompts import format_interest_profile, load_prompt
from ..logging_setup import get_logger
from ..models import CallMeta, ChunkBullets, SummaryBasis, Tier1Result
from ..notify import Notifier
from ..state import EpisodeStatus
from ..utils import describe_age, episode_age_days, estimate_tokens, format_duration, iso_now
from .chunking import chunk_transcript, needs_map_reduce, truncate_to_tokens

log = get_logger(__name__)

#: Per-prompt versions. tier1 and tier1_reduce moved to v2 for archive-aware
#: framing (A2); tier1_map is unchanged, so it keeps v1 and its provenance.
PROMPT_VERSIONS = {"tier1": "v2", "tier1_map": "v1", "tier1_reduce": "v2"}

#: Cap on map steps for one episode. At 6k tokens/chunk this covers ~8 hours of
#: speech; beyond that the tail is dropped rather than spending unbounded cost.
MAX_CHUNKS = 20


class Tier1Stage:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        llm: StructuredLLM,
        notifier: Notifier | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._llm = llm
        self._notifier = notifier
        self._single = load_prompt("tier1", PROMPT_VERSIONS["tier1"])
        self._map = load_prompt("tier1_map", PROMPT_VERSIONS["tier1_map"])
        self._reduce = load_prompt("tier1_reduce", PROMPT_VERSIONS["tier1_reduce"])
        self._profile_text = format_interest_profile(settings.interest_profile)
        self._valid_keys = {i.key for i in settings.interest_profile}
        self._profile_version = settings.interest_profile_version()

    async def summarize(self, episode: Doc, *, threshold: int | None = None) -> EpisodeStatus:
        """Normal pipeline path: TRANSCRIBED/TRANSCRIPT_FAILED -> classified."""
        result, block, target = await self._score(episode, threshold=threshold)

        def _apply(doc: Doc) -> None:
            doc["tier1"] = block
            bump_attempt(doc, "tier1")

        # Store the result and classify in one write: SUMMARIZED is an internal
        # waypoint, so going straight to the classified status keeps the doc
        # consistent if the process dies here.
        await transition(self._store, episode["_id"], EpisodeStatus.SUMMARIZED, mutate=_apply)
        await transition(self._store, episode["_id"], target)
        self._log_scored("tier1.summarized", episode, result, block, target)
        # Notify only after the summary is durably stored, so a dead notification
        # endpoint can never cost us the work (E4).
        await self._maybe_notify(episode, block)
        return target

    async def _maybe_notify(self, episode: Doc, block: dict[str, Any]) -> None:
        if self._notifier is None:
            return
        await self._notifier.notify_episode({**episode, "tier1": block})

    async def rescore(self, episode: Doc) -> EpisodeStatus:
        """Re-run scoring against the current interest profile (C2).

        Reuses the stored transcript, so this costs Tier-1 tokens only — no
        re-fetching and no re-transcription. Applies to episodes that have not
        yet been published: rewriting the score of an episode already written
        into a digest on disk would leave the database disagreeing with the file.
        """
        episode_id = episode["_id"]
        previous = (episode.get("tier1") or {}).get("relevance_score")
        result, block, target = await self._score(episode)
        block["rescored_at"] = iso_now()
        block["previous_score"] = previous

        def _apply(doc: Doc) -> None:
            doc["tier1"] = block

        # One write: the new score and its classification land together.
        await transition(self._store, episode_id, target, mutate=_apply)
        self._log_scored("tier1.rescored", episode, result, block, target, previous=previous)
        return target

    async def _score(
        self, episode: Doc, *, threshold: int | None = None
    ) -> tuple[Tier1Result, dict[str, Any], EpisodeStatus]:
        """Run Tier-1 and build the stored block. No status changes here."""
        episode_id = episode["_id"]
        content, basis = await self._resolve_content(episode)
        if not content.strip():
            raise ValueError(
                f"{episode_id}: nothing to summarise (empty transcript and description)"
            )

        pipeline = self._settings.pipeline
        if basis != "description_only" and needs_map_reduce(content, pipeline.max_input_tokens):
            result, metas, chunk_count = await self._map_reduce(episode, content)
        else:
            result, meta = await self._single_call(episode, content, basis)
            metas, chunk_count = [meta], 0

        matched = filter_interest_keys(result.matched_interests, self._valid_keys)
        effective_threshold = pipeline.digest_threshold if threshold is None else threshold
        target = (
            EpisodeStatus.READY_FOR_DIGEST
            if result.relevance_score >= effective_threshold
            else EpisodeStatus.SCORED_LOW
        )
        block: dict[str, Any] = {
            "relevance_score": result.relevance_score,
            "matched_interests": matched,
            "why_it_matters": result.why_it_matters,
            "summary_md": result.summary_md,
            "key_takeaways": result.key_takeaways,
            "entities": result.entities,
            "listen_anyway": result.listen_anyway,
            # Code-set provenance, not model-reported.
            "summary_basis": basis,
            # Which interest profile produced this score (C2). Lets /status
            # report drift and /runs/rescore find what needs re-running.
            "profile_version": self._profile_version,
            "chunks": chunk_count,
            "llm_calls": len(metas),
            "at": iso_now(),
            **call_meta_dict(metas[-1]),
            # Whole-episode totals across map+reduce calls.
            "cost_usd": sum(m.cost_usd for m in metas),
            "latency_ms": sum(m.latency_ms for m in metas),
        }
        return result, block, target

    def _log_scored(
        self,
        event: str,
        episode: Doc,
        result: Tier1Result,
        block: dict[str, Any],
        target: EpisodeStatus,
        *,
        previous: int | None = None,
    ) -> None:
        log.info(
            event,
            episode_id=episode["_id"],
            podcast=episode["podcast_slug"],
            score=result.relevance_score,
            previous_score=previous,
            threshold=self._settings.pipeline.digest_threshold,
            status=target.value,
            basis=block["summary_basis"],
            chunks=block["chunks"],
            llm_calls=block["llm_calls"],
            matched_interests=block["matched_interests"],
            listen_anyway=result.listen_anyway,
            profile_version=block["profile_version"],
            cost_usd=round(float(block["cost_usd"]), 6),
        )

    # --- content resolution -------------------------------------------------

    async def _resolve_content(self, episode: Doc) -> tuple[str, SummaryBasis]:
        """Return the text to summarise and its honest provenance label."""
        status = episode.get("status")
        if status != EpisodeStatus.TRANSCRIPT_FAILED.value:
            transcript = await load_transcript(self._store, episode["_id"])
            if transcript and transcript.strip():
                basis: SummaryBasis = (
                    "published_transcript"
                    if episode.get("transcript_source") in ("feed", "scrape")
                    else "transcript"
                )
                return transcript, basis
        return episode.get("description_raw") or "", "description_only"

    # --- single call --------------------------------------------------------

    async def _single_call(
        self, episode: Doc, content: str, basis: SummaryBasis
    ) -> tuple[Tier1Result, CallMeta]:
        system, user = self._single.render(
            interest_profile=self._profile_text,
            podcast_name=episode.get("podcast_name") or episode["podcast_slug"],
            title=episode.get("title") or "(untitled)",
            published_at=(episode.get("published_at") or "")[:10],
            age_note=describe_age(episode_age_days(episode.get("published_at"))),
            duration=format_duration(episode.get("duration_s")),
            basis=basis,
            content=truncate_to_tokens(content, self._settings.pipeline.max_input_tokens),
        )
        return await self._llm.complete_structured(
            "tier1",
            system,
            user,
            Tier1Result,
            episode_id=episode["_id"],
            prompt_version=self._single.versioned_name,
        )

    # --- map-reduce ---------------------------------------------------------

    async def _map_reduce(
        self, episode: Doc, content: str
    ) -> tuple[Tier1Result, list[CallMeta], int]:
        episode_id = episode["_id"]
        chunks = chunk_transcript(content, self._settings.pipeline.chunk_target_tokens)
        if len(chunks) > MAX_CHUNKS:
            log.warning(
                "tier1.chunks_truncated",
                episode_id=episode_id,
                chunks=len(chunks),
                kept=MAX_CHUNKS,
            )
            chunks = chunks[:MAX_CHUNKS]

        log.info(
            "tier1.map_reduce_start",
            episode_id=episode_id,
            chunks=len(chunks),
            estimated_tokens=estimate_tokens(content),
        )

        metas: list[CallMeta] = []
        all_bullets: list[str] = []
        all_entities: list[str] = []
        podcast_name = episode.get("podcast_name") or episode["podcast_slug"]
        title = episode.get("title") or "(untitled)"

        # Sequential on purpose: a local 27B+ model on the NAS has no spare
        # capacity for parallel long-context calls, and ordering keeps the
        # reduce step's input in episode order.
        for index, chunk in enumerate(chunks, start=1):
            system, user = self._map.render(
                interest_profile=self._profile_text,
                podcast_name=podcast_name,
                title=title,
                index=index,
                total=len(chunks),
                content=chunk,
            )
            bullets, meta = await self._llm.complete_structured(
                "tier1",
                system,
                user,
                ChunkBullets,
                episode_id=episode_id,
                prompt_version=self._map.versioned_name,
            )
            metas.append(meta)
            all_bullets.extend(bullets.bullets)
            all_entities.extend(bullets.entities)
            log.debug(
                "tier1.map_chunk_done",
                episode_id=episode_id,
                chunk=index,
                of=len(chunks),
                bullets=len(bullets.bullets),
            )

        if not all_bullets:
            raise ValueError(
                f"{episode_id}: map phase extracted no content from {len(chunks)} chunk(s)"
            )

        # Deduplicate entities case-insensitively, preserving first-seen order.
        seen: set[str] = set()
        unique_entities: list[str] = []
        for entity in all_entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)

        system, user = self._reduce.render(
            interest_profile=self._profile_text,
            podcast_name=podcast_name,
            title=title,
            published_at=(episode.get("published_at") or "")[:10],
            age_note=describe_age(episode_age_days(episode.get("published_at"))),
            duration=format_duration(episode.get("duration_s")),
            slice_count=len(chunks),
            bullets="\n".join(f"- {b}" for b in all_bullets),
            entities=", ".join(unique_entities) if unique_entities else "",
        )
        result, reduce_meta = await self._llm.complete_structured(
            "tier1",
            system,
            user,
            Tier1Result,
            episode_id=episode_id,
            prompt_version=self._reduce.versioned_name,
        )
        metas.append(reduce_meta)
        return result, metas, len(chunks)
