"""Stage 2 — Tier-0 triage (§4).

One cheap LLM call per new episode, then a code-side routing decision. Triage and
dispatch are separate writes so a crash between them is recoverable: the stored
route is replayed on the next run rather than re-billing the LLM call.
"""

from __future__ import annotations

from ..config import Settings
from ..db import Doc, Store
from ..episodes import call_meta_dict, filter_interest_keys, transition
from ..llm.base import StructuredLLM
from ..llm.prompts import format_interest_profile, load_prompt
from ..logging_setup import get_logger
from ..models import Route, Tier0Result
from ..podcasts import PodcastRegistry
from ..state import EpisodeStatus
from ..utils import format_duration, iso_now
from .routing import RoutingDecision, decide_route

log = get_logger(__name__)

PROMPT_NAME = "tier0"
PROMPT_VERSION = "v1"


class Tier0Stage:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        llm: StructuredLLM,
        registry: PodcastRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._llm = llm
        self._registry = registry or PodcastRegistry(settings)
        self._prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
        self._profile_text = format_interest_profile(settings.interest_profile)
        self._valid_keys = {i.key for i in settings.interest_profile}
        self._profile_version = settings.interest_profile_version()

    async def triage(self, episode: Doc) -> RoutingDecision:
        """Run Tier-0 on one episode: NEW -> TRIAGED with the route recorded."""
        episode_id = episode["_id"]
        podcast = self._registry.podcast_by_slug(episode["podcast_slug"])
        always_escalate = bool(podcast.always_escalate) if podcast else False

        system, user = self._prompt.render(
            interest_profile=self._profile_text,
            podcast_name=episode.get("podcast_name") or episode["podcast_slug"],
            priority=(podcast.priority.value if podcast else "med"),
            title=episode.get("title") or "(untitled)",
            published_at=(episode.get("published_at") or "")[:10],
            duration=format_duration(episode.get("duration_s")),
            description=episode.get("description_raw") or "",
        )

        result, meta = await self._llm.complete_structured(
            "tier0",
            system,
            user,
            Tier0Result,
            episode_id=episode_id,
            prompt_version=self._prompt.versioned_name,
        )

        decision = decide_route(result, self._settings.pipeline, always_escalate=always_escalate)
        matched = filter_interest_keys(result.matched_interests, self._valid_keys)

        def _apply(doc: Doc) -> None:
            doc["tier0"] = {
                "relevance_guess": result.relevance_guess,
                "confidence": result.confidence,
                "matched_interests": matched,
                "reasoning": result.reasoning,
                # Recorded for prompt evaluation; never acted upon.
                "model_suggested_route": result.route.value,
                "route": decision.route.value,
                "rule": decision.rule,
                # Which interest profile produced this verdict (C2).
                "profile_version": self._profile_version,
                "at": iso_now(),
                **call_meta_dict(meta),
            }
            doc.setdefault("attempts", {})["tier0"] = (
                int((doc.get("attempts") or {}).get("tier0") or 0) + 1
            )

        await transition(self._store, episode_id, EpisodeStatus.TRIAGED, mutate=_apply)

        log.info(
            "tier0.routed",
            episode_id=episode_id,
            podcast=episode["podcast_slug"],
            title=(episode.get("title") or "")[:100],
            relevance=result.relevance_guess,
            confidence=result.confidence,
            route=decision.route.value,
            rule=decision.rule,
            model_suggested=result.route.value,
            matched_interests=matched,
            cost_usd=round(meta.cost_usd, 6),
        )
        return decision

    async def dispatch(self, episode: Doc) -> EpisodeStatus:
        """Apply the stored Tier-0 route: TRIAGED -> next status."""
        episode_id = episode["_id"]
        tier0 = episode.get("tier0") or {}
        raw_route = tier0.get("route")
        try:
            route = Route(str(raw_route))
        except ValueError:
            # Should be unreachable: triage always writes a valid route.
            raise ValueError(
                f"{episode_id}: TRIAGED without a valid route ({raw_route!r})"
            ) from None

        target = {
            Route.DROP: EpisodeStatus.DROPPED,
            Route.DIGEST_DIRECT: EpisodeStatus.DIGEST_DIRECT,
            Route.ESCALATE: EpisodeStatus.AWAITING_TRANSCRIPT,
        }[route]

        await transition(self._store, episode_id, target)
        log.debug("tier0.dispatched", episode_id=episode_id, status=target.value)
        return target
