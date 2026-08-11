"""Weekly cross-episode synthesis — the digest's opening section (roadmap D1).

One extra LLM call per digest, over the week's *summaries* rather than its
transcripts. That is the whole reason this is cheap: the expensive reading was
already done per episode, and what is missing is the thing no single episode's
summary can contain — where the week converged, where shows disagreed, and what
moved since last time.

Nothing here may fail a digest. A model that is down, slow, or returns something
unusable costs the reader an opening section and nothing else: the episode
summaries underneath are the artefact, and they are already written.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..db import Doc
from ..llm.base import LLMUnavailable, StructuredLLM
from ..llm.prompts import format_interest_profile, load_prompt
from ..logging_setup import get_logger
from ..models import WeeklySynthesis
from ..utils import estimate_tokens

log = get_logger(__name__)

PROMPT_NAME = "digest_themes"
PROMPT_VERSION = "v1"

#: A theme needs more than one episode by definition, and the highest-scoring
#: episodes are the ones the reader cares whether we connected. Beyond this the
#: input stops fitting a local model's window and the extra episodes are the
#: least relevant ones anyway.
MAX_EPISODES = 30

#: Below this there is no cross-episode structure to find, and asking for three
#: themes from two episodes invites invention.
MIN_EPISODES = 3

#: Key points kept per episode. Enough to recognise a shared thread, short
#: enough that thirty episodes still fit.
MAX_TAKEAWAYS = 5


def _episode_block(episode: Doc) -> str:
    """One episode as the synthesis prompt sees it: why it mattered, in brief."""
    tier1 = episode.get("tier1") or {}
    show = episode.get("podcast_name") or episode.get("podcast_slug") or "unknown show"
    title = episode.get("title") or "(untitled)"
    score = tier1.get("relevance_score")
    lines = [f'- show: "{show}" | episode: "{title}"' + (f" | score: {score}/10" if score else "")]
    if why := str(tier1.get("why_it_matters") or "").strip():
        lines.append(f"  why it mattered: {why}")
    takeaways = [str(t).strip() for t in (tier1.get("key_takeaways") or []) if str(t).strip()]
    for takeaway in takeaways[:MAX_TAKEAWAYS]:
        lines.append(f"  * {takeaway}")
    entities = [str(e).strip() for e in (tier1.get("entities") or []) if str(e).strip()]
    if entities:
        lines.append(f"  mentioned: {', '.join(entities[:12])}")
    return "\n".join(lines)


def select_episodes(episodes: list[Doc]) -> list[Doc]:
    """The summarised episodes worth synthesising, best first.

    Only episodes that actually have a Tier-1 summary: a digest-direct one-liner
    carries a guess from a feed description, and feeding those in would let the
    opening section assert things nobody read.
    """
    summarised = [
        e
        for e in episodes
        if (e.get("tier1") or {}).get("summary_md") and (e.get("tier1") or {}).get("why_it_matters")
    ]
    summarised.sort(
        key=lambda e: (-(int((e.get("tier1") or {}).get("relevance_score") or 0)), e.get("_id", ""))
    )
    return summarised[:MAX_EPISODES]


def previous_theme_titles(previous: Doc | None) -> list[str]:
    """Last digest's theme titles, for the "what's new" comparison."""
    if not previous:
        return []
    stored = (previous.get("synthesis") or {}).get("themes") or []
    return [str(t.get("title") or "").strip() for t in stored if t.get("title")]


class WeeklySynthesizer:
    def __init__(self, settings: Settings, llm: StructuredLLM) -> None:
        self._settings = settings
        self._llm = llm
        self._prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
        self._profile_text = format_interest_profile(settings.interest_profile)

    async def build(
        self,
        episodes: list[Doc],
        *,
        period_from: str,
        period_to: str,
        previous_themes: list[str] | None = None,
    ) -> WeeklySynthesis | None:
        """Return the week's synthesis, or None when there is nothing to say.

        Never raises. Every failure mode here — too few episodes, the model
        down, a response that validates to nothing — is the same outcome for the
        reader: the digest opens with its episode summaries, as it always did.
        """
        selected = select_episodes(episodes)
        if len(selected) < MIN_EPISODES:
            log.info(
                "synthesis.skipped",
                reason="too few summarised episodes to find a cross-episode theme",
                episodes=len(selected),
                minimum=MIN_EPISODES,
            )
            return None

        blocks = "\n".join(_episode_block(e) for e in selected)
        system, user = self._prompt.render(
            interest_profile=self._profile_text,
            period_from=period_from,
            period_to=period_to,
            episode_count=len(selected),
            episode_digests=blocks,
            previous_themes="\n".join(f"- {t}" for t in (previous_themes or [])),
        )

        try:
            result, meta = await self._llm.complete_structured(
                "tier1",
                system,
                user,
                WeeklySynthesis,
                prompt_version=self._prompt.versioned_name,
            )
        except LLMUnavailable as exc:
            # Expected whenever the local model is down and cloud endpoints are
            # off. The digest is not held back for it.
            log.warning("synthesis.unavailable", error=str(exc))
            return None
        except Exception as exc:
            log.error("synthesis.failed", error=str(exc), exc_info=True)
            return None

        if result.is_empty():
            log.info("synthesis.empty", detail="the model found no cross-episode structure")
            return None

        log.info(
            "synthesis.built",
            episodes=len(selected),
            themes=len(result.themes),
            disagreements=len(result.disagreements),
            model=meta.model,
            cost_usd=meta.cost_usd,
            input_tokens_estimated=estimate_tokens(blocks),
        )
        return result


def as_view(synthesis: WeeklySynthesis | None) -> dict[str, Any] | None:
    """Template-ready form. Fields are already sanitised by the model."""
    if synthesis is None:
        return None
    return {
        "themes": [
            {"title": t.title, "summary": t.summary, "shows": t.shows} for t in synthesis.themes
        ],
        "disagreements": list(synthesis.disagreements),
        "whats_new": list(synthesis.whats_new),
    }
