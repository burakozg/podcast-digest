"""Content seeds — openings worth writing about (roadmap E3).

The digest answers "what happened". This answers a different question the same
material can support: *is there anything here I should say something about?*

Same economics as the weekly synthesis. It reads summaries rather than
transcripts, so it is one call however many episodes qualify, and the expensive
reading was already done per episode.

Off by default, and it stays off until configured. A system that starts
suggesting what to post without being asked is presumptuous in a way the rest of
this deliberately is not — and the output is only useful when it is short enough
to read every line, which means narrowing it to the interests you actually write
about rather than everything you read.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from .config import Settings
from .db import Doc, Store, typed_sort
from .digest.generate import _atomic_write
from .llm.base import LLMUnavailable, StructuredLLM
from .llm.prompts import format_interest_profile, load_prompt
from .logging_setup import get_logger
from .models import ContentSeeds
from .sanitize import md_escape_inline, safe_url
from .state import EpisodeStatus
from .utils import iso, utcnow

log = get_logger(__name__)

PROMPT_NAME = "content_seeds"
PROMPT_VERSION = "v1"

OUTPUT_FILENAME = "content-seeds.md"

#: Key points kept per episode. Enough to recognise an opening, short enough
#: that a month of episodes still fits a local model's window.
MAX_TAKEAWAYS = 5

#: Statuses meaning the episode was actually summarised and read.
ELIGIBLE = frozenset({EpisodeStatus.READY_FOR_DIGEST.value, EpisodeStatus.PUBLISHED.value})


def _block(index: int, episode: Doc) -> str:
    tier1 = episode.get("tier1") or {}
    show = episode.get("podcast_name") or episode.get("podcast_slug") or "unknown show"
    lines = [
        f'{index}. show: "{show}" | episode: "{episode.get("title") or "(untitled)"}" '
        f"| {(episode.get('published_at') or '')[:10]} "
        f"| score: {tier1.get('relevance_score')}/10"
    ]
    if why := str(tier1.get("why_it_matters") or "").strip():
        lines.append(f"   why it mattered: {why}")
    for takeaway in [str(t).strip() for t in (tier1.get("key_takeaways") or []) if str(t).strip()][
        :MAX_TAKEAWAYS
    ]:
        lines.append(f"   * {takeaway}")
    if entities := [str(e).strip() for e in (tier1.get("entities") or []) if str(e).strip()]:
        lines.append(f"   mentioned: {', '.join(entities[:12])}")
    return "\n".join(lines)


async def select(store: Store, settings: Settings) -> list[Doc]:
    """Episodes worth considering, best first.

    Two filters, and both matter. The score floor is higher than the digest's
    because an episode has to be worth your time twice over — worth reading,
    then worth writing about. The interest filter is what keeps the output short
    enough to read every line.
    """
    cfg = settings.content
    since = iso(utcnow() - timedelta(days=cfg.window_days))
    wanted = set(cfg.interests)

    docs = await store.find(
        {"type": "episode", "published_at": {"$gte": since}},
        sort=typed_sort("published_at", "desc"),
        limit=1000,
    )
    kept: list[Doc] = []
    for episode in docs:
        if episode.get("status") not in ELIGIBLE:
            continue
        tier1 = episode.get("tier1") or {}
        if not tier1.get("summary_md"):
            continue
        if int(tier1.get("relevance_score") or 0) < cfg.min_score:
            continue
        if wanted and not wanted.intersection(tier1.get("matched_interests") or []):
            continue
        kept.append(episode)

    kept.sort(
        key=lambda e: (
            -(int((e.get("tier1") or {}).get("relevance_score") or 0)),
            e.get("published_at") or "",
        ),
        reverse=False,
    )
    return kept[: cfg.max_episodes]


class ContentSeedBuilder:
    def __init__(self, settings: Settings, store: Store, llm: StructuredLLM) -> None:
        self._settings = settings
        self._store = store
        self._llm = llm
        self._prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
        self._profile_text = format_interest_profile(settings.interest_profile)

    async def build(self) -> tuple[ContentSeeds | None, list[Doc]]:
        """Return the seeds and the episodes they refer to.

        Never raises. A model that is down costs a file nobody was waiting for.
        """
        episodes = await select(self._store, self._settings)
        if not episodes:
            log.info("content.no_candidates", window_days=self._settings.content.window_days)
            return None, []

        blocks = "\n".join(_block(i + 1, e) for i, e in enumerate(episodes))
        period_to = utcnow()
        system, user = self._prompt.render(
            interest_profile=self._profile_text,
            period_from=iso(period_to - timedelta(days=self._settings.content.window_days))[:10],
            period_to=iso(period_to)[:10],
            episode_count=len(episodes),
            episode_digests=blocks,
        )
        try:
            result, meta = await self._llm.complete_structured(
                "tier1", system, user, ContentSeeds, prompt_version=self._prompt.versioned_name
            )
        except LLMUnavailable as exc:
            log.warning("content.unavailable", error=str(exc))
            return None, episodes
        except Exception as exc:
            log.error("content.failed", error=str(exc), exc_info=True)
            return None, episodes

        if result.is_empty():
            # A real answer, and the prompt asks for it: a month with no opening
            # worth writing about is an ordinary month.
            log.info("content.nothing_worth_writing", episodes=len(episodes))
            return result, episodes

        log.info(
            "content.built",
            episodes=len(episodes),
            seeds=len(result.seeds),
            threads=len(result.threads),
            model=meta.model,
            cost_usd=meta.cost_usd,
        )
        return result, episodes


def render(seeds: ContentSeeds, episodes: list[Doc], settings: Settings) -> str:
    """Markdown for the vault. Every seed carries the episode it came from."""
    by_ref = {i + 1: e for i, e in enumerate(episodes)}
    generated = utcnow().astimezone(settings.tz).isoformat(timespec="seconds")
    out = [
        "---",
        "type: content-seeds",
        f"generated: {generated}",
        f"window_days: {settings.content.window_days}",
        f"episodes_considered: {len(episodes)}",
        f"seeds: {len(seeds.seeds)}",
        "tags: [content-seeds, cybersecurity]",
        "---",
        "",
        "# Content seeds",
        "",
        f"*From {len(episodes)} episode{'s' if len(episodes) != 1 else ''} in the last "
        f"{settings.content.window_days} days, scoring "
        f"{settings.content.min_score}+"
        + (f" on {', '.join(settings.content.interests)}" if settings.content.interests else "")
        + ".*",
        "",
    ]

    if seeds.threads:
        out += ["## Threads worth a longer piece", ""]
        for thread in seeds.threads:
            out.append(f"### {thread.title}")
            out += ["", thread.argument, ""]
            for ref in thread.refs:
                if episode := by_ref.get(ref):
                    out.append(f"- {_cite(episode)}")
            out.append("")

    if seeds.seeds:
        out += ["## Single-episode angles", ""]
        for seed in seeds.seeds:
            episode = by_ref.get(seed.ref)
            if episode is None:
                # The model referenced an episode that was not in the list.
                # Dropped rather than rendered: an angle whose source cannot be
                # named is exactly the thing this must not produce.
                log.warning("content.unknown_ref", ref=seed.ref)
                continue
            flag = " · **against the grain**" if seed.contrarian else ""
            out.append(f"### {_cite(episode)}{flag}")
            out += ["", seed.angle, ""]
            if seed.why_now:
                out += [f"*Why now:* {seed.why_now}", ""]

    if not seeds.threads and not seeds.seeds:
        out += [
            "Nothing this period offered an opening worth writing about.",
            "",
            "That is an ordinary outcome, not a failure — the alternative is a list",
            "of mediocre angles nobody reads.",
            "",
        ]

    out += [
        "---",
        "*Suggested openings, not drafts. Every angle traces to the episode above it;"
        " check the claim before you publish it.*",
    ]
    return "\n".join(out) + "\n"


def _cite(episode: Doc) -> str:
    show = md_escape_inline(
        episode.get("podcast_name") or episode.get("podcast_slug") or "", max_chars=80
    )
    title = md_escape_inline(episode.get("title") or "(untitled)", max_chars=180)
    date = (episode.get("published_at") or "")[:10]
    link = safe_url(episode.get("link"))
    cite = f"{show} — {title}"
    return f"[{cite}]({link}) · {date}" if link else f"{cite} · {date}"


def write(settings: Settings, body: str) -> Path:
    """Write the seeds file, never overwriting an earlier one."""
    return _atomic_write(settings.output.digest_dir, Path(OUTPUT_FILENAME), body)
