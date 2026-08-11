"""Pydantic structures the LLM tiers must produce, plus call telemetry.

Design note (deviation from design doc §4, deliberate): ``summary_basis`` is NOT
part of :class:`Tier1Result`. Whether a summary came from a transcript or only
from a description is a fact the pipeline knows for certain; asking the model to
self-report it would let untrusted output relabel its own provenance, and the
digest uses that label to be honest with the reader. It is set in code and merged
into the stored ``tier1`` block instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from .sanitize import md_escape_inline, sanitize_bullet, sanitize_md_block


class Route(StrEnum):
    """Tier-0 routing outcome. Decided in code from validated numeric fields."""

    DROP = "DROP"
    DIGEST_DIRECT = "DIGEST_DIRECT"
    ESCALATE = "ESCALATE"


SummaryBasis = Literal["transcript", "published_transcript", "description_only"]

#: Where a transcript came from, recorded on the episode doc.
TranscriptSource = Literal["feed", "scrape", "asr", "none"]


def clamped(limit: int) -> Any:
    """Trim an over-long value instead of rejecting it.

    `Field(max_length=...)` alone *rejects*, and the constraint is checked before
    any ``mode="after"`` validator — so the cleaning validators that truncate
    never ran on the values that needed truncating. A model returning 27 entities
    against a cap of 25 failed the whole call, retried, failed again, and lost
    the summary entirely.

    The caps bound output size; they are not requirements the model must meet.
    Two extra entities are worth keeping 25 of, and are certainly not worth
    discarding a transcript's worth of work over. Same reasoning as the note on
    `Tier1Result.key_takeaways`: accept a slightly off-spec response rather than
    burn retries on it. The `max_length` stays as a backstop.
    """

    def _trim(value: Any) -> Any:
        if isinstance(value, list | str) and len(value) > limit:
            return value[:limit]
        return value

    return BeforeValidator(_trim)


class Tier0Result(BaseModel):
    """Cheap description-level triage verdict (§4 stage 2)."""

    model_config = ConfigDict(extra="ignore")

    relevance_guess: int = Field(ge=0, le=10, description="Relevance 0-10 vs the interest profile")
    confidence: int = Field(
        ge=0, le=10, description="How informative the description is for judging relevance, 0-10"
    )
    matched_interests: Annotated[list[str], clamped(20)] = Field(
        default_factory=list, max_length=20, description="Interest profile keys that matched"
    )
    reasoning: Annotated[str, clamped(600)] = Field(
        default="", max_length=600, description="1-2 sentences, for audit only"
    )
    #: The model's suggested route. Recorded for audit and prompt evaluation but
    #: NEVER acted on — routing is recomputed in code (see triage.routing).
    route: Route = Route.ESCALATE

    @field_validator("reasoning")
    @classmethod
    def _clean_reasoning(cls, v: str) -> str:
        return md_escape_inline(v, max_chars=600)


class Tier1Result(BaseModel):
    """Full structured summary + final score (§4 stage 4)."""

    model_config = ConfigDict(extra="ignore")

    relevance_score: int = Field(ge=0, le=10)
    matched_interests: Annotated[list[str], clamped(20)] = Field(
        default_factory=list, max_length=20
    )
    why_it_matters: Annotated[str, clamped(1000)] = Field(default="", max_length=1000)
    summary_md: Annotated[str, clamped(12_000)] = Field(default="", max_length=12_000)
    # The prompt asks for 3-7 bullets. Validation is deliberately wider so a
    # slightly off-spec response is used rather than burning retries on it.
    key_takeaways: Annotated[list[str], clamped(12)] = Field(default_factory=list, max_length=12)
    entities: Annotated[list[str], clamped(40)] = Field(default_factory=list, max_length=40)
    listen_anyway: bool = False

    @field_validator("why_it_matters")
    @classmethod
    def _clean_why(cls, v: str) -> str:
        return md_escape_inline(v, max_chars=1000)

    @field_validator("summary_md")
    @classmethod
    def _clean_summary(cls, v: str) -> str:
        return sanitize_md_block(v)

    @field_validator("key_takeaways")
    @classmethod
    def _clean_takeaways(cls, v: list[str]) -> list[str]:
        return [b for b in (sanitize_bullet(x) for x in v) if b]

    @field_validator("entities")
    @classmethod
    def _clean_entities(cls, v: list[str]) -> list[str]:
        cleaned = [md_escape_inline(x, max_chars=80) for x in v]
        # Preserve order, drop blanks and duplicates.
        seen: set[str] = set()
        out: list[str] = []
        for item in cleaned:
            if item and item.lower() not in seen:
                seen.add(item.lower())
                out.append(item)
        return out


class ChunkBullets(BaseModel):
    """Map-step output when a transcript exceeds the single-call budget."""

    model_config = ConfigDict(extra="ignore")

    bullets: Annotated[list[str], clamped(15)] = Field(default_factory=list, max_length=15)
    entities: Annotated[list[str], clamped(25)] = Field(default_factory=list, max_length=25)

    @field_validator("bullets")
    @classmethod
    def _clean_bullets(cls, v: list[str]) -> list[str]:
        return [b for b in (sanitize_bullet(x, max_chars=300) for x in v) if b]

    @field_validator("entities")
    @classmethod
    def _clean_entities(cls, v: list[str]) -> list[str]:
        return [e for e in (md_escape_inline(x, max_chars=80) for x in v) if e]


class WeeklyTheme(BaseModel):
    """One thread running across several of the week's episodes (roadmap D1)."""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    summary: str = ""
    #: Show names the theme was drawn from, so a claim can be traced back.
    shows: Annotated[list[str], clamped(8)] = Field(default_factory=list, max_length=8)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        return md_escape_inline(v, max_chars=120)

    @field_validator("summary")
    @classmethod
    def _clean_summary(cls, v: str) -> str:
        return sanitize_md_block(v, max_chars=1200)

    @field_validator("shows")
    @classmethod
    def _clean_shows(cls, v: list[str]) -> list[str]:
        return [s for s in (md_escape_inline(x, max_chars=80) for x in v) if s]


class WeeklySynthesis(BaseModel):
    """A second-order pass over the week's summaries (roadmap D1).

    Reads summaries rather than transcripts, so it is one cheap call however
    many episodes the week held.
    """

    model_config = ConfigDict(extra="ignore")

    themes: Annotated[list[WeeklyTheme], clamped(4)] = Field(default_factory=list, max_length=4)
    #: Where shows or hosts genuinely disagreed. Often the most useful part, and
    #: the part no single episode's summary can contain.
    disagreements: Annotated[list[str], clamped(4)] = Field(default_factory=list, max_length=4)
    #: What is new relative to the previous digest's themes.
    whats_new: Annotated[list[str], clamped(4)] = Field(default_factory=list, max_length=4)

    @field_validator("disagreements", "whats_new")
    @classmethod
    def _clean_lines(cls, v: list[str]) -> list[str]:
        return [b for b in (sanitize_bullet(x, max_chars=400) for x in v) if b]

    def is_empty(self) -> bool:
        return not (self.themes or self.disagreements or self.whats_new)


class ContentSeed(BaseModel):
    """One episode, and what about it is worth writing (roadmap E3)."""

    model_config = ConfigDict(extra="ignore")

    #: Index of the episode in the list the prompt was given. A number rather
    #: than a title: a model asked to echo a title paraphrases it, and then
    #: nothing can be linked back to the episode it came from.
    ref: int = -1
    angle: str = ""
    why_now: str = ""
    #: True when the episode cuts against the usual position on its topic.
    #: Those are the ones worth writing; agreement rarely is.
    contrarian: bool = False

    @field_validator("angle")
    @classmethod
    def _clean_angle(cls, v: str) -> str:
        return sanitize_bullet(v, max_chars=400)

    @field_validator("why_now")
    @classmethod
    def _clean_why(cls, v: str) -> str:
        return sanitize_bullet(v, max_chars=300)


class ContentThread(BaseModel):
    """A topic with enough material across episodes for a longer piece."""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    argument: str = ""
    refs: Annotated[list[int], clamped(10)] = Field(default_factory=list, max_length=10)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        return md_escape_inline(v, max_chars=120)

    @field_validator("argument")
    @classmethod
    def _clean_argument(cls, v: str) -> str:
        return sanitize_md_block(v, max_chars=1200)


class ContentSeeds(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seeds: Annotated[list[ContentSeed], clamped(30)] = Field(default_factory=list, max_length=30)
    threads: Annotated[list[ContentThread], clamped(5)] = Field(default_factory=list, max_length=5)

    def is_empty(self) -> bool:
        return not (self.seeds or self.threads)


@dataclass(slots=True)
class CallMeta:
    """Telemetry for one structured LLM invocation (§6 ``llm_call`` doc)."""

    tier: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    fallback_used: bool = False
    validation_retries: int = 0
    prompt_version: str = ""
    episode_id: str | None = None
    #: Endpoints attempted, in order — useful when diagnosing fallback churn.
    attempted_models: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptResult:
    text: str
    source: TranscriptSource
    #: Populated for ASR runs so the digest can note detected language mismatch.
    detected_language: str | None = None
    duration_s: int | None = None
