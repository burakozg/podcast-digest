"""Layered configuration: config.yaml (non-secret) + environment (secrets).

Invalid configuration is a startup crash, never a half-configured run (§8).
Every model uses ``extra="forbid"`` so a typo'd key fails loudly instead of
being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_CONFIG_FILE = "config.yaml"


class Priority(StrEnum):
    HIGH = "high"
    MED = "med"
    LOW = "low"


class Provider(StrEnum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"


#: Providers that send content off the LAN. Gated by ``allow_cloud_fallback``.
CLOUD_PROVIDERS: frozenset[Provider] = frozenset({Provider.OPENROUTER, Provider.ANTHROPIC})

#: litellm route prefix per provider. Ollama maps to ``ollama_chat`` (the /api/chat
#: endpoint) rather than ``ollama`` (/api/generate): the pipeline always sends
#: chat messages and relies on JSON-mode structured output, which needs /api/chat.
LITELLM_PREFIX: dict[Provider, str] = {
    Provider.OLLAMA: "ollama_chat",
    Provider.OPENROUTER: "openrouter",
    Provider.ANTHROPIC: "anthropic",
}

#: Used when an ollama endpoint omits api_base.
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"must be an http(s) URL, got {value!r}")
    if not parsed.netloc:
        raise ValueError(f"missing host in URL {value!r}")
    return value


class PodcastConfig(StrictModel):
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64)
    name: str = Field(min_length=1, max_length=200)
    feed_url: str
    priority: Priority = Priority.MED
    #: Force full Tier-1 treatment regardless of the Tier-0 verdict.
    always_escalate: bool = False
    #: Observed to publish <podcast:transcript>; purely informational hint used
    #: in logs and /status — acquisition always probes the feed anyway.
    has_feed_transcripts: bool = False
    #: CSS selector for show-notes transcript scraping. Generic scraping is
    #: never attempted; a show must opt in explicitly (§3 stage 3).
    transcript_selector: str | None = None
    #: Rewrite the episode link before scraping, as ``[from, to]``.
    #:
    #: Several publishers put the transcript on a sibling page rather than the
    #: one the feed links to — CyberWire links to ``…/2594/notes`` and serves
    #: the transcript from ``…/2594/transcript``. Without this a selector is
    #: applied to the wrong page and matches nothing, which reads exactly like
    #: a show that publishes no transcript.
    #:
    #: A plain substring substitution, not a regex: it only has to reach a
    #: sibling path, and a pattern from a config file is not worth the
    #: backtracking risk. The rewritten URL is guarded like any other.
    transcript_url_sub: tuple[str, str] | None = None
    enabled: bool = True
    #: Whether the routine pipeline may download and transcribe this show's
    #: audio locally when no published transcript exists. Off by default: ASR is
    #: the expensive path, and most shows do not repay it. Backfill additionally
    #: Governs both the weekly pipeline and the archive walk: this is the only
    #: switch that lets either transcribe, so an archive is opted into per
    #: podcast rather than globally.
    asr_enabled: bool = False
    #: How this show is treated during archive backfill (roadmap A1).
    #: ``tier0_only`` skips summarisation — right for daily news shows, where a
    #: two-year-old headline round-up is not worth a summary. ``full`` gives
    #: evergreen shows the same treatment as a fresh episode. ``skip`` excludes
    #: the show from backfill entirely.
    #: Defaults to ``skip``: adding a podcast should not silently start walking
    #: years of its back catalogue. Archiving one is a decision, made per
    #: podcast, after seeing what it publishes.
    backfill_mode: Literal["full", "tier0_only", "skip"] = "skip"
    #: How far back the archive walk reaches for *this* show, in months. ``None``
    #: inherits ``backfill.months``, so the common case needs nothing set.
    #:
    #: A short list rather than a free number: each step is roughly another year
    #: of archive for this show, and the estimate read before starting was built
    #: for one of these.
    backfill_months: Literal[12, 24, 36] | None = None

    _check_feed_url = field_validator("feed_url")(_require_http_url)


#: Archive windows offered per podcast. A short list rather than a free number:
#: each step is roughly another year of archive for that show.
WINDOW_CHOICES: tuple[int, ...] = (12, 24, 36)


class InterestItem(StrictModel):
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)
    label: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    weight: int = Field(ge=1, le=10)


class SchedulerConfig(StrictModel):
    timezone: str = "Europe/Stockholm"
    ingest_cron: str = "0 */6 * * *"
    pipeline_cron: str = "*/30 * * * *"
    digest_cron: str = "0 6 * * fri"
    retention_cron: str = "0 4 * * *"
    #: Reader marks into the vault. Half an hour after the digest, so a week's
    #: marks land beside the digest that prompted them.
    signals_cron: str = "30 6 * * fri"
    #: Archive walk. Fires often, but does nothing unless backfill is resumed —
    #: the run-state flag defaults to paused.
    backfill_cron: str = "*/20 * * * *"
    #: Keeps the search index current. Offset from `pipeline_cron` rather than
    #: sharing it, so the sync runs *after* a batch of summaries lands instead
    #: of alongside it. Cheap: it pages episode documents and only fetches a
    #: transcript for an episode whose content actually changed.
    search_cron: str = "15,45 * * * *"
    #: Kick ingest+pipeline once at boot (useful for a fresh deployment).
    run_on_startup: bool = False

    @field_validator("timezone")
    @classmethod
    def _tz_exists(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {v!r}") from exc
        return v

    @model_validator(mode="after")
    def _crons_parse(self) -> SchedulerConfig:
        # Validate here so a bad cron string fails at startup, not at first fire.
        from apscheduler.triggers.cron import CronTrigger  # local: keep import cost off cold paths

        for field in (
            "ingest_cron",
            "pipeline_cron",
            "digest_cron",
            "retention_cron",
            "signals_cron",
            "backfill_cron",
            "search_cron",
        ):
            expr = getattr(self, field)
            try:
                CronTrigger.from_crontab(expr, timezone=self.timezone)
            except Exception as exc:
                raise ValueError(f"{field}: invalid cron expression {expr!r} ({exc})") from exc
        return self


class PipelineConfig(StrictModel):
    t_conf_high: int = Field(default=7, ge=0, le=10)
    t_rel_low: int = Field(default=4, ge=0, le=10)
    t_rel_high: int = Field(default=7, ge=0, le=10)
    digest_threshold: int = Field(default=5, ge=0, le=10)
    top_pick_threshold: int = Field(default=8, ge=0, le=10)
    initial_lookback_days: int = Field(default=14, ge=0, le=3650)
    #: How far back a digest will reach for episodes nothing has claimed yet.
    #:
    #: Selection is claim-once via `digest_id`, so this is not what stops an
    #: episode appearing twice — it only stops a fresh install dragging its whole
    #: initial ingest into the first digest. It has to be comfortably longer than
    #: the gap between digests: an episode still awaiting a transcript when a
    #: digest runs is behind the next window's start by definition, and anything
    #: this does not reach back far enough to catch is stranded for good.
    digest_catch_up_days: int = Field(default=30, ge=1, le=3650)
    #: Roadmap D1. One extra LLM call per digest, over the week's summaries
    #: rather than its transcripts — so it costs about as much as summarising a
    #: single episode, whatever the week held. Off leaves the digest exactly as
    #: it was; a failure at run time does the same, silently.
    weekly_synthesis: bool = True
    max_retries: int = Field(default=3, ge=0, le=10)
    description_max_chars: int = Field(default=2000, ge=200, le=50_000)
    max_input_tokens: int = Field(default=24_000, ge=1000)
    chunk_target_tokens: int = Field(default=6000, ge=500)
    max_triage_per_run: int = Field(default=60, ge=1)
    max_transcripts_per_run: int = Field(default=6, ge=1)
    max_summaries_per_run: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> PipelineConfig:
        if self.t_rel_low > self.t_rel_high:
            raise ValueError(
                f"t_rel_low ({self.t_rel_low}) must be <= t_rel_high ({self.t_rel_high})"
            )
        if self.top_pick_threshold < self.digest_threshold:
            raise ValueError(
                f"top_pick_threshold ({self.top_pick_threshold}) must be >= "
                f"digest_threshold ({self.digest_threshold})"
            )
        if self.chunk_target_tokens > self.max_input_tokens:
            raise ValueError("chunk_target_tokens must be <= max_input_tokens")
        return self


class BackfillConfig(StrictModel):
    """Deliberate, rate-limited walk backwards through the archive (roadmap A1).

    Separate from ``pipeline`` throughout: backfill is a different economic
    proposition from routine polling, so its thresholds, caps and output are its
    own. Nothing here affects the weekly digest.
    """

    #: How far back to walk, in whole months.
    months: int = Field(default=12, ge=1, le=240)
    #: Stricter than the weekly threshold — an old episode must earn its summary.
    digest_threshold: int = Field(default=7, ge=0, le=10)
    #: Month-windows processed per podcast per run, so a run has a bounded cost.
    months_per_run: int = Field(default=1, ge=1, le=12)
    #: Hard ceiling on episodes ingested in one run, across all shows.
    max_episodes_per_run: int = Field(default=200, ge=1, le=5000)
    #: Episodes summarised per processing run.
    max_summaries_per_run: int = Field(default=20, ge=1, le=500)
    #: Episodes whose transcript is acquired per processing run. Its own knob,
    #: not the summary budget: acquisition may transcribe locally, which costs
    #: hours where a summary costs seconds, and an archive walk is unattended.
    #: Sizing the transcription batch by ``max_summaries_per_run`` was a
    #: copy-paste that silently tied one to the other.
    max_transcripts_per_run: int = Field(default=6, ge=1, le=500)


class LLMEndpoint(StrictModel):
    provider: Provider
    model: str = Field(min_length=1)
    api_base: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    #: Escape hatch passed straight through to litellm (e.g. num_ctx for Ollama).
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_base")
    @classmethod
    def _base_url(cls, v: str | None) -> str | None:
        return None if v is None else _require_http_url(v)

    @property
    def is_cloud(self) -> bool:
        return self.provider in CLOUD_PROVIDERS

    def litellm_model(self) -> str:
        """The provider-prefixed model string litellm expects."""
        return f"{LITELLM_PREFIX[self.provider]}/{self.model}"

    def resolved_api_base(self) -> str | None:
        if self.api_base:
            return self.api_base
        return DEFAULT_OLLAMA_BASE if self.provider is Provider.OLLAMA else None


class LLMTierConfig(StrictModel):
    primary: LLMEndpoint
    fallbacks: list[LLMEndpoint] = Field(default_factory=list)
    timeout_s: int = Field(default=120, ge=5, le=3600)
    validation_retries: int = Field(default=2, ge=0, le=5)
    #: Data-sovereignty switch (§10.6). When false, cloud endpoints are dropped
    #: from the chain entirely — work queues instead of leaving the LAN.
    allow_cloud_fallback: bool = True

    def active_chain(self) -> list[LLMEndpoint]:
        """Endpoints actually eligible for use, in priority order."""
        chain = [self.primary, *self.fallbacks]
        if not self.allow_cloud_fallback:
            chain = [e for e in chain if not e.is_cloud]
        return chain

    @model_validator(mode="after")
    def _chain_not_empty_when_local_only(self) -> LLMTierConfig:
        if not self.active_chain():
            raise ValueError(
                "allow_cloud_fallback is false but every configured endpoint is a "
                "cloud provider — this tier could never run"
            )
        return self


class LLMConfig(StrictModel):
    tiers: dict[str, LLMTierConfig]
    #: DEBUG-level logging of full prompts/responses. Off by default (§10.1).
    log_llm_io: bool = False

    REQUIRED_TIERS: ClassVar[tuple[str, ...]] = ("tier0", "tier1")

    @model_validator(mode="after")
    def _has_required_tiers(self) -> LLMConfig:
        missing = [t for t in self.REQUIRED_TIERS if t not in self.tiers]
        if missing:
            raise ValueError(f"llm.tiers missing required tier(s): {', '.join(missing)}")
        return self


class ASRConfig(StrictModel):
    backend: Literal["local", "remote"] = "local"
    model: str = "large-v3-turbo"
    compute_type: str = "int8"
    device: str = "cpu"
    language: str | None = "en"
    beam_size: int = Field(default=1, ge=1, le=10)
    max_audio_mb: int = Field(default=300, ge=1, le=5000)
    keep_audio: bool = False
    asr_concurrency: int = Field(default=1, ge=1, le=4)
    download_concurrency: int = Field(default=2, ge=1, le=8)
    remote_url: str | None = None
    #: How long to wait for a remote transcription. Generous on purpose: this is
    #: one HTTP call covering the whole decode, so it must outlast the longest
    #: episode on the slowest machine that might answer. A 3-hour episode at 2x
    #: realtime is 90 minutes of silence on the socket.
    remote_timeout_s: int = Field(default=2700, ge=30, le=21600)

    @model_validator(mode="after")
    def _remote_needs_url(self) -> ASRConfig:
        if self.backend == "remote" and not self.remote_url:
            raise ValueError("asr.remote_url is required when asr.backend is 'remote'")
        return self


class OutputConfig(StrictModel):
    digest_dir: Path = Path("/data/digests")
    #: Scratch space for audio downloads and other transient files.
    work_dir: Path = Path("/data/work")
    episode_notes: bool = False


class CouchDBConfig(StrictModel):
    url: str = "http://couchdb-podcast:5984"
    db: str = Field(default="podcast_agent", pattern=r"^[a-z][a-z0-9_$()+/-]*$")
    user: str = "podagent"

    _check_url = field_validator("url")(_require_http_url)


class APIConfig(StrictModel):
    host: str = "0.0.0.0"  # noqa: S104 — bound to the container's LAN IP by compose
    port: int = Field(default=8080, ge=1, le=65535)


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"


class RetentionConfig(StrictModel):
    #: Days to keep transcripts, or **0 to keep them indefinitely**.
    #:
    #: Zero is the interesting value. Transcripts are the corpus: entity
    #: timelines and any retrieval over the archive can only reach back as far
    #: as the transcripts still exist, so expiring them silently caps what those
    #: can ever answer. The archive walk is filling that corpus now, and at
    #: ~14 KB gzipped per episode the whole thing is tens of megabytes — the
    #: storage this was protecting is not worth the capability it costs.
    #:
    #: Summaries are kept regardless; this has only ever governed transcripts.
    transcript_days: int = Field(default=180, ge=0)
    llm_call_days: int = Field(default=365, ge=1)
    #: Job-run history. Shorter than llm_call_days: these answer "what has
    #: been happening lately", and a year of scheduler firings is noise.
    run_days: int = Field(default=90, ge=1)
    #: Stored warnings and errors. Shorter still: these answer "what has
    #: been going wrong lately", and stale ones are a table nobody reads.
    log_days: int = Field(default=30, ge=1)


class ContentConfig(StrictModel):
    """Seeds for writing of your own, drawn from the corpus (roadmap E3).

    Off by default. It is a personal workflow rather than part of the digest,
    and a system that starts suggesting what to post without being asked is
    presumptuous in a way the rest of this deliberately is not.
    """

    enabled: bool = False
    #: Interest keys that make an episode worth writing about. Empty means any —
    #: which is almost always too wide to be useful. Narrowing it is the point:
    #: the output is only valuable if it is short enough to read every line.
    interests: list[str] = Field(default_factory=list)
    #: Relevance floor. Higher than the digest's, because an episode has to be
    #: worth *your* time twice over: worth reading, then worth writing about.
    min_score: int = Field(default=7, ge=0, le=10)
    window_days: int = Field(default=30, ge=1, le=365)
    #: Cap on episodes sent to the model in one pass. Also the cap on how long
    #: the output can get, which matters more.
    max_episodes: int = Field(default=15, ge=1, le=60)


class NotificationConfig(StrictModel):
    """Push notifications for exceptional episodes only (roadmap E4)."""

    enabled: bool = False
    #: Base URL of an ntfy server, e.g. http://ntfy.lan:8080 (self-hosted).
    ntfy_url: str | None = None
    topic: str | None = None
    #: Strict by design: a notification that fires weekly stops being read.
    min_score: int = Field(default=9, ge=0, le=10)
    priority: Literal["min", "low", "default", "high", "urgent"] = "default"
    tags: list[str] = Field(default_factory=lambda: ["headphones"])

    @field_validator("ntfy_url")
    @classmethod
    def _url(cls, v: str | None) -> str | None:
        return None if v is None else _require_http_url(v)

    @model_validator(mode="after")
    def _enabled_needs_a_target(self) -> NotificationConfig:
        if self.enabled and not (self.ntfy_url and self.topic):
            raise ValueError(
                "notifications.enabled is true but ntfy_url/topic are not set — "
                "notifications would silently never fire"
            )
        return self


class SecurityConfig(StrictModel):
    enforce_domain_allowlist: bool = True
    cdn_allowlist: list[str] = Field(default_factory=list)
    #: Hard ceiling on any single fetched text resource (transcripts, pages).
    max_text_download_mb: int = Field(default=25, ge=1, le=500)


def _yaml_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level of the config file must be a mapping")
    return loaded


class _YamlSource(PydanticBaseSettingsSource):
    """Feeds config.yaml into the settings chain below environment variables."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Unused: the whole document is supplied at once via __call__.
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        return _yaml_settings(self._path)


class _OverridesSource(PydanticBaseSettingsSource):
    """Feeds the console-editable overrides in *below* environment variables.

    These used to be deep-merged into the YAML dict and handed to ``Settings``
    as init kwargs, which outrank every other source. Two things went wrong with
    that, both silent:

    * Environment variables for an overridden section stopped applying at all.
      ``PODAGENT_ASR__DOWNLOAD_CONCURRENCY=1`` was ignored for five days in
      production because an ``asr`` override existed, and nothing said so.
    * Keys absent from a *partial* override fell back to **field defaults**
      rather than to config.yaml or the environment — so overriding
      ``asr.model`` quietly reset ``asr.remote_url`` to None and
      ``download_concurrency`` to 2.

    As a source they compose properly instead: pydantic-settings deep-merges
    across sources, so a partial override changes only the keys it names, and
    the environment still wins over it.
    """

    def __init__(self, settings_cls: type[BaseSettings], overrides: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._overrides = overrides

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Unused: the whole document is supplied at once via __call__.
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        return self._overrides


#: Set by load_settings() before the model is constructed. Module state is the
#: only way to parameterise a pydantic-settings source at class level.
_active_yaml_path: Path = Path(DEFAULT_CONFIG_FILE)
_active_overrides: dict[str, Any] = {}


class Settings(BaseSettings):
    """Fully-resolved application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PODAGENT_",
        env_nested_delimiter="__",
        # A typo'd key in config.yaml must be a startup crash, not a silent no-op.
        extra="forbid",
        # Secrets come from the process environment; .env is a dev convenience.
        env_file=".env",
        env_file_encoding="utf-8",
        # One .env is shared with docker-compose, which needs its own non-prefixed
        # vars (COUCHDB_USER, DIGEST_DIR, macvlan settings...). Without this the
        # default dotenv source passes those through and extra="forbid" rejects
        # them, so a correctly-filled .env would refuse to boot. "match_prefix"
        # keeps only PODAGENT_* from the file — the compose vars are simply not
        # this application's business.
        dotenv_filtering="match_prefix",
    )

    podcasts: list[PodcastConfig] = Field(default_factory=list)
    interest_profile: list[InterestItem] = Field(default_factory=list)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    backfill: BackfillConfig = Field(default_factory=BackfillConfig)
    llm: LLMConfig
    asr: ASRConfig = Field(default_factory=ASRConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    couchdb: CouchDBConfig = Field(default_factory=CouchDBConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)

    # --- Secrets: environment only, never YAML, never logged (§8) -----------
    admin_api_key: SecretStr | None = None
    ntfy_token: SecretStr | None = None
    couchdb_password: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: init args > env > .env > console overrides > config.yaml
        # > defaults.
        #
        # Overrides sit below the environment on purpose. They are edited from a
        # browser; deployment topology (where the database is, which machine
        # transcribes) comes from the environment and must not be displaceable
        # by one.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _OverridesSource(settings_cls, _active_overrides),
            _YamlSource(settings_cls, _active_yaml_path),
            file_secret_settings,
        )

    @field_validator("podcasts")
    @classmethod
    def _unique_slugs(cls, v: list[PodcastConfig]) -> list[PodcastConfig]:
        dupes = sorted({s for s, n in Counter(p.slug for p in v).items() if n > 1})
        if dupes:
            raise ValueError(f"duplicate podcast slug(s): {', '.join(dupes)}")
        return v

    @field_validator("interest_profile")
    @classmethod
    def _unique_interest_keys(cls, v: list[InterestItem]) -> list[InterestItem]:
        if not v:
            raise ValueError("interest_profile must define at least one interest")
        dupes = sorted({k for k, n in Counter(i.key for i in v).items() if n > 1})
        if dupes:
            raise ValueError(f"duplicate interest key(s): {', '.join(dupes)}")
        return v

    @model_validator(mode="after")
    def _cloud_creds_present(self) -> Settings:
        """A configured cloud endpoint without its API key is a silent-failure trap."""
        needed: set[Provider] = set()
        for tier in self.llm.tiers.values():
            for endpoint in tier.active_chain():
                if endpoint.is_cloud:
                    needed.add(endpoint.provider)
        if Provider.OPENROUTER in needed and not self.openrouter_api_key:
            raise ValueError(
                "an openrouter endpoint is active but PODAGENT_OPENROUTER_API_KEY is unset "
                "(set the key, or set allow_cloud_fallback: false for that tier)"
            )
        if Provider.ANTHROPIC in needed and not self.anthropic_api_key:
            raise ValueError(
                "an anthropic endpoint is active but PODAGENT_ANTHROPIC_API_KEY is unset "
                "(set the key, or set allow_cloud_fallback: false for that tier)"
            )
        return self

    @model_validator(mode="after")
    def _enabled_podcasts_exist(self) -> Settings:
        if not [p for p in self.podcasts if p.enabled]:
            raise ValueError("no enabled podcasts configured — nothing to ingest")
        return self

    # --- Convenience --------------------------------------------------------

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.scheduler.timezone)

    def interest_profile_version(self) -> str:
        """Short stable hash of the interest profile (C2).

        Recorded on every tier result so it is always answerable which profile a
        score was produced under. Covers key, label, description and weight —
        all of them reach the prompt, so any change can move a score. Ordering is
        normalised so reordering the YAML is not treated as a change.
        """
        payload = json.dumps(
            sorted(
                (
                    {
                        "key": item.key,
                        "label": item.label,
                        "description": item.description,
                        "weight": item.weight,
                    }
                    for item in self.interest_profile
                ),
                key=lambda entry: str(entry["key"]),
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def podcast_by_slug(self, slug: str) -> PodcastConfig | None:
        return next((p for p in self.podcasts if p.slug == slug), None)

    def enabled_podcasts(self) -> list[PodcastConfig]:
        return [p for p in self.podcasts if p.enabled]

    def api_key_for(self, provider: Provider) -> str | None:
        match provider:
            case Provider.OPENROUTER:
                key = self.openrouter_api_key
                return key.get_secret_value() if key else None
            case Provider.ANTHROPIC:
                return self.anthropic_api_key.get_secret_value() if self.anthropic_api_key else None
            case _:
                return None


def load_settings(
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build Settings from ``config_file`` (default: $PODAGENT_CONFIG_FILE or ./config.yaml).

    ``overrides`` is the console-editable layer, deep-merged over the file before
    validation — so an override that produces incoherent config fails the same
    way a bad file does, rather than at first use.
    """
    global _active_yaml_path, _active_overrides
    path = Path(config_file or os.environ.get("PODAGENT_CONFIG_FILE", DEFAULT_CONFIG_FILE))
    _active_yaml_path = path
    _active_overrides = overrides or {}
    # No init kwargs: overrides are a *source* now, ranked below the environment
    # (see _OverridesSource). Passing them here would put them above it again.
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (used by FastAPI dependencies)."""
    return load_settings()
