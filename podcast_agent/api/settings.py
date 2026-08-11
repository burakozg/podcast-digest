"""Configuration endpoints for the console.

Exposes the editable slice of configuration — which model each tier uses, the
routing thresholds, and the interest profile — reading `config.yaml` as the
baseline and storing changes as overrides in the database.

Changes apply at the next restart. Swapping a provider rebuilds the LLM router
and changing the interest profile invalidates every cached prompt; doing either
underneath a summarisation in flight is a poor trade for a setting that changes a
few times a year. The endpoints report whether a restart is pending.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import Settings
from ..db import Store
from ..logging_setup import get_logger
from ..settings_store import (
    OVERRIDABLE_ASR_KEYS,
    OVERRIDABLE_BACKFILL_KEYS,
    OVERRIDABLE_PIPELINE_KEYS,
    OVERRIDABLE_SECTIONS,
    OverrideRejected,
    get_state,
    pending_restart,
    set_overrides,
)
from .auth import require_api_key

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/settings", dependencies=[Depends(require_api_key)])


class EndpointIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "openrouter", "anthropic"]
    model: str = Field(min_length=1, max_length=200)
    api_base: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    #: Unset lets the provider choose, and providers choose badly: OpenRouter cut
    #: tier-1 off mid-JSON on every call, which costs a full request and returns
    #: nothing usable. Editable here so a truncating model can be fixed without a
    #: config-file edit.
    max_tokens: int | None = Field(default=None, ge=1, le=200_000)
    extra_params: dict[str, Any] = Field(default_factory=dict)


class TierIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: EndpointIn
    fallbacks: list[EndpointIn] = Field(default_factory=list)
    timeout_s: int = Field(ge=5, le=3600)
    validation_retries: int = Field(default=2, ge=0, le=5)
    allow_cloud_fallback: bool = True


class InterestIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)
    label: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    weight: int = Field(ge=1, le=10)


class SettingsIn(BaseModel):
    """Everything the console may change. Omitted sections are left alone."""

    model_config = ConfigDict(extra="forbid")

    tiers: dict[str, TierIn] | None = None
    pipeline: dict[str, int] | None = None
    interest_profile: list[InterestIn] | None = None
    asr: dict[str, Any] | None = None
    backfill: dict[str, Any] | None = None


def _asr_installed() -> bool:
    """Whether faster-whisper is importable in this process.

    The package is an optional extra, so local transcription can be configured
    on an install that cannot perform it. Better said on the settings page than
    discovered when an episode fails.
    """
    from importlib.util import find_spec

    return find_spec("faster_whisper") is not None


def _store(request: Request) -> Store:
    return request.app.state.store  # type: ignore[no-any-return]


def _settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _baseline(request: Request) -> Settings:
    """Configuration as the file and environment declare it, before overrides.

    What an override is judged against, so a stored one cannot widen the set of
    hosts a later override is allowed to use.
    """
    return getattr(request.app.state, "baseline_settings", None) or _settings(request)


def _endpoint_view(endpoint: Any) -> dict[str, Any]:
    return {
        "provider": endpoint.provider.value,
        "model": endpoint.model,
        "api_base": endpoint.api_base,
        "temperature": endpoint.temperature,
        "max_tokens": endpoint.max_tokens,
        "extra_params": endpoint.extra_params,
        "is_cloud": endpoint.is_cloud,
        "litellm_model": endpoint.litellm_model(),
    }


@router.get("", summary="Editable configuration, as this process is running it")
async def read_settings(request: Request) -> dict[str, Any]:
    settings = _settings(request)
    state = await get_state(_store(request))

    return {
        "tiers": {
            name: {
                "primary": _endpoint_view(tier.primary),
                "fallbacks": [_endpoint_view(e) for e in tier.fallbacks],
                "timeout_s": tier.timeout_s,
                "validation_retries": tier.validation_retries,
                "allow_cloud_fallback": tier.allow_cloud_fallback,
                # What would actually be tried, in order, right now.
                "active_chain": [e.litellm_model() for e in tier.active_chain()],
            }
            for name, tier in settings.llm.tiers.items()
        },
        "pipeline": {
            key: getattr(settings.pipeline, key) for key in sorted(OVERRIDABLE_PIPELINE_KEYS)
        },
        "asr": {key: getattr(settings.asr, key) for key in sorted(OVERRIDABLE_ASR_KEYS)},
        "backfill": {
            key: getattr(settings.backfill, key) for key in sorted(OVERRIDABLE_BACKFILL_KEYS)
        },
        # Fixed here rather than in the browser: these protect the machine.
        "asr_fixed": {
            "max_audio_mb": settings.asr.max_audio_mb,
            "asr_concurrency": settings.asr.asr_concurrency,
            "download_concurrency": settings.asr.download_concurrency,
            "remote_url": settings.asr.remote_url,
        },
        "asr_installed": _asr_installed(),
        "interest_profile": [
            {
                "key": i.key,
                "label": i.label,
                "description": i.description,
                "weight": i.weight,
            }
            for i in settings.interest_profile
        ],
        "interest_profile_version": settings.interest_profile_version(),
        "editable_sections": sorted(OVERRIDABLE_SECTIONS),
        "editable_pipeline_keys": sorted(OVERRIDABLE_PIPELINE_KEYS),
        "overrides": state["overrides"],
        "pending_restart": pending_restart(state),
        "updated_at": state["updated_at"],
        "applied_at": state["applied_at"],
        # So the page can say *which* process is out of date, and since when. A
        # restart aimed at the wrong thing leaves the banner up with no way to
        # tell it apart from a restart that never happened.
        "started_at": getattr(request.app.state, "started_at", None),
        "has_openrouter_key": settings.openrouter_api_key is not None,
        "has_anthropic_key": settings.anthropic_api_key is not None,
    }


@router.put("", summary="Replace the console-editable configuration")
async def write_settings(request: Request, body: Annotated[SettingsIn, Body()]) -> dict[str, Any]:
    """Validated exactly as a config file is, before anything is stored.

    A change that would refuse to boot is rejected here instead — otherwise the
    next restart is the moment you find out.
    """
    state = await get_state(_store(request))
    overrides = dict(state["overrides"])

    if body.tiers is not None:
        overrides.setdefault("llm", {})["tiers"] = {
            name: tier.model_dump(exclude_none=False) for name, tier in body.tiers.items()
        }
    if body.pipeline is not None:
        overrides["pipeline"] = {**overrides.get("pipeline", {}), **body.pipeline}
    if body.interest_profile is not None:
        overrides["interest_profile"] = [i.model_dump() for i in body.interest_profile]
    if body.asr is not None:
        overrides["asr"] = {**overrides.get("asr", {}), **body.asr}
    if body.backfill is not None:
        overrides["backfill"] = {**overrides.get("backfill", {}), **body.backfill}

    try:
        # Dry-run against the configuration this process is actually running,
        # rather than re-reading the file: the running config is what the change
        # will be layered onto, and it already has env-supplied values applied.
        _validate_against(_settings(request), overrides, baseline=_baseline(request))
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_readable(exc),
        ) from exc
    except OverrideRejected as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        new_state = await set_overrides(_store(request), overrides)
    except OverrideRejected as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    log.info("settings.updated", sections=sorted(overrides))
    return {
        "saved": True,
        "pending_restart": pending_restart(new_state),
        "detail": "Saved. Restart the service to apply.",
    }


@router.delete("", summary="Discard all overrides and return to config.yaml")
async def reset_settings(request: Request) -> dict[str, Any]:
    new_state = await set_overrides(_store(request), {})
    return {
        "saved": True,
        "pending_restart": pending_restart(new_state),
        "detail": "Overrides cleared. Restart to return to config.yaml.",
    }


#: Secrets are excluded from the round-trip: they are SecretStr in the dump and
#: come back from the environment on re-validation anyway.
_SECRET_FIELDS = frozenset(
    {"admin_api_key", "couchdb_password", "openrouter_api_key", "anthropic_api_key", "ntfy_token"}
)


def _validate_against(settings: Settings, overrides: dict[str, Any], *, baseline: Settings) -> None:
    """Rebuild the whole configuration with the overrides applied.

    Raises exactly what a bad config file would, so a change that could not boot
    is refused now rather than at the next restart. The endpoint check is judged
    against ``baseline`` rather than the running settings — see
    :func:`podcast_agent.settings_store.allowed_api_base_hosts`.
    """
    from ..settings_store import allowed_api_base_hosts, check_api_bases, deep_merge

    base = settings.model_dump(mode="json", exclude=set(_SECRET_FIELDS))
    candidate = Settings(**deep_merge(base, overrides))
    check_api_bases(candidate, allowed_api_base_hosts(baseline))


def _readable(exc: ValidationError) -> str:
    """Config errors should read like the startup message, not like a stack trace."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(p) for p in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts[:6])
