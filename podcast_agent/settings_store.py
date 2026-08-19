"""Database-backed overrides for configuration (console-editable settings).

Same constraint as the show list: `config.yaml` is mounted read-only, so a
console that can change which model runs needs somewhere writable. The file stays
the declared baseline; this holds a deep-merged override on top.

**Overrides apply at startup, not immediately.** Swapping a provider rebuilds the
LLM router, and changing the interest profile changes what every cached prompt
says — doing that underneath a summarisation already in flight is a bad trade for
a setting that changes a few times a year. The console reports pending changes
and the restart needed to apply them.

Only a deliberate subset is overridable. Storage, network binding and the
security allowlist are not: a typo in a browser should not be able to make the
service unreachable or unable to find its own database.
"""

from __future__ import annotations

import copy
from typing import Any
from urllib.parse import urlparse

from .config import Provider, Settings
from .db import Doc, Store
from .logging_setup import get_logger
from .utils import iso_now

log = get_logger(__name__)

SETTINGS_DOC_ID = "control:settings"

#: Top-level config sections the console may override.
#:
#: Deliberately excludes couchdb (lock yourself out of storage), api (lock
#: yourself out of the console), security (disable the fetch guards), output
#: (point digests at a path the container cannot write) and scheduler (a bad
#: cron expression refuses to boot).
OVERRIDABLE_SECTIONS = frozenset(
    {
        "llm",
        "pipeline",
        "interest_profile",
        "asr",
        "tts",
        "notifications",
        "backfill",
        "content",
    }
)

#: Within `asr`, the knobs worth changing without editing a file: which model
#: runs and how it runs. `max_audio_mb` and the concurrency caps stay in the file
#: because they protect the machine rather than express a preference, and
#: `remote_url` is deployment topology.
OVERRIDABLE_ASR_KEYS = frozenset(
    {"backend", "model", "compute_type", "device", "language", "beam_size", "keep_audio"}
)

#: Within `tts`, the knobs that express a preference. `base_url` is excluded for
#: the same reason `asr.remote_url` is — it is deployment topology, it comes from
#: the environment, and console overrides deliberately rank below that.
OVERRIDABLE_TTS_KEYS = frozenset({"enabled", "model", "voice", "speed", "response_format"})

#: Within `backfill`, what the archive walk produces. Transcription is not here:
#: it is a per-podcast toggle on the Podcasts page, not a global switch.
OVERRIDABLE_BACKFILL_KEYS = frozenset({"digest_threshold"})

#: Within `pipeline`, only the tuning knobs. Batch caps stay in the file because
#: they protect the machine rather than express a preference.
OVERRIDABLE_PIPELINE_KEYS = frozenset(
    {
        "t_conf_high",
        "t_rel_low",
        "t_rel_high",
        "digest_threshold",
        "top_pick_threshold",
        "initial_lookback_days",
        "digest_catch_up_days",
        "weekly_synthesis",
        "max_input_tokens",
        "chunk_target_tokens",
    }
)


class OverrideRejected(ValueError):
    """The override targets something the console is not allowed to change."""


def validate_overrides(overrides: dict[str, Any]) -> None:
    """Raise unless every path in ``overrides`` is permitted."""
    unknown = set(overrides) - OVERRIDABLE_SECTIONS
    if unknown:
        raise OverrideRejected(
            f"not overridable from the console: {', '.join(sorted(unknown))}. "
            f"Editable sections are {', '.join(sorted(OVERRIDABLE_SECTIONS))}."
        )
    pipeline = overrides.get("pipeline") or {}
    bad_keys = set(pipeline) - OVERRIDABLE_PIPELINE_KEYS
    if bad_keys:
        raise OverrideRejected(
            f"pipeline keys not overridable: {', '.join(sorted(bad_keys))}. "
            "Batch caps stay in config.yaml because they protect the machine."
        )
    backfill = overrides.get("backfill") or {}
    bad_backfill = set(backfill) - OVERRIDABLE_BACKFILL_KEYS
    if bad_backfill:
        raise OverrideRejected(
            f"backfill keys not overridable: {', '.join(sorted(bad_backfill))}. "
            "The window is per-podcast, and pacing stays in config.yaml."
        )
    asr = overrides.get("asr") or {}
    bad_asr = set(asr) - OVERRIDABLE_ASR_KEYS
    if bad_asr:
        raise OverrideRejected(
            f"asr keys not overridable: {', '.join(sorted(bad_asr))}. "
            "Size caps, concurrency and remote_url stay in config.yaml."
        )
    tts = overrides.get("tts") or {}
    bad_tts = set(tts) - OVERRIDABLE_TTS_KEYS
    if bad_tts:
        raise OverrideRejected(
            f"tts keys not overridable: {', '.join(sorted(bad_tts))}. "
            "base_url, chunking and the timeout stay in config.yaml."
        )


#: Where a provider sends work when no ``api_base`` is given. Included in the
#: allowed set unconditionally: they are the vendors' own endpoints, which is
#: what choosing that provider already means.
_PROVIDER_DEFAULT_HOSTS: dict[Provider, str] = {
    Provider.OLLAMA: "localhost",
    Provider.OPENROUTER: "openrouter.ai",
    Provider.ANTHROPIC: "api.anthropic.com",
}

#: Always permitted: the Ollama default, in the forms a person writes it.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def allowed_api_base_hosts(baseline: Settings) -> set[str]:
    """Hosts an override may point a model endpoint at.

    ``baseline`` must be the configuration *before* stored overrides are
    applied — the file plus the environment. Deriving the allowed set from the
    running configuration instead would let a stored override authorise itself:
    once applied at boot it would appear in the baseline and be permitted from
    then on.

    Reaching a new destination therefore takes file-level authority. That is the
    point: `api_base` decides where every prompt and transcript is sent, so a
    console key alone must not be able to add one.
    """
    hosts = set(_LOOPBACK_HOSTS) | set(_PROVIDER_DEFAULT_HOSTS.values())
    for tier in baseline.llm.tiers.values():
        for endpoint in (tier.primary, *tier.fallbacks):
            if endpoint.api_base and (host := urlparse(endpoint.api_base).hostname):
                hosts.add(host)
    return hosts


def check_api_bases(candidate: Settings, allowed: set[str]) -> None:
    """Raise unless every endpoint in ``candidate`` points somewhere allowed."""
    for name, tier in candidate.llm.tiers.items():
        for endpoint in (tier.primary, *tier.fallbacks):
            if not endpoint.api_base:
                continue
            host = urlparse(endpoint.api_base).hostname
            if host not in allowed:
                raise OverrideRejected(
                    f"llm.tiers.{name}: {host!r} is not a configured model host, so it "
                    f"cannot be set from the console. Known hosts are "
                    f"{', '.join(sorted(allowed))}. To use a new one, add it to "
                    "config.yaml and restart — where model work is sent is a "
                    "deployment decision, not a console setting."
                )


#: Nodes whose *children* replace their baseline rather than merging into it.
#:
#: An LLM tier describes one provider's endpoint, and merging it onto a
#: different provider's produces a chimera: the console stored an OpenRouter
#: primary, the file declared an Ollama one, and the merge handed OpenRouter
#: Ollama's `api_base` and its `num_ctx`/`num_predict` — which OpenRouter
#: ignores, so tier-1 ran with no reply limit and truncated every call. The
#: console always writes a whole tier (`TierIn` dumps every field, including
#: nulls), so there is nothing in the baseline worth keeping underneath it.
_REPLACE_CHILDREN_OF: frozenset[tuple[str, ...]] = frozenset({("llm", "tiers")})


def deep_merge(
    base: dict[str, Any], overlay: dict[str, Any], _path: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Merge ``overlay`` onto ``base``.

    Dictionaries merge key by key; lists replace wholesale, because a partially
    merged interest profile would be nonsense — you mean the list you supplied.
    Children of :data:`_REPLACE_CHILDREN_OF` replace wholesale for the same
    reason: a half-overridden endpoint is not a configuration anyone wrote.
    """
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        mergeable = isinstance(value, dict) and isinstance(merged.get(key), dict)
        if mergeable and _path not in _REPLACE_CHILDREN_OF:
            merged[key] = deep_merge(merged[key], value, (*_path, key))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


async def get_overrides(store: Store) -> dict[str, Any]:
    doc = await store.get(SETTINGS_DOC_ID)
    if doc is None:
        return {}
    return dict(doc.get("overrides") or {})


async def get_state(store: Store) -> dict[str, Any]:
    doc = await store.get(SETTINGS_DOC_ID)
    return {
        "overrides": dict((doc or {}).get("overrides") or {}),
        "updated_at": (doc or {}).get("updated_at"),
        "applied_at": (doc or {}).get("applied_at"),
    }


async def set_overrides(store: Store, overrides: dict[str, Any]) -> dict[str, Any]:
    """Replace the stored overrides wholesale, after validating the paths."""
    validate_overrides(overrides)
    existing = await store.get(SETTINGS_DOC_ID)
    doc: Doc = {
        "_id": SETTINGS_DOC_ID,
        "type": "control",
        "key": "settings",
        "overrides": overrides,
        "updated_at": iso_now(),
        # Preserved so the console can tell "changed since this process booted".
        "applied_at": (existing or {}).get("applied_at"),
    }
    if existing:
        doc["_rev"] = existing["_rev"]
    await store.put(doc)
    log.info("settings.overrides_set", sections=sorted(overrides))
    return await get_state(store)


async def mark_applied(store: Store) -> None:
    """Record that the running process booted with the current overrides."""
    doc = await store.get(SETTINGS_DOC_ID)
    if doc is None:
        return
    doc["applied_at"] = doc.get("updated_at")
    await store.put(doc)


def pending_restart(state: dict[str, Any]) -> bool:
    """True when overrides have changed since this process last applied them."""
    updated, applied = state.get("updated_at"), state.get("applied_at")
    return bool(updated and updated != applied)
