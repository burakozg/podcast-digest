"""Test helpers importable from any test module (no fixtures here)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import podcast_agent.config as config_module
from podcast_agent.backfill import ROUTINE_ORIGIN
from podcast_agent.config import Settings
from podcast_agent.models import CallMeta, ChunkBullets, Tier0Result, Tier1Result
from podcast_agent.state import EpisodeStatus
from podcast_agent.utils import episode_doc_id, iso

FIXTURES = Path(__file__).parent / "fixtures"

#: Pass as an override value to omit a key entirely, so the real settings sources
#: (.env, environment) supply it instead of the test default.
DROP = object()


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Build a fully-explicit Settings, ignoring the repo's config.yaml."""
    # Point the YAML source at a non-existent file: init kwargs are the only input.
    config_module._active_yaml_path = tmp_path / "no-such-config.yaml"

    data: dict[str, Any] = {
        "podcasts": [
            {
                "slug": "test-show",
                "name": "Test Show",
                "feed_url": "https://example.com/feed.xml",
                "priority": "med",
                "always_escalate": False,
                # Most tests exercise the transcribe-then-summarise path, so the
                # fixture show opts into ASR explicitly. Shows default to off.
                "asr_enabled": True,
                # Likewise the archive: podcasts default to `skip`, so a fixture
                # used by archive tests has to opt in the way a person would.
                "backfill_mode": "full",
            },
            {
                "slug": "priority-show",
                "name": "Priority Show",
                "feed_url": "https://priority-show.net/feed.xml",
                "priority": "high",
                "always_escalate": True,
                "backfill_mode": "full",
            },
        ],
        "interest_profile": [
            {
                "key": "ot_ics",
                "label": "OT/ICS security",
                "description": "ICS, SCADA, PLC",
                "weight": 10,
            },
            {
                "key": "ai_agent_security",
                "label": "AI & agent security",
                "description": "LLM and agent security",
                "weight": 9,
            },
        ],
        "llm": {
            "tiers": {
                "tier0": {
                    "primary": {"provider": "ollama", "model": "test-small"},
                    "timeout_s": 30,
                },
                "tier1": {
                    "primary": {"provider": "ollama", "model": "test-large"},
                    "timeout_s": 60,
                },
            }
        },
        "output": {"digest_dir": tmp_path / "digests", "work_dir": tmp_path / "work"},
        "security": {
            "enforce_domain_allowlist": True,
            "cdn_allowlist": ["cdn-host.net", "transcript-host.net"],
        },
        "admin_api_key": "test-admin-key",
        "logging": {"level": "WARNING", "format": "console"},
    }
    for key, value in overrides.items():
        if value is DROP:
            data.pop(key, None)
        elif isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return Settings(**data)


class FakeLLM:
    """Stands in for :class:`StructuredLLM` at its exact boundary.

    Either supply a ``handler(tier, system, user, response_model)`` or rely on the
    defaults, which return a fixed plausible result per response model.
    """

    def __init__(self, handler: Any = None) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None

    async def complete_structured(
        self,
        tier: str,
        system: str,
        user: str,
        response_model: type[Any],
        *,
        episode_id: str | None = None,
        prompt_version: str = "",
    ) -> tuple[Any, CallMeta]:
        self.calls.append(
            {
                "tier": tier,
                "system": system,
                "user": user,
                "response_model": response_model.__name__,
                "episode_id": episode_id,
                "prompt_version": prompt_version,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        result = (
            self._handler(tier, system, user, response_model)
            if self._handler
            else default_result(response_model)
        )
        meta = CallMeta(
            tier=tier,
            provider="ollama",
            model="test-model",
            latency_ms=42,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0,
            prompt_version=prompt_version,
            episode_id=episode_id,
        )
        return result, meta


def default_result(response_model: type[Any]) -> Any:
    if response_model is Tier0Result:
        return Tier0Result(
            relevance_guess=8,
            confidence=9,
            matched_interests=["ot_ics"],
            reasoning="Clearly about ICS security.",
        )
    if response_model is Tier1Result:
        return Tier1Result(
            relevance_score=8,
            matched_interests=["ot_ics"],
            why_it_matters="Directly relevant to your OT/ICS work.",
            summary_md="The hosts walk through a **PLC** compromise at a water utility.",
            key_takeaways=["Segment OT networks", "Patch the HMI"],
            entities=["Modbus", "CVE-2026-1234"],
            listen_anyway=False,
        )
    if response_model is ChunkBullets:
        return ChunkBullets(bullets=["A point from this slice"], entities=["Modbus"])
    raise AssertionError(f"FakeLLM has no default for {response_model.__name__}")


def make_episode(
    *,
    slug: str = "test-show",
    guid: str = "guid-1",
    title: str = "Episode about ICS malware",
    status: EpisodeStatus = EpisodeStatus.NEW,
    published_at: datetime | None = None,
    description: str = "A detailed discussion of PLC malware in water utilities.",
    duration_s: int | None = 3720,
    **extra: Any,
) -> dict[str, Any]:
    published = published_at or datetime(2026, 7, 28, 10, 0, tzinfo=UTC)
    doc: dict[str, Any] = {
        "_id": episode_doc_id(slug, guid),
        "type": "episode",
        "podcast_slug": slug,
        "podcast_name": "Test Show" if slug == "test-show" else "Priority Show",
        "guid": guid,
        "title": title,
        "link": "https://example.com/ep1",
        "description_raw": description,
        "published_at": iso(published),
        "enclosure_url": "https://cdn-host.net/ep1.mp3",
        "enclosure_type": "audio/mpeg",
        "enclosure_bytes": 40_000_000,
        "duration_s": duration_s,
        "feed_transcripts": [],
        "status": status.value,
        # Shaped as ingestion writes it. Selectors match `origin` on equality,
        # so a helper that omitted it would build episodes the pipeline cannot
        # see — a test double diverging from the real document again.
        "origin": ROUTINE_ORIGIN,
        "tier0": None,
        "tier1": None,
        "transcript_source": "none",
        "digest_id": None,
        "attempts": {"transcript": 0, "tier0": 0, "tier1": 0},
        "last_error": None,
        "created_at": iso(published),
        "updated_at": iso(published),
    }
    doc.update(extra)
    return doc


def days_ago(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)
