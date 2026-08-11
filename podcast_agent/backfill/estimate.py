"""Pre-flight cost estimate for archive backfill (roadmap A1).

The roadmap is explicit that cost guardrails matter far more here than in
steady state, and that a run should require an explicit confirmation. This
module answers "what would this actually cost me?" using *this deployment's own*
telemetry rather than generic guesses — the whole point of recording per-call
latency and cost since v1.

With no telemetry yet it says so, rather than inventing a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..db import Store, typed_sort
from ..logging_setup import get_logger

log = get_logger(__name__)

#: Fallback used only when a tier has no recorded calls at all.
_NO_HISTORY = "no telemetry yet"


@dataclass(slots=True)
class TierCost:
    calls: int
    median_latency_ms: int
    mean_cost_usd: float
    samples: int

    @property
    def known(self) -> bool:
        return self.samples > 0


async def _tier_history(store: Store, tier: str, limit: int = 500) -> TierCost:
    docs = await store.find(
        {"type": "llm_call", "tier": tier}, sort=typed_sort("ts", "desc"), limit=limit
    )
    if not docs:
        return TierCost(calls=0, median_latency_ms=0, mean_cost_usd=0.0, samples=0)
    latencies = sorted(int(d.get("latency_ms") or 0) for d in docs)
    costs = [float(d.get("cost_usd") or 0.0) for d in docs]
    return TierCost(
        calls=0,
        median_latency_ms=latencies[len(latencies) // 2],
        mean_cost_usd=sum(costs) / len(costs),
        samples=len(docs),
    )


async def estimate_backfill(
    settings: Settings,
    store: Store,
    *,
    episodes_to_ingest: int,
    tier0_only_share: float,
    without_transcript_share: float = 0.0,
) -> dict[str, Any]:
    """Project the cost of processing ``episodes_to_ingest`` archive episodes.

    ``tier0_only_share`` is the fraction belonging to podcasts configured
    ``backfill_mode: tier0_only``, which never reach Tier-1.

    ``without_transcript_share`` is the fraction that can reach neither a
    published transcript nor local transcription. Every episode is indexed and
    triaged, but these cannot be summarised — an escalation is downgraded to an
    index entry rather than spending a Tier-1 call on a description. Ignoring
    that would badly over-estimate the walk.
    """
    tier0 = await _tier_history(store, "tier0")
    tier1 = await _tier_history(store, "tier1")

    # Every ingested episode is triaged.
    tier0_calls = episodes_to_ingest
    # Only full-mode episodes can reach Tier-1, and only those Tier-0 escalates.
    # Escalation rate is measured where possible, otherwise assumed generous.
    escalation_rate = await _measured_escalation_rate(store)
    reachable = episodes_to_ingest * (1.0 - tier0_only_share)
    reachable *= 1.0 - without_transcript_share
    tier1_calls = int(reachable * escalation_rate)

    tier0_ms = tier0_calls * tier0.median_latency_ms
    tier1_ms = tier1_calls * tier1.median_latency_ms
    total_hours = (tier0_ms + tier1_ms) / 3_600_000

    projected_cost = tier0_calls * tier0.mean_cost_usd + tier1_calls * tier1.mean_cost_usd

    return {
        "episodes": episodes_to_ingest,
        "tier0_calls": tier0_calls,
        "tier1_calls": tier1_calls,
        "escalation_rate": round(escalation_rate, 3),
        "tier0_only_share": round(tier0_only_share, 3),
        "without_transcript_share": round(without_transcript_share, 3),
        # Indexed and triaged; nothing to summarise from.
        "indexed_only": int(episodes_to_ingest * without_transcript_share),
        # Whatever the walk cannot get a published transcript for, and is
        # allowed to transcribe, becomes an ASR job. Which podcasts those are is
        # a per-podcast decision, so this is not a single number.
        "asr_jobs": "per podcast — see the Transcribe locally column",
        "estimated_wall_hours": round(total_hours, 2) if tier0.known else _NO_HISTORY,
        "estimated_cost_usd": round(projected_cost, 4) if tier0.known else _NO_HISTORY,
        "basis": {
            "tier0_median_latency_ms": tier0.median_latency_ms if tier0.known else _NO_HISTORY,
            "tier1_median_latency_ms": tier1.median_latency_ms if tier1.known else _NO_HISTORY,
            "tier0_samples": tier0.samples,
            "tier1_samples": tier1.samples,
        },
        "note": (
            "Estimated from this deployment's own telemetry. "
            "Local models cost nothing but time; the wall-hours figure is the "
            "one that matters."
            if tier0.known
            else "No LLM calls recorded yet, so no estimate is possible. Run the "
            "normal pipeline once first."
        ),
    }


async def _measured_escalation_rate(store: Store, default: float = 0.5) -> float:
    """Fraction of triaged episodes that Tier-0 sent for a full summary."""
    docs = await store.find(
        {"type": "episode", "status": {"$exists": True}},
        sort=typed_sort("published_at", "desc"),
        limit=500,
    )
    routed = [d for d in docs if (d.get("tier0") or {}).get("route")]
    if len(routed) < 10:
        return default
    escalated = sum(1 for d in routed if d["tier0"]["route"] == "ESCALATE")
    return escalated / len(routed)
