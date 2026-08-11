"""Tier-0 routing: the model proposes, code disposes (§4 stage 2).

Kept as a pure function with no I/O so the routing table is exhaustively
unit-testable — this is the decision that determines whether an episode is
silently discarded, so it must never depend on model free text.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import PipelineConfig
from ..models import Route, Tier0Result


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    route: Route
    #: Machine-readable rule that fired, for logs and tests.
    rule: str

    def __str__(self) -> str:
        return f"{self.route.value} ({self.rule})"


def decide_route(
    result: Tier0Result,
    cfg: PipelineConfig,
    *,
    always_escalate: bool = False,
) -> RoutingDecision:
    """Choose the route from validated numeric fields only.

    Rules, in precedence order (§4):

    1. ``always_escalate`` show     -> ESCALATE (per-show override)
    2. confidence < ``t_conf_high`` -> ESCALATE (description too thin to judge —
       the core requirement: never drop an episode on a vague description)
    3. relevance < ``t_rel_low``    -> DROP     (confidently irrelevant)
    4. relevance >= ``t_rel_high``  -> ESCALATE (confidently relevant: the owner
       reads summaries, so relevant episodes always get Tier-1 treatment)
    5. otherwise                    -> DIGEST_DIRECT (grey zone: a one-line
       "maybe interesting" entry, so nothing relevant is silently lost)

    ``result.route`` — the model's own suggestion — is deliberately ignored.
    """
    if always_escalate:
        return RoutingDecision(Route.ESCALATE, "always_escalate")
    if result.confidence < cfg.t_conf_high:
        return RoutingDecision(Route.ESCALATE, "low_confidence")
    if result.relevance_guess < cfg.t_rel_low:
        return RoutingDecision(Route.DROP, "confident_irrelevant")
    if result.relevance_guess >= cfg.t_rel_high:
        return RoutingDecision(Route.ESCALATE, "confident_relevant")
    return RoutingDecision(Route.DIGEST_DIRECT, "grey_zone")
