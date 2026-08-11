"""Tier-0 routing tests.

This is the decision that can silently discard an episode, so every rule and
boundary is pinned, including the requirement that the model's own suggested
route is ignored (§4).
"""

from __future__ import annotations

import pytest

from podcast_agent.config import PipelineConfig
from podcast_agent.models import Route, Tier0Result
from podcast_agent.triage.routing import decide_route


@pytest.fixture
def cfg() -> PipelineConfig:
    # Defaults from §4: T_conf_high=7, T_rel_low=4, T_rel_high=7.
    return PipelineConfig()


def result(relevance: int, confidence: int, **kw: object) -> Tier0Result:
    return Tier0Result(relevance_guess=relevance, confidence=confidence, **kw)  # type: ignore[arg-type]


def test_low_confidence_always_escalates(cfg: PipelineConfig) -> None:
    """The core requirement: a thin description is never a reason to drop."""
    decision = decide_route(result(relevance=0, confidence=6), cfg)
    assert decision.route is Route.ESCALATE
    assert decision.rule == "low_confidence"


def test_confident_and_irrelevant_drops(cfg: PipelineConfig) -> None:
    decision = decide_route(result(relevance=3, confidence=9), cfg)
    assert decision.route is Route.DROP
    assert decision.rule == "confident_irrelevant"


def test_confident_and_relevant_escalates(cfg: PipelineConfig) -> None:
    """Relevant episodes always get a real summary — the owner reads summaries."""
    decision = decide_route(result(relevance=8, confidence=8), cfg)
    assert decision.route is Route.ESCALATE
    assert decision.rule == "confident_relevant"


def test_grey_zone_becomes_digest_direct(cfg: PipelineConfig) -> None:
    """Nothing possibly-relevant is silently lost; it becomes a one-liner."""
    decision = decide_route(result(relevance=5, confidence=9), cfg)
    assert decision.route is Route.DIGEST_DIRECT
    assert decision.rule == "grey_zone"


def test_always_escalate_overrides_everything(cfg: PipelineConfig) -> None:
    decision = decide_route(result(relevance=0, confidence=10), cfg, always_escalate=True)
    assert decision.route is Route.ESCALATE
    assert decision.rule == "always_escalate"


def test_model_suggested_route_is_ignored(cfg: PipelineConfig) -> None:
    """The model proposes, code disposes — a hostile description cannot route itself."""
    hostile = result(relevance=3, confidence=9, route=Route.ESCALATE)
    assert decide_route(hostile, cfg).route is Route.DROP

    other = result(relevance=9, confidence=9, route=Route.DROP)
    assert decide_route(other, cfg).route is Route.ESCALATE


@pytest.mark.parametrize(
    ("relevance", "confidence", "expected"),
    [
        # Confidence boundary: 7 is "high enough" (>=), 6 is not.
        (0, 7, Route.DROP),
        (0, 6, Route.ESCALATE),
        # Relevance lower boundary: <4 drops, ==4 is grey zone.
        (3, 10, Route.DROP),
        (4, 10, Route.DIGEST_DIRECT),
        # Relevance upper boundary: 6 is grey zone, 7 escalates (>=).
        (6, 10, Route.DIGEST_DIRECT),
        (7, 10, Route.ESCALATE),
        # Extremes.
        (10, 10, Route.ESCALATE),
        (0, 0, Route.ESCALATE),
    ],
)
def test_threshold_boundaries(
    cfg: PipelineConfig, relevance: int, confidence: int, expected: Route
) -> None:
    assert decide_route(result(relevance, confidence), cfg).route is expected


def test_custom_thresholds_are_honoured() -> None:
    strict = PipelineConfig(t_conf_high=9, t_rel_low=6, t_rel_high=9)
    # Confidence 8 would pass the default but not this config.
    assert decide_route(result(relevance=7, confidence=8), strict).route is Route.ESCALATE
    # Relevance 5 is below the raised floor.
    assert decide_route(result(relevance=5, confidence=10), strict).route is Route.DROP
    assert decide_route(result(relevance=7, confidence=10), strict).route is Route.DIGEST_DIRECT


def test_routing_is_total() -> None:
    """Every (relevance, confidence) pair must produce a route — no gaps."""
    cfg = PipelineConfig()
    for relevance in range(11):
        for confidence in range(11):
            decision = decide_route(result(relevance, confidence), cfg)
            assert decision.route in set(Route)
            assert decision.rule
