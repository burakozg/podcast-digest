"""Tests for the real LLMClient fallback/retry machinery (§7).

The pipeline tests fake the LLM at the ``complete_structured`` boundary, which
left this module — the part with the actual failover logic — entirely untested.
Two production bugs escaped that way:

  * ``instructor.exceptions`` is not a package attribute, so the except clause
    raised AttributeError *while handling* the real error and masked it;
  * instructor wraps transport failures in ``InstructorRetryException`` too, so
    timeouts were counted as validation failures and retried against a dead
    endpoint instead of failing over.

Nothing here touches the network: the instructor client is replaced with a stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import instructor
import litellm
import pytest
from helpers import make_settings
from instructor.core import InstructorRetryException
from litellm.types.utils import Choices, ModelResponse
from litellm.types.utils import Message as LiteLLMMessage
from pydantic import ValidationError

from podcast_agent.db import MemoryStore
from podcast_agent.llm.base import LLMUnavailable
from podcast_agent.llm.client import (
    ENDPOINT_COOLDOWN_S,
    INSTRUCTOR_MODE,
    LLMClient,
    _root_cause,
)
from podcast_agent.models import Tier0Result


class _Usage:
    prompt_tokens = 120
    completion_tokens = 34


class _RawResponse:
    model = "test-model"
    usage = _Usage()


class StubCompletions:
    """Stands in for instructor's chat.completions, scripted per call."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create_with_completion(self, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else self.outcomes
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, _RawResponse()


def install_stub(client: LLMClient, outcomes: list[Any]) -> StubCompletions:
    stub = StubCompletions(outcomes)
    client._instructor = type(  # type: ignore[assignment]
        "StubInstructor", (), {"chat": type("Chat", (), {"completions": stub})()}
    )()
    return stub


def two_endpoint_settings(tmp_path: Path, **over: Any):
    """tier0 with a local primary and a cloud fallback."""
    llm = {
        "tiers": {
            "tier0": {
                "primary": {"provider": "ollama", "model": "local-small"},
                "fallbacks": [{"provider": "openrouter", "model": "vendor/remote"}],
                "validation_retries": 2,
                **over,
            },
            "tier1": {"primary": {"provider": "ollama", "model": "local-big"}},
        }
    }
    return make_settings(tmp_path, llm=llm, openrouter_api_key="sk-test")


def valid_result() -> Tier0Result:
    return Tier0Result(relevance_guess=8, confidence=9)


def a_timeout() -> Exception:
    return litellm.Timeout(message="timed out", model="m", llm_provider="ollama")


def a_validation_error() -> ValidationError:
    try:
        Tier0Result(relevance_guess=99, confidence=0)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def wrapped(cause: BaseException) -> InstructorRetryException:
    """How instructor actually surfaces a failure: wrapped, with __cause__ set."""
    exc = InstructorRetryException(str(cause), n_attempts=1, total_usage=0)
    exc.__cause__ = cause
    return exc


class TestRootCause:
    def test_unwraps_instructor_wrapper(self) -> None:
        cause = a_timeout()
        assert _root_cause(wrapped(cause)) is cause

    def test_returns_plain_exception_unchanged(self) -> None:
        exc = ValueError("plain")
        assert _root_cause(exc) is exc

    def test_survives_a_self_referential_cause(self) -> None:
        exc = InstructorRetryException("looping", n_attempts=1, total_usage=0)
        exc.__cause__ = exc
        assert _root_cause(exc) is exc  # terminates instead of looping forever


class TestSuccessPath:
    async def test_returns_result_and_telemetry(self, tmp_path: Path) -> None:
        store = MemoryStore()
        client = LLMClient(two_endpoint_settings(tmp_path), store)
        install_stub(client, [valid_result()])

        result, meta = await client.complete_structured(
            "tier0", "sys", "user", Tier0Result, episode_id="episode:1", prompt_version="tier0_v1"
        )

        assert result.relevance_guess == 8
        assert meta.provider == "ollama"
        assert meta.fallback_used is False
        assert meta.validation_retries == 0
        assert meta.input_tokens == 120
        assert meta.output_tokens == 34
        assert meta.cost_usd == 0.0  # local model is free
        assert meta.prompt_version == "tier0_v1"

    async def test_writes_a_telemetry_document(self, tmp_path: Path) -> None:
        store = MemoryStore()
        client = LLMClient(two_endpoint_settings(tmp_path), store)
        install_stub(client, [valid_result()])
        await client.complete_structured("tier0", "s", "u", Tier0Result, episode_id="episode:1")

        calls = store.docs_of_type("llm_call")
        assert len(calls) == 1
        assert calls[0]["tier"] == "tier0"
        assert calls[0]["episode_id"] == "episode:1"
        assert calls[0]["provider"] == "ollama"

    async def test_targets_the_primary_deployment_first(self, tmp_path: Path) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(client, [valid_result()])
        await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert stub.calls[0]["model"] == "tier0"

    async def test_sends_system_and_user_messages(self, tmp_path: Path) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(client, [valid_result()])
        await client.complete_structured("tier0", "SYSTEM TEXT", "USER TEXT", Tier0Result)
        messages = stub.calls[0]["messages"]
        assert messages[0] == {"role": "system", "content": "SYSTEM TEXT"}
        assert messages[1] == {"role": "user", "content": "USER TEXT"}


class TestTransportFailover:
    async def test_timeout_fails_over_immediately(self, tmp_path: Path) -> None:
        """A dead endpoint must not consume the validation retry budget."""
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(client, [a_timeout(), valid_result()])

        _, meta = await client.complete_structured("tier0", "s", "u", Tier0Result)

        assert len(stub.calls) == 2  # one attempt each, no retry on the dead one
        assert stub.calls[1]["model"] == "tier0__fb0"
        assert meta.fallback_used is True
        assert meta.provider == "openrouter"
        assert meta.validation_retries == 0

    async def test_instructor_wrapped_timeout_is_still_transport(self, tmp_path: Path) -> None:
        """The regression: wrapped timeouts were miscounted as validation failures."""
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(client, [wrapped(a_timeout()), valid_result()])

        _, meta = await client.complete_structured("tier0", "s", "u", Tier0Result)

        assert len(stub.calls) == 2  # failed over rather than retrying
        assert meta.fallback_used is True
        assert meta.validation_retries == 0

    async def test_connection_error_fails_over(self, tmp_path: Path) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        install_stub(
            client,
            [
                litellm.APIConnectionError(message="refused", model="m", llm_provider="ollama"),
                valid_result(),
            ],
        )
        _, meta = await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert meta.fallback_used is True

    async def test_unknown_error_also_fails_over(self, tmp_path: Path) -> None:
        """An unrecognised provider error must not strand the tier."""
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        install_stub(client, [RuntimeError("something odd"), valid_result()])
        _, meta = await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert meta.fallback_used is True

    async def test_exception_handling_never_raises_a_new_error(self, tmp_path: Path) -> None:
        """The AttributeError regression: the except clause itself blew up, so the
        real cause was replaced by a nonsense error at the call site."""
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        install_stub(client, [a_timeout(), a_timeout()])

        with pytest.raises(LLMUnavailable) as excinfo:
            await client.complete_structured("tier0", "s", "u", Tier0Result)
        # The surfaced error names the real failure, not an AttributeError.
        assert "AttributeError" not in str(excinfo.value)
        assert "timed out" in str(excinfo.value).lower()


class TestValidationRetries:
    async def test_retries_on_the_same_endpoint_then_fails_over(self, tmp_path: Path) -> None:
        """§7: 2x validation failure moves to the next endpoint."""
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(
            client,
            [
                wrapped(a_validation_error()),
                wrapped(a_validation_error()),
                wrapped(a_validation_error()),
                valid_result(),
            ],
        )
        _, meta = await client.complete_structured("tier0", "s", "u", Tier0Result)

        # 3 attempts on the primary (1 + 2 retries), then the fallback.
        assert [c["model"] for c in stub.calls] == [
            "tier0",
            "tier0",
            "tier0",
            "tier0__fb0",
        ]
        assert meta.fallback_used is True
        assert meta.validation_retries == 3

    async def test_recovers_without_failing_over(self, tmp_path: Path) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(client, [wrapped(a_validation_error()), valid_result()])
        _, meta = await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert [c["model"] for c in stub.calls] == ["tier0", "tier0"]
        assert meta.fallback_used is False
        assert meta.validation_retries == 1

    async def test_zero_retries_config_fails_over_at_once(self, tmp_path: Path) -> None:
        settings = two_endpoint_settings(tmp_path, validation_retries=0)
        client = LLMClient(settings, MemoryStore())
        stub = install_stub(client, [wrapped(a_validation_error()), valid_result()])
        await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert [c["model"] for c in stub.calls] == ["tier0", "tier0__fb0"]


class TestChainExhaustion:
    async def test_raises_llm_unavailable_listing_attempts(self, tmp_path: Path) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        install_stub(client, [a_timeout(), a_timeout()])
        with pytest.raises(LLMUnavailable) as excinfo:
            await client.complete_structured("tier0", "s", "u", Tier0Result)
        message = str(excinfo.value)
        assert "ollama_chat/local-small" in message
        assert "openrouter/vendor/remote" in message

    async def test_unknown_tier_is_rejected(self, tmp_path: Path) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        with pytest.raises(LLMUnavailable, match="no endpoints configured"):
            await client.complete_structured("tier99", "s", "u", Tier0Result)

    async def test_no_telemetry_written_when_every_endpoint_fails(self, tmp_path: Path) -> None:
        store = MemoryStore()
        client = LLMClient(two_endpoint_settings(tmp_path), store)
        install_stub(client, [a_timeout(), a_timeout()])
        with pytest.raises(LLMUnavailable):
            await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert store.docs_of_type("llm_call") == []


class TestCloudFallbackSwitch:
    async def test_local_only_tier_has_a_single_endpoint(self, tmp_path: Path) -> None:
        """§10.6: with the switch off, the cloud endpoint is not merely skipped —
        it is absent from the chain, so nothing can route to it."""
        settings = two_endpoint_settings(tmp_path, allow_cloud_fallback=False)
        client = LLMClient(settings, MemoryStore())
        assert [alias for alias, _ in client._chains["tier0"]] == ["tier0"]

        install_stub(client, [a_timeout()])
        with pytest.raises(LLMUnavailable) as excinfo:
            await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert "openrouter" not in str(excinfo.value)


class TestEndpointParams:
    def test_local_endpoint_gets_base_url_and_no_key(self, tmp_path: Path) -> None:
        settings = two_endpoint_settings(tmp_path)
        client = LLMClient(settings, MemoryStore())
        params = client._litellm_params(settings.llm.tiers["tier0"].primary, 60)
        assert params["model"] == "ollama_chat/local-small"
        assert params["api_base"] == "http://localhost:11434"
        assert "api_key" not in params
        assert params["timeout"] == 60

    def test_cloud_endpoint_gets_its_api_key(self, tmp_path: Path) -> None:
        settings = two_endpoint_settings(tmp_path)
        client = LLMClient(settings, MemoryStore())
        params = client._litellm_params(settings.llm.tiers["tier0"].fallbacks[0], 60)
        assert params["api_key"] == "sk-test"

    def test_extra_params_pass_through(self, tmp_path: Path) -> None:
        """Needed for provider-specific flags such as Ollama's `think: false`."""
        settings = make_settings(
            tmp_path,
            llm={
                "tiers": {
                    "tier0": {
                        "primary": {
                            "provider": "ollama",
                            "model": "m",
                            "extra_params": {"think": False, "num_ctx": 8192},
                        }
                    },
                    "tier1": {"primary": {"provider": "ollama", "model": "big"}},
                }
            },
        )
        client = LLMClient(settings, MemoryStore())
        params = client._litellm_params(settings.llm.tiers["tier0"].primary, 60)
        assert params["think"] is False
        assert params["num_ctx"] == 8192


class TestEndpointCooldown:
    """A down endpoint costs a full timeout on every call that walks past it.

    That is what makes a deferred stage drain so slowly once the backend is
    back: every episode pays the same wait before reaching the endpoint that
    works. Remembering the failure for a minute removes the repeat.
    """

    async def test_a_second_call_skips_the_endpoint_that_just_timed_out(
        self, tmp_path: Path
    ) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(client, [a_timeout(), valid_result(), valid_result()])

        await client.complete_structured("tier0", "s", "u", Tier0Result)
        await client.complete_structured("tier0", "s", "u", Tier0Result)

        # Three calls, not four: the second request went straight to the fallback.
        assert [c["model"] for c in stub.calls] == ["tier0", "tier0__fb0", "tier0__fb0"]

    async def test_the_skip_still_reports_a_fallback_was_used(self, tmp_path: Path) -> None:
        """Telemetry describes the chain, not the shortcut taken through it."""
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        install_stub(client, [a_timeout(), valid_result(), valid_result()])

        await client.complete_structured("tier0", "s", "u", Tier0Result)
        _, meta = await client.complete_structured("tier0", "s", "u", Tier0Result)

        assert meta.fallback_used is True
        assert meta.provider == "openrouter"

    async def test_a_validation_failure_does_not_cool_an_endpoint(self, tmp_path: Path) -> None:
        """A model emitting bad JSON is answering, and answering is the point."""
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(
            client,
            [
                a_validation_error(),
                a_validation_error(),
                a_validation_error(),
                valid_result(),
                valid_result(),
            ],
        )
        await client.complete_structured("tier0", "s", "u", Tier0Result)
        await client.complete_structured("tier0", "s", "u", Tier0Result)

        # The primary is tried again on the second call rather than stepped over.
        assert stub.calls[-1]["model"] == "tier0"

    async def test_success_clears_the_cooldown(self, tmp_path: Path) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        install_stub(client, [a_timeout(), valid_result()])
        await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert "tier0" in client._cooling

        # Time passes; the primary answers and is trusted again.
        client._cooling.clear()
        stub = install_stub(client, [valid_result(), valid_result()])
        await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert "tier0" not in client._cooling
        await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert [c["model"] for c in stub.calls] == ["tier0", "tier0"]

    async def test_the_cooldown_expires(self, tmp_path: Path, monkeypatch) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        stub = install_stub(client, [a_timeout(), valid_result(), valid_result()])
        await client.complete_structured("tier0", "s", "u", Tier0Result)

        # Wind the clock past the window; the primary is due another chance.
        client._cooling["tier0"] -= ENDPOINT_COOLDOWN_S + 1
        await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert stub.calls[-1]["model"] == "tier0"

    async def test_every_endpoint_cooling_still_walks_the_whole_chain(self, tmp_path: Path) -> None:
        """A cooldown avoids a wasted timeout. It must never be the reason a
        tier reports itself unavailable while an endpoint might answer."""
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        install_stub(client, [a_timeout(), a_timeout()])
        with pytest.raises(LLMUnavailable):
            await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert set(client._cooling) == {"tier0", "tier0__fb0"}

        stub = install_stub(client, [valid_result()])
        _, meta = await client.complete_structured("tier0", "s", "u", Tier0Result)
        assert stub.calls[0]["model"] == "tier0"
        assert meta.fallback_used is False

    async def test_one_tier_cooling_does_not_affect_another(self, tmp_path: Path) -> None:
        client = LLMClient(two_endpoint_settings(tmp_path), MemoryStore())
        install_stub(client, [a_timeout(), valid_result()])
        await client.complete_structured("tier0", "s", "u", Tier0Result)

        stub = install_stub(client, [valid_result()])
        await client.complete_structured("tier1", "s", "u", Tier0Result)
        assert stub.calls[0]["model"] == "tier1"


class TestReplyParsing:
    """The mode instructor reads replies in.

    A whole tier reported itself unavailable — every endpoint, every retry —
    while both providers were answering correctly, because the answers arrived
    wrapped in a markdown fence and the parser was reading them verbatim.
    """

    FENCED = (
        "```json\n"
        '{"relevance_guess": 6, "confidence": 7, '
        '"matched_interests": ["ai_agent_security"], '
        '"reasoning": "npm supply chain plus an AI CI/CD privilege flaw.", '
        '"route": "DIGEST_DIRECT"}\n'
        "```"
    )
    BARE = FENCED.removeprefix("```json\n").removesuffix("\n```")

    @staticmethod
    def _canned(content: str):
        async def _completion(*_args: Any, **_kwargs: Any) -> Any:
            return ModelResponse(
                id="x",
                created=0,
                model="m",
                object="chat.completion",
                choices=[
                    Choices(
                        finish_reason="stop",
                        index=0,
                        message=LiteLLMMessage(content=content, role="assistant"),
                    )
                ],
            )

        return _completion

    async def _parse(self, content: str) -> Tier0Result:
        client = instructor.from_litellm(
            self._canned(content), mode=INSTRUCTOR_MODE, async_client=True
        )
        result, _raw = await client.chat.completions.create_with_completion(
            model="m",
            response_model=Tier0Result,
            max_retries=1,
            messages=[{"role": "user", "content": "x"}],
        )
        return result  # type: ignore[no-any-return]

    async def test_parses_json_wrapped_in_a_markdown_fence(self) -> None:
        assert (await self._parse(self.FENCED)).route == "DIGEST_DIRECT"

    async def test_still_parses_bare_json(self) -> None:
        """The fix must be a superset — the providers that never fence still work."""
        assert (await self._parse(self.BARE)).route == "DIGEST_DIRECT"
