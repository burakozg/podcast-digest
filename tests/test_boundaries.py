"""Architectural boundary and prompt tests.

These guard invariants the design states explicitly but that nothing else would
catch: the LLM vendor stays behind one module (§7), prompts are versioned and
carry their injection defences (§7, §10.2), and secrets never reach the logs (§10.1).
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from podcast_agent.config import InterestItem
from podcast_agent.llm.prompts import (
    PROMPT_DIR,
    PromptError,
    format_interest_profile,
    load_prompt,
)

PACKAGE_ROOT = Path(__file__).parent.parent / "podcast_agent"
LLM_PACKAGE = PACKAGE_ROOT / "llm"

VENDOR_IMPORTS = re.compile(r"^\s*(?:import|from)\s+(litellm|instructor)\b", re.MULTILINE)


class TestLlmVendorIsolation:
    def test_nothing_outside_llm_imports_litellm_or_instructor(self) -> None:
        """§7: 'Nothing outside llm/ may import litellm/instructor directly.'

        The point is that swapping provider libraries touches one module, and that
        the pipeline stays testable without loading a heavy vendor SDK.
        """
        offenders: list[str] = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            if LLM_PACKAGE in path.parents:
                continue
            if VENDOR_IMPORTS.search(path.read_text()):
                offenders.append(str(path.relative_to(PACKAGE_ROOT)))
        assert offenders == [], (
            f"{offenders} import litellm/instructor directly; route the call through "
            "llm.base.StructuredLLM instead"
        )

    def test_llm_package_init_does_not_import_the_vendor_eagerly(self) -> None:
        """Importing podcast_agent.llm must not drag in litellm — tests rely on it."""
        assert not VENDOR_IMPORTS.search((LLM_PACKAGE / "__init__.py").read_text())

    def test_pipeline_stages_depend_only_on_the_protocol(self) -> None:
        for module in (
            "podcast_agent/triage/tier0.py",
            "podcast_agent/summarize/tier1.py",
            "podcast_agent/pipeline/runner.py",
        ):
            source = (PACKAGE_ROOT.parent / module).read_text()
            assert "llm.client" not in source, f"{module} must not reach for the concrete client"


class TestPrompts:
    @pytest.mark.parametrize("name", ["tier0", "tier1", "tier1_map", "tier1_reduce"])
    def test_every_prompt_loads_with_both_sections(self, name: str) -> None:
        prompt = load_prompt(name)
        assert prompt.system
        assert prompt.user
        assert prompt.versioned_name == f"{name}_v1"

    @pytest.mark.parametrize("name", ["tier0", "tier1", "tier1_map", "tier1_reduce"])
    def test_every_prompt_declares_content_untrusted(self, name: str) -> None:
        """§10.2 mitigation 1: the system prompt must state that quoted content is
        data, not instructions."""
        system = load_prompt(name).system
        assert "UNTRUSTED DATA" in system
        assert "instruction" in system.lower()

    @pytest.mark.parametrize("name", ["tier0", "tier1", "tier1_map", "tier1_reduce"])
    def test_every_prompt_injects_the_interest_profile(self, name: str) -> None:
        """§7: the profile comes from config and is never hardcoded."""
        assert "{{ interest_profile }}" in load_prompt(name).system

    def test_missing_prompt_is_a_clear_error(self) -> None:
        with pytest.raises(PromptError, match="no prompt file"):
            load_prompt("does_not_exist")

    def test_missing_variable_raises_rather_than_rendering_blank(self) -> None:
        """A silently-empty prompt variable would degrade output invisibly."""
        with pytest.raises(PromptError):
            load_prompt("tier0").render(interest_profile="x")  # other vars missing

    def test_prompt_files_are_version_suffixed(self) -> None:
        for path in PROMPT_DIR.glob("*.md"):
            assert re.search(r"_v\d+\.md$", path.name), (
                f"{path.name} must carry a version suffix so calls are attributable"
            )

    def test_render_produces_both_parts(self) -> None:
        system, user = load_prompt("tier0").render(
            interest_profile="- `k` — **L** (weight 9/10): d",
            podcast_name="Show",
            priority="high",
            title="Title",
            published_at="2026-07-28",
            duration="30 min",
            description="Some description",
        )
        assert "weight 9/10" in system
        assert "Some description" in user
        assert "<episode_data>" in user


class TestInterestProfileFormatting:
    def test_orders_by_descending_weight(self) -> None:
        profile = [
            InterestItem(key="low", label="Low", description="d", weight=3),
            InterestItem(key="high", label="High", description="d", weight=10),
        ]
        rendered = format_interest_profile(profile)
        assert rendered.index("`high`") < rendered.index("`low`")

    def test_includes_keys_labels_and_weights(self) -> None:
        rendered = format_interest_profile(
            [InterestItem(key="ot_ics", label="OT/ICS", description="SCADA stuff", weight=10)]
        )
        assert "`ot_ics`" in rendered
        assert "**OT/ICS**" in rendered
        assert "weight 10/10" in rendered
        assert "SCADA stuff" in rendered


class TestLogRedaction:
    def test_secret_shaped_keys_are_redacted(self) -> None:
        from podcast_agent.logging_setup import _redact_secrets

        event = {
            "event": "test",
            "api_key": "sk-secret",
            "couchdb_password": "hunter2",
            "Authorization": "Bearer abc",
            "admin_token": "t",
            "episode_id": "episode:abc",
        }
        cleaned = _redact_secrets(None, "info", dict(event))
        assert cleaned["api_key"] == "***redacted***"
        assert cleaned["couchdb_password"] == "***redacted***"
        assert cleaned["Authorization"] == "***redacted***"
        assert cleaned["admin_token"] == "***redacted***"
        # Non-secret context is preserved.
        assert cleaned["episode_id"] == "episode:abc"

    def test_bulk_content_is_truncated(self) -> None:
        """§10.1: never log full transcripts or prompts."""
        from podcast_agent.logging_setup import _truncate_bulk

        event = {"transcript": "x" * 5000, "prompt": "y" * 5000, "title": "short"}
        cleaned = _truncate_bulk(None, "debug", dict(event))
        assert len(cleaned["transcript"]) < 300
        assert "5000 chars total" in cleaned["transcript"]
        assert len(cleaned["prompt"]) < 300
        assert cleaned["title"] == "short"


class TestSecondInstanceRefused:
    """Starting a second agent must fail before it can touch shared state.

    uvicorn runs the ASGI lifespan *before* binding the socket, so a second
    process started by mistake ran the whole startup — migrations, scheduler,
    and `mark_applied`, which clears the console's "waiting for a restart"
    banner — and only then died on the port. The first process kept serving the
    old configuration, so a saved setting appeared to revert on every restart.
    """

    def _busy_port(self) -> tuple[str, int, socket.socket]:
        held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        return "127.0.0.1", held.getsockname()[1], held

    def test_refuses_a_port_already_listening(self, capsys) -> None:
        from podcast_agent.main import require_free_port

        host, port, held = self._busy_port()
        try:
            with pytest.raises(SystemExit) as exit_info:
                require_free_port(host, port)
        finally:
            held.close()

        assert exit_info.value.code == 1
        # The message has to name the fix; the whole failure mode is that the
        # operator believes the restart worked.
        err = capsys.readouterr().err
        assert "already in use" in err
        assert "podcast-agent" in err

    def test_allows_a_free_port(self) -> None:
        from podcast_agent.main import require_free_port

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        require_free_port("127.0.0.1", port)  # must not raise

    def test_the_probe_does_not_keep_the_port(self) -> None:
        """Otherwise the check itself would block the server it guards."""
        from podcast_agent.main import require_free_port

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        require_free_port("127.0.0.1", port)
        require_free_port("127.0.0.1", port)


class TestShutdownLetsJobsCleanUp:
    """Cancelling a background job is not the same as waiting for it.

    The lifespan cancelled its tasks and moved straight on to closing the HTTP
    clients and the store. A cancelled job's `finally` — which releases its
    database lease, a write — therefore ran against a closed client:

        joblock.release_failed  Cannot send a request, as the client has been closed
        joblock.busy            backfill

    Every restart during a backfill orphaned `control:lock:backfill` and locked
    the job out until the lease expired on its own.
    """

    def test_the_lifespan_awaits_what_it_cancels(self) -> None:
        source = (PACKAGE_ROOT / "main.py").read_text()
        cancel = source.index("task.cancel()")
        close = source.index("await active_store.close()")
        gather = source.index("asyncio.gather(*running", cancel)
        assert cancel < gather < close, (
            "cancelled tasks must be awaited between cancellation and teardown, "
            "or their cleanup runs against closed clients"
        )

    def test_the_wait_is_bounded(self) -> None:
        """A job that declines to stop must not hold shutdown open forever."""
        source = (PACKAGE_ROOT / "main.py").read_text()
        assert "asyncio.timeout(SHUTDOWN_GRACE_S)" in source

    def test_startup_reclaims_before_the_scheduler_starts(self) -> None:
        """Otherwise a job can be refused by a lock its own predecessor left."""
        source = (PACKAGE_ROOT / "main.py").read_text()
        assert source.index("reclaim_local_leases(active_store)") < source.index("build_scheduler(")
