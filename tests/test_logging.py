"""Logging configuration tests (§10.1).

These exist because a real startup crash slipped through: `add_logger_name`
requires a stdlib logger, but the factory produced a PrintLogger. Nothing in the
suite had ever *called* configure_logging and emitted a line, so a broken
processor chain only failed when the app actually booted.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from podcast_agent.config import LoggingConfig
from podcast_agent.logging_setup import bind_run, clear_run_context, configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_logging():
    """Restore root handlers so configuring logging cannot leak between tests."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    clear_run_context()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _emit_json(capsys: pytest.CaptureFixture[str], **kwargs) -> dict:
    """Configure JSON logging, emit one line, and parse it back."""
    configure_logging(LoggingConfig(level="INFO", format="json"))
    get_logger("test.module").info("test.event", **kwargs)
    out = capsys.readouterr().out.strip().splitlines()
    assert out, "nothing was logged"
    return json.loads(out[-1])


class TestJsonOutput:
    def test_emits_parseable_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The documented `docker logs | jq` workflow needs every line to parse."""
        record = _emit_json(capsys, episode_id="episode:abc", stage="tier0")
        assert record["event"] == "test.event"
        assert record["episode_id"] == "episode:abc"
        assert record["stage"] == "tier0"

    def test_includes_level_logger_and_timestamp(self, capsys: pytest.CaptureFixture[str]) -> None:
        record = _emit_json(capsys)
        assert record["level"] == "info"
        # The regression: add_logger_name needs a stdlib logger to read .name from.
        assert record["logger"] == "test.module"
        assert record["timestamp"].endswith("Z") or "+00:00" in record["timestamp"]

    def test_run_context_is_bound_to_every_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        """§10.1: every log line carries run_id."""
        configure_logging(LoggingConfig(level="INFO", format="json"))
        bind_run("run123", job="pipeline")
        get_logger("test").info("first")
        get_logger("other").info("second")
        lines = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
        assert all(entry["run_id"] == "run123" for entry in lines)
        assert all(entry["job"] == "pipeline" for entry in lines)

    def test_context_clears(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(LoggingConfig(level="INFO", format="json"))
        bind_run("run123")
        clear_run_context()
        get_logger("test").info("after")
        record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert "run_id" not in record


class TestRedactionIsWiredIn:
    def test_secrets_are_redacted_in_real_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Redaction must be installed in the chain, not merely implemented."""
        record = _emit_json(capsys, api_key="sk-live-secret", password="hunter2")
        assert "sk-live-secret" not in json.dumps(record)
        assert "hunter2" not in json.dumps(record)
        assert record["api_key"] == "***redacted***"

    def test_bulk_content_is_truncated_in_real_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record = _emit_json(capsys, transcript="x" * 10_000)
        assert len(record["transcript"]) < 300


class TestThirdPartyLogging:
    def test_stdlib_records_render_in_the_same_format(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A plain-text uvicorn line interleaved with JSON breaks log parsing."""
        configure_logging(LoggingConfig(level="INFO", format="json"))
        logging.getLogger("uvicorn.error").warning("started on port %d", 8080)
        line = capsys.readouterr().out.strip().splitlines()[-1]
        record = json.loads(line)  # must not raise
        assert record["event"] == "started on port 8080"
        assert record["level"] == "warning"
        assert record["logger"] == "uvicorn.error"

    def test_noisy_libraries_are_clamped_to_warning(self) -> None:
        configure_logging(LoggingConfig(level="INFO", format="json"))
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("litellm").level >= logging.WARNING


class TestConfigurationBehaviour:
    def test_console_format_is_human_readable(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(LoggingConfig(level="INFO", format="console"))
        get_logger("test").info("readable.event", episode_id="episode:abc")
        out = capsys.readouterr().out
        assert "readable.event" in out
        assert not out.strip().startswith("{")

    def test_level_filtering_applies(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(LoggingConfig(level="WARNING", format="json"))
        log = get_logger("test")
        log.info("suppressed")
        log.warning("kept")
        out = capsys.readouterr().out
        assert "suppressed" not in out
        assert "kept" in out

    def test_repeated_calls_do_not_duplicate_lines(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """configure_logging is called from both create_app and cli."""
        configure_logging(LoggingConfig(level="INFO", format="json"))
        configure_logging(LoggingConfig(level="INFO", format="json"))
        get_logger("test").info("once")
        assert len(capsys.readouterr().out.strip().splitlines()) == 1

    def test_exception_info_is_rendered(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(LoggingConfig(level="INFO", format="json"))
        try:
            raise ValueError("boom")
        except ValueError:
            get_logger("test").warning("failed.thing", exc_info=True)
        record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert "ValueError: boom" in record.get("exception", "")


class TestRedactionPrecision:
    """Over-redaction is a real failure mode: it silently destroys telemetry.

    Regression: "token" matched inside "input_tokens"/"output_tokens", so every
    per-call token count was logged as ***redacted*** — exactly the data the
    local-vs-cloud cost comparison in §6 depends on.
    """

    def test_token_counts_survive(self, capsys: pytest.CaptureFixture[str]) -> None:
        record = _emit_json(capsys, input_tokens=1234, output_tokens=56, total_tokens=1290)
        assert record["input_tokens"] == 1234
        assert record["output_tokens"] == 56
        assert record["total_tokens"] == 1290

    def test_actual_credentials_are_still_redacted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record = _emit_json(
            capsys,
            access_token="at-secret",
            auth_token="auth-secret",
            api_key="k",
            password="p",
        )
        for key in ("access_token", "auth_token", "api_key", "password"):
            assert record[key] == "***redacted***", key

    def test_ordinary_pipeline_fields_survive(self, capsys: pytest.CaptureFixture[str]) -> None:
        record = _emit_json(
            capsys, episode_id="episode:x", latency_ms=42, cost_usd=0.01, relevance=9
        )
        assert record["latency_ms"] == 42
        assert record["cost_usd"] == 0.01
        assert record["relevance"] == 9


class TestThirdPartyNoise:
    """litellm's logging, made to behave like everything else's.

    Two separate problems. It attaches its own stderr handler at import — after
    the app has configured logging — so every line it emits appears twice, once
    as JSON through the shared chain and once as its own coloured text. And it
    warns on every Router construction that a model is missing from its built-in
    cost map, naming it by the opaque id the Router assigns each deployment.
    """

    def _configure(self) -> None:
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logging_setup import configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))

    def test_the_cost_map_warning_is_dropped(self) -> None:
        """True, harmless and repeated: local models cost nothing, and this
        deployment does not use prompt caching at all."""
        from podcast_agent.logbuffer import buffer

        self._configure()
        buffer.clear()
        logging.getLogger("LiteLLM").warning(
            "register_model: model=abc123 not in built-in cost map and no "
            "prefix/region variant matched; cache cost fields will default to 0."
        )
        assert buffer.tail(limit=5) == []

    def test_other_litellm_warnings_still_get_through(self) -> None:
        """A filter that swallows everything is worse than the noise."""
        from podcast_agent.logbuffer import buffer

        self._configure()
        buffer.clear()
        logging.getLogger("LiteLLM").warning("fallback fired: primary endpoint timed out")
        events = [str(e.get("event")) for e in buffer.tail(limit=5)]
        assert any("fallback fired" in e for e in events)

    def test_litellm_keeps_no_handler_of_its_own(self) -> None:
        """Its handler is what produced the second, differently formatted copy."""
        from podcast_agent.logging_setup import tame_litellm_logging

        self._configure()
        logging.getLogger("LiteLLM").addHandler(logging.StreamHandler())
        tame_litellm_logging()
        for name in ("LiteLLM", "LiteLLM Router", "litellm"):
            logger = logging.getLogger(name)
            assert logger.handlers == [], f"{name} still has its own handler"
            assert logger.propagate is True

    def test_taming_twice_does_not_stack_filters(self) -> None:
        """It is called from configure_logging and again after litellm imports."""
        from podcast_agent.logging_setup import tame_litellm_logging

        self._configure()
        tame_litellm_logging()
        tame_litellm_logging()
        assert len(logging.getLogger("LiteLLM").filters) == 1


class TestCancelledJobsAreNotFailures:
    """Restarting mid-job is not a fault, and must not be logged as one.

    CancelledError is a BaseException, so `except Exception` in the job wrapper
    never saw it: every restart during a transcription or an LLM call escaped to
    APScheduler and was logged as "Job ... raised an exception". Four such
    errors appeared in one evening, all of them caused by restarts — burying the
    real ones, and now doubly so since errors are kept in the database.
    """

    def _job(self, func):
        from podcast_agent.scheduler import _guarded

        return _guarded("backfill", func)

    async def test_cancellation_during_shutdown_is_reported_as_such(self) -> None:
        import podcast_agent.scheduler as scheduler_module
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))

        async def cancelled() -> None:
            raise asyncio.CancelledError

        scheduler_module._shutting_down = False
        scheduler_module.mark_shutting_down()
        buffer.clear()
        try:
            await self._job(cancelled)()  # must not raise
            events = [e.get("event") for e in buffer.tail(limit=5)]
            assert "scheduler.job_cancelled_at_shutdown" in events
            assert "scheduler.job_failed" not in events
        finally:
            scheduler_module._shutting_down = False

    async def test_cancellation_at_any_other_time_still_propagates(self) -> None:
        """Swallowing it outside shutdown would hide a real interruption."""
        import podcast_agent.scheduler as scheduler_module

        async def cancelled() -> None:
            raise asyncio.CancelledError

        scheduler_module._shutting_down = False
        with pytest.raises(asyncio.CancelledError):
            await self._job(cancelled)()

    async def test_a_genuine_failure_is_still_an_error(self) -> None:
        """The point is to stop hiding these behind restart noise."""
        import podcast_agent.scheduler as scheduler_module
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logbuffer import buffer
        from podcast_agent.logging_setup import configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))

        async def broken() -> None:
            raise RuntimeError("the feed exploded")

        scheduler_module._shutting_down = True
        buffer.clear()
        try:
            await self._job(broken)()
            assert "scheduler.job_failed" in [e.get("event") for e in buffer.tail(limit=5)]
        finally:
            scheduler_module._shutting_down = False


class TestSchedulerOverlapNoise:
    """APScheduler's overlap notice is routine, and at warning level it hid work.

    Backfill polls every 20 minutes but a run takes hours, so `max_instances=1`
    refuses nearly every fire. That is the design. As ~40 warnings a night it
    also buried a repeatedly truncated audio download and a truncated LLM reply
    in the same log, for a week.
    """

    def _record(self, message: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="apscheduler.scheduler",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )

    def test_the_overlap_notice_is_dropped_at_info(self) -> None:
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logging_setup import _demote_overlap_skips, configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))
        record = self._record(
            'Execution of job "backfill (trigger: cron[...])" skipped: '
            "maximum number of running instances reached (1)"
        )
        assert _demote_overlap_skips(record) is False
        assert record.levelno == logging.DEBUG

    def test_it_is_still_visible_at_debug(self) -> None:
        """Demoted, not deleted — it is the evidence a job is overrunning."""
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logging_setup import _demote_overlap_skips, configure_logging

        configure_logging(LoggingConfig(level="DEBUG", format="json"))
        try:
            record = self._record(
                'Execution of job "backfill" skipped: '
                "maximum number of running instances reached (1)"
            )
            assert _demote_overlap_skips(record) is True
        finally:
            configure_logging(LoggingConfig(level="INFO", format="json"))

    def test_every_other_scheduler_warning_is_untouched(self) -> None:
        from podcast_agent.logging_setup import _demote_overlap_skips

        record = self._record('Job "ingest" raised an exception')
        assert _demote_overlap_skips(record) is True
        assert record.levelno == logging.WARNING

    def test_the_filter_is_installed_on_the_right_logger(self) -> None:
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logging_setup import _demote_overlap_skips, configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))
        assert _demote_overlap_skips in logging.getLogger("apscheduler.scheduler").filters

    def test_configuring_twice_does_not_stack_the_filter(self) -> None:
        from podcast_agent.config import LoggingConfig
        from podcast_agent.logging_setup import _demote_overlap_skips, configure_logging

        configure_logging(LoggingConfig(level="INFO", format="json"))
        configure_logging(LoggingConfig(level="INFO", format="json"))
        installed = logging.getLogger("apscheduler.scheduler").filters
        assert installed.count(_demote_overlap_skips) == 1
