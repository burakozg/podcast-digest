"""The in-memory log tail behind /admin/logs.

Two things must hold, and neither is obvious from the happy path: the buffer is
bounded (an agent left running for a week must not accumulate a log in RAM), and
the tap can never break logging, because a logging pipeline that raises takes the
process's only diagnostic channel with it.
"""

from __future__ import annotations

import logging

import pytest

from podcast_agent.config import LoggingConfig
from podcast_agent.logbuffer import LogBuffer, sink
from podcast_agent.logging_setup import configure_logging, get_logger


@pytest.fixture
def buf() -> LogBuffer:
    return LogBuffer(capacity=5)


class TestBounded:
    def test_it_never_grows_past_capacity(self, buf: LogBuffer) -> None:
        for i in range(500):
            buf.add({"event": f"e{i}", "level": "info"})
        assert len(buf) == 5

    def test_the_oldest_events_are_the_ones_dropped(self, buf: LogBuffer) -> None:
        for i in range(8):
            buf.add({"event": f"e{i}", "level": "info"})
        assert [e["event"] for e in buf.tail(limit=10)] == ["e7", "e6", "e5", "e4", "e3"]

    def test_a_huge_field_is_trimmed_before_it_is_kept(self) -> None:
        """One traceback must not be allowed to consume the whole budget."""
        from podcast_agent.logbuffer import MAX_VALUE_CHARS, buffer

        buffer.clear()
        sink(None, "", {"event": "big", "level": "error", "blob": "x" * 100_000})
        held = buffer.tail(limit=1)[0]
        assert len(held["blob"]) <= MAX_VALUE_CHARS + 1
        buffer.clear()


class TestFiltering:
    @pytest.fixture
    def filled(self, buf: LogBuffer) -> LogBuffer:
        buf.add({"event": "quiet", "level": "debug", "logger": "a.b"})
        buf.add({"event": "normal", "level": "info", "logger": "a.b"})
        buf.add({"event": "odd", "level": "warning", "logger": "c.d"})
        buf.add({"event": "broken", "level": "error", "logger": "c.d", "podcast": "risky"})
        return buf

    def test_newest_first(self, filled: LogBuffer) -> None:
        assert next(e["event"] for e in filled.tail()) == "broken"

    def test_level_filter_means_at_or_above(self, filled: LogBuffer) -> None:
        """ "Show me warnings" must not hide the errors."""
        events = {e["event"] for e in filled.tail(level="warning")}
        assert events == {"odd", "broken"}

    def test_errors_only(self, filled: LogBuffer) -> None:
        assert {e["event"] for e in filled.tail(level="error")} == {"broken"}

    def test_text_filter_searches_every_field_not_just_the_message(self, filled: LogBuffer) -> None:
        """The useful part of a structured log is in the key/values."""
        assert {e["event"] for e in filled.tail(contains="risky")} == {"broken"}

    def test_logger_filter(self, filled: LogBuffer) -> None:
        assert {e["event"] for e in filled.tail(logger="c.d")} == {"odd", "broken"}

    def test_limit_is_applied_after_filtering(self, filled: LogBuffer) -> None:
        assert len(filled.tail(level="debug", limit=2)) == 2

    def test_level_counts_are_reported(self, filled: LogBuffer) -> None:
        assert filled.levels() == {"debug": 1, "info": 1, "warning": 1, "error": 1}

    def test_an_unknown_level_is_never_silently_swallowed(self, buf: LogBuffer) -> None:
        buf.add({"event": "strange", "level": "notice"})
        assert {e["event"] for e in buf.tail()} == {"strange"}


class TestTheTapIsSafe:
    def test_it_returns_its_input_unchanged(self) -> None:
        """It is a tap, not a transform: stdout must be unaffected."""
        event = {"event": "hello", "level": "info"}
        assert sink(None, "", event) is event

    def test_an_unserialisable_value_does_not_raise(self) -> None:
        class Awkward:
            def __str__(self) -> str:
                raise RuntimeError("nope")

        # Must not propagate — a failing tap would take down logging itself.
        assert sink(None, "", {"event": "x", "level": "info", "bad": Awkward()})


class TestWiredIntoLogging:
    def test_a_real_log_call_lands_in_the_buffer(self) -> None:
        from podcast_agent.logbuffer import buffer

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        get_logger("test.buffer").info("wired_up", podcast="risky-business", count=3)

        held = buffer.tail(limit=5)
        assert any(e.get("event") == "wired_up" for e in held)
        entry = next(e for e in held if e.get("event") == "wired_up")
        assert entry["podcast"] == "risky-business"
        assert entry["count"] == 3
        assert entry["level"] == "info"

    def test_foreign_stdlib_logging_is_captured_too(self) -> None:
        """uvicorn and httpx log through stdlib; the console should show them."""
        from podcast_agent.logbuffer import buffer

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        logging.getLogger("uvicorn.error").warning("a foreign warning")
        assert any("foreign warning" in str(e.get("event")) for e in buffer.tail(limit=5))

    def test_secrets_are_redacted_before_they_reach_the_buffer(self) -> None:
        """The buffer is served over the API, so redaction must precede it."""
        from podcast_agent.logbuffer import buffer

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        get_logger("test.buffer").info("auth", api_key="super-secret-value")

        entry = next(e for e in buffer.tail(limit=5) if e.get("event") == "auth")
        assert "super-secret-value" not in str(entry), entry

    def test_token_counts_survive_redaction(self) -> None:
        """Regression: a broad "token" match once destroyed usage numbers."""
        from podcast_agent.logbuffer import buffer

        configure_logging(LoggingConfig(level="INFO", format="json"))
        buffer.clear()
        get_logger("test.buffer").info("llm_call", input_tokens=1234, output_tokens=56)

        entry = next(e for e in buffer.tail(limit=5) if e.get("event") == "llm_call")
        assert entry["input_tokens"] == 1234
        assert entry["output_tokens"] == 56


def test_capacity_is_reported_for_the_console(buf: LogBuffer) -> None:
    assert buf.capacity == 5


def test_clear_empties_it(buf: LogBuffer) -> None:
    buf.add({"event": "x", "level": "info"})
    buf.clear()
    assert len(buf) == 0 and buf.tail() == []


def test_events_carry_a_monotonic_sequence(buf: LogBuffer) -> None:
    """Lets the console poll for "anything newer than this" without duplicates."""
    for i in range(3):
        buf.add({"event": f"e{i}", "level": "info"})
    seqs = [e["seq"] for e in buf.tail()]
    assert seqs == sorted(seqs, reverse=True)
    assert len(set(seqs)) == 3


def test_since_seq_returns_only_newer_events(buf: LogBuffer) -> None:
    for i in range(4):
        buf.add({"event": f"e{i}", "level": "info"})
    mark = buf.tail(limit=10)[1]["seq"]
    assert [e["event"] for e in buf.tail(since_seq=mark)] == ["e3"]


def test_add_is_safe_from_many_threads(buf: LogBuffer) -> None:
    """Logging happens on the loop, on APScheduler threads and on uvicorn's."""
    import threading

    big = LogBuffer(capacity=5000)

    def hammer() -> None:
        for i in range(200):
            big.add({"event": f"e{i}", "level": "info"})

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(big) == 1600
    seqs = [e["seq"] for e in big.snapshot()]
    assert len(set(seqs)) == 1600, "sequence numbers collided under concurrency"
