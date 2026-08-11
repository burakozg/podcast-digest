"""The durable half of the log.

The in-memory tail answers "what is happening now" and a restart empties it —
which is exactly when someone asks what went wrong. This keeps warning and above
in the database.

Everything here is about not making things worse: a logging sink must not block
the pipeline, must not recurse when its own write fails, and must not grow
without bound when the database it writes to is the thing that is broken.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from podcast_agent.db import MemoryStore
from podcast_agent.logstore import KEEP_LEVELS, LogStore, sink


def event(level: str = "warning", name: str = "something_odd", **extra: Any) -> dict[str, Any]:
    return {"level": level, "event": name, "logger": "podcast_agent.x", **extra}


class TestOnlyTheExceptions:
    @pytest.mark.parametrize("level", sorted(KEEP_LEVELS))
    def test_warning_and_above_is_kept(self, level: str) -> None:
        store = LogStore()
        store.offer(event(level=level))
        assert len(store) == 1

    @pytest.mark.parametrize("level", ["debug", "info"])
    def test_the_running_commentary_is_not(self, level: str) -> None:
        """Info is what the in-memory tail is for; storing it is a table nobody reads."""
        store = LogStore()
        store.offer(event(level=level))
        assert len(store) == 0


class TestItCannotMakeThingsWorse:
    def test_the_queue_is_bounded(self) -> None:
        """An unreachable database must not be able to exhaust memory."""
        store = LogStore(capacity=10)
        for i in range(500):
            store.offer(event(name=f"e{i}"))
        assert len(store) == 10
        assert store.dropped > 0, "dropping should be counted, not silent"

    def test_offering_never_raises(self) -> None:
        """It runs inside a log call, inside the pipeline."""

        class Awkward:
            def __len__(self) -> int:
                raise RuntimeError("nope")

        store = LogStore()
        store.offer(event(bad=Awkward()))  # must not propagate

    async def test_a_failing_write_is_swallowed(self) -> None:
        """Reporting it would log, which queues, which fails the same way."""

        class Broken(MemoryStore):
            async def create(self, doc: Any) -> bool:
                raise RuntimeError("couch is down")

        store = LogStore()
        store.start(Broken())
        store.offer(event())
        await store.flush()  # must not raise
        await store.stop()

    def test_a_huge_value_is_trimmed(self) -> None:
        from podcast_agent.logstore import MAX_VALUE_CHARS

        store = LogStore()
        store.offer(event(traceback="x" * 100_000))
        # Plus the elision marker, which says how much went missing.
        assert len(store._queue[0]["traceback"]) <= MAX_VALUE_CHARS + 60

    def test_the_end_of_a_traceback_survives(self) -> None:
        """Python puts the exception type and message last.

        Clipping the tail kept a stack of framework frames and threw away the
        line saying what went wrong — on exactly the events that exist so a
        restart does not take the answer with it. Two real 500s were
        unexplainable afterwards for this reason.
        """
        store = LogStore()
        frames = "\n".join(f'  File "x.py", line {i}, in handler' for i in range(2000))
        store.offer(event(exception=f"Traceback:\n{frames}\nValueError: the actual cause"))

        kept = store._queue[0]["exception"]
        assert "ValueError: the actual cause" in kept, "the cause must survive"
        assert kept.startswith("Traceback:"), "and enough of the head to place it"
        assert "omitted" in kept, "and it should say what was dropped"


class TestWriting:
    async def test_an_event_reaches_the_database(self) -> None:
        db = MemoryStore()
        store = LogStore()
        store.start(db)
        store.offer(event(podcast="risky-business"))
        await store.flush()
        await store.stop()

        docs = db.docs_of_type("log")
        assert len(docs) == 1
        assert docs[0]["event"] == "something_odd"
        assert docs[0]["podcast"] == "risky-business"
        assert docs[0]["occurrences"] == 1
        assert docs[0]["at"]

    async def test_repeats_collapse_into_one_row_with_a_count(self) -> None:
        """One missing index once produced the same warning on every query."""
        db = MemoryStore()
        store = LogStore()
        store.start(db)
        for _ in range(50):
            store.offer(event())
        await store.flush()
        await store.stop()

        docs = db.docs_of_type("log")
        assert len(docs) == 1
        assert docs[0]["occurrences"] == 50

    async def test_different_events_stay_separate(self) -> None:
        db = MemoryStore()
        store = LogStore()
        store.start(db)
        store.offer(event(name="one"))
        store.offer(event(name="two"))
        store.offer(event(name="one", level="error"))
        await store.flush()
        await store.stop()
        assert len(db.docs_of_type("log")) == 3

    async def test_nothing_is_written_before_a_store_exists(self) -> None:
        """The sink queues from configure_logging; the database arrives later."""
        store = LogStore()
        store.offer(event())
        await store.flush()
        assert len(store) == 1, "kept, not discarded"

    async def test_stopping_flushes_what_is_left(self) -> None:
        """Shutdown is when the interesting failures happen."""
        db = MemoryStore()
        store = LogStore()
        store.start(db)
        store.offer(event(name="dying_breath"))
        await store.stop()
        assert [d["event"] for d in db.docs_of_type("log")] == ["dying_breath"]

    async def test_stop_returns_rather_than_hanging(self) -> None:
        """The drain loop must let cancellation out, not swallow and loop."""
        db = MemoryStore()
        store = LogStore()
        store.start(db)
        await asyncio.wait_for(store.stop(), timeout=2.0)


class TestTheTap:
    def test_it_returns_its_input_untouched(self) -> None:
        payload = event()
        assert sink(None, "", payload) is payload


def test_importing_logging_setup_first_does_not_deadlock_imports() -> None:
    """A real cycle: logging_setup -> logstore -> db -> logging_setup.

    The application happened to work because `main` imports `db` before
    `logging_setup`, so the loop never closed. Any entry point reaching
    logging_setup first — a script, a test, a console command — hit
    ImportError instead. The Store import is TYPE_CHECKING-only for this reason.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import podcast_agent.logging_setup; print('ok')"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "ok" in result.stdout
