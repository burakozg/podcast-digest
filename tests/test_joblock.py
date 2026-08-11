"""Cross-process job leases.

The in-process `asyncio.Lock` was the whole guarantee, and it only ever covered
one event loop in one process. These tests cover what it could not see: a second
instance, a CLI run, a replica — anything else holding the same database.
"""

from __future__ import annotations

import asyncio
import os
import socket
from datetime import timedelta

import pytest

from podcast_agent.db import MemoryStore
from podcast_agent.joblock import (
    JobLease,
    LeaseLost,
    current_holders,
    held,
    lock_id,
    reclaim_local_leases,
)
from podcast_agent.utils import iso, utcnow


class TestAcquisition:
    async def test_the_first_taker_wins(self, store: MemoryStore) -> None:
        assert await JobLease(store, "pipeline").acquire() is True

    async def test_the_second_is_refused(self, store: MemoryStore) -> None:
        assert await JobLease(store, "pipeline").acquire() is True
        assert await JobLease(store, "pipeline").acquire() is False

    async def test_different_jobs_do_not_block_each_other(self, store: MemoryStore) -> None:
        assert await JobLease(store, "ingest").acquire() is True
        assert await JobLease(store, "digest").acquire() is True

    async def test_exactly_one_of_a_simultaneous_pair_wins(self, store: MemoryStore) -> None:
        """The race the MVCC insert exists to settle."""
        leases = [JobLease(store, "backfill") for _ in range(8)]
        results = await asyncio.gather(*(lease.acquire() for lease in leases))
        assert sum(results) == 1

    async def test_the_document_says_who_holds_it(self, store: MemoryStore) -> None:
        """An operator waiting on a job needs to know which machine to look at."""
        lease = JobLease(store, "pipeline")
        await lease.acquire()
        doc = await store.get(lock_id("pipeline"))
        assert doc is not None
        assert doc["owner"] == lease.owner
        assert doc["host"] and doc["pid"]
        assert doc["acquired_at"] and doc["expires_at"]


class TestExpiry:
    """A process killed mid-run must not wedge its job forever."""

    async def _stale(self, store: MemoryStore, job: str, *, age_s: int = 60) -> None:
        expired = iso(utcnow() - timedelta(seconds=age_s))
        await store.create(
            {
                "_id": lock_id(job),
                "type": "job_lock",
                "job": job,
                "owner": "a-process-that-died",
                "acquired_at": expired,
                "expires_at": expired,
                "host": "gone",
                "pid": 1,
            }
        )

    async def test_an_expired_lease_can_be_taken_over(self, store: MemoryStore) -> None:
        await self._stale(store, "backfill")
        assert await JobLease(store, "backfill").acquire() is True

    async def test_the_takeover_records_who_it_took_it_from(self, store: MemoryStore) -> None:
        await self._stale(store, "backfill")
        await JobLease(store, "backfill").acquire()
        doc = await store.get(lock_id("backfill"))
        assert doc is not None
        assert doc["taken_over_from"] == "a-process-that-died"

    async def test_a_live_lease_is_not_taken_over(self, store: MemoryStore) -> None:
        await JobLease(store, "backfill", ttl_s=3600).acquire()
        assert await JobLease(store, "backfill").acquire() is False

    async def test_only_one_taker_wins_the_takeover(self, store: MemoryStore) -> None:
        """Two processes noticing the same corpse must not both revive it."""
        await self._stale(store, "backfill")
        leases = [JobLease(store, "backfill") for _ in range(6)]
        results = await asyncio.gather(*(lease.acquire() for lease in leases))
        assert sum(results) == 1


class TestRenewal:
    async def test_renewing_extends_the_expiry(self, store: MemoryStore) -> None:
        lease = JobLease(store, "backfill", ttl_s=60)
        await lease.acquire()
        before = (await store.get(lock_id("backfill")) or {})["expires_at"]
        await asyncio.sleep(0.01)
        assert await lease.renew() is True
        after = (await store.get(lock_id("backfill")) or {})["expires_at"]
        assert after > before

    async def test_renewing_keeps_the_original_acquisition_time(self, store: MemoryStore) -> None:
        """Otherwise `acquired_at` reports the last heartbeat, not the start."""
        lease = JobLease(store, "backfill", ttl_s=60)
        await lease.acquire()
        started = (await store.get(lock_id("backfill")) or {})["acquired_at"]
        await lease.renew()
        assert (await store.get(lock_id("backfill")) or {})["acquired_at"] == started

    async def test_a_lost_lease_cannot_be_renewed(self, store: MemoryStore) -> None:
        lease = JobLease(store, "backfill", ttl_s=60)
        await lease.acquire()
        doc = await store.get(lock_id("backfill"))
        assert doc is not None
        doc["owner"] = "someone-else"
        await store.put(doc)
        assert await lease.renew() is False


class TestRelease:
    async def test_releasing_frees_the_job(self, store: MemoryStore) -> None:
        lease = JobLease(store, "digest")
        await lease.acquire()
        await lease.release()
        assert await store.get(lock_id("digest")) is None
        assert await JobLease(store, "digest").acquire() is True

    async def test_it_never_deletes_someone_elses_lease(self, store: MemoryStore) -> None:
        """The failure this guards is subtle and bad.

        A process stalled past its TTL is replaced, finishes, and releases on the
        way out. Without the ownership check it deletes the *successor's* lock,
        and the next caller is told the job is free while a run is in progress.
        """
        stalled = JobLease(store, "backfill", ttl_s=60)
        await stalled.acquire()

        successor = JobLease(store, "backfill")
        doc = await store.get(lock_id("backfill"))
        assert doc is not None
        doc["owner"] = successor.owner
        await store.put(doc)

        await stalled.release()
        assert await store.get(lock_id("backfill")) is not None

    async def test_releasing_something_never_held_is_a_no_op(self, store: MemoryStore) -> None:
        await JobLease(store, "ingest", ttl_s=3600).acquire()
        loser = JobLease(store, "ingest")
        assert await loser.acquire() is False
        await loser.release()
        assert await store.get(lock_id("ingest")) is not None


class TestHeldContextManager:
    async def test_it_holds_then_frees(self, store: MemoryStore) -> None:
        async with held(store, "pipeline"):
            assert await store.get(lock_id("pipeline")) is not None
        assert await store.get(lock_id("pipeline")) is None

    async def test_it_frees_when_the_body_raises(self, store: MemoryStore) -> None:
        with pytest.raises(RuntimeError):
            async with held(store, "pipeline"):
                raise RuntimeError("boom")
        assert await store.get(lock_id("pipeline")) is None

    async def test_a_busy_job_raises_before_the_body_runs(self, store: MemoryStore) -> None:
        ran = False
        async with held(store, "pipeline"):
            with pytest.raises(LeaseLost):
                async with held(store, "pipeline"):
                    ran = True
        assert ran is False

    async def test_the_heartbeat_keeps_a_long_run_alive(self, store: MemoryStore) -> None:
        """A six-hour backfill must not lose the lease it is still using."""
        async with held(store, "backfill", ttl_s=1):
            first = (await store.get(lock_id("backfill")) or {})["expires_at"]
            # The renewal interval has a one-second floor, so this is the
            # shortest wait that can observe a beat.
            await asyncio.sleep(1.2)
            later = (await store.get(lock_id("backfill")) or {})["expires_at"]
            assert later > first

    async def test_the_heartbeat_stops_with_the_block(self, store: MemoryStore) -> None:
        """A leaked task would renew a lease nobody holds, forever."""
        before = len(asyncio.all_tasks())
        async with held(store, "digest", ttl_s=1):
            pass
        await asyncio.sleep(0)
        assert len(asyncio.all_tasks()) <= before


class TestCurrentHolders:
    async def test_it_lists_live_leases(self, store: MemoryStore) -> None:
        async with held(store, "pipeline"):
            holders = await current_holders(store)
        assert [h["job"] for h in holders] == ["pipeline"]

    async def test_expired_leases_are_not_reported_as_held(self, store: MemoryStore) -> None:
        expired = iso(utcnow() - timedelta(hours=1))
        await store.create(
            {
                "_id": lock_id("backfill"),
                "type": "job_lock",
                "job": "backfill",
                "owner": "dead",
                "acquired_at": expired,
                "expires_at": expired,
            }
        )
        assert await current_holders(store) == []

    async def test_a_storage_failure_is_not_fatal(self, store: MemoryStore) -> None:
        """It decorates a status page; it must never be the reason one 500s."""

        async def _boom(*_a: object, **_k: object) -> list[dict[str, object]]:
            raise RuntimeError("couch is down")

        store.find = _boom  # type: ignore[method-assign]
        assert await current_holders(store) == []


class TestReclaimingAfterACrash:
    """A lease outliving its holder locks the job out for a whole TTL.

    A graceful shutdown releases its own. A `kill -9`, an OOM, or — as happened
    here — a cleanup that ran after the store had already closed, does not.
    """

    async def _lease_owned_by(self, store: MemoryStore, job: str, *, pid: int, host: str) -> None:
        future = iso(utcnow() + timedelta(hours=1))
        await store.create(
            {
                "_id": lock_id(job),
                "type": "job_lock",
                "job": job,
                "owner": "gone",
                "acquired_at": future,
                "expires_at": future,
                "host": host,
                "pid": pid,
            }
        )

    async def test_a_dead_local_pid_is_reclaimed(self, store: MemoryStore) -> None:
        # 2**22 is above the default pid_max everywhere we run, so it is free.
        await self._lease_owned_by(store, "backfill", pid=2**22, host=socket.gethostname())
        assert await reclaim_local_leases(store) == 1
        assert await store.get(lock_id("backfill")) is None

    async def test_a_live_local_pid_is_left_alone(self, store: MemoryStore) -> None:
        """Two agents on one host is unusual, not impossible."""
        await self._lease_owned_by(store, "backfill", pid=os.getpid(), host=socket.gethostname())
        assert await reclaim_local_leases(store) == 0
        assert await store.get(lock_id("backfill")) is not None

    async def test_another_host_is_never_touched(self, store: MemoryStore) -> None:
        """We cannot see whether its holder is alive, and guessing wrong means
        two processes running the same job."""
        await self._lease_owned_by(store, "backfill", pid=2**22, host="some-other-machine")
        assert await reclaim_local_leases(store) == 0
        assert await store.get(lock_id("backfill")) is not None

    async def test_the_job_is_runnable_again_afterwards(self, store: MemoryStore) -> None:
        await self._lease_owned_by(store, "backfill", pid=2**22, host=socket.gethostname())
        assert await JobLease(store, "backfill").acquire() is False
        await reclaim_local_leases(store)
        assert await JobLease(store, "backfill").acquire() is True

    async def test_a_storage_failure_is_not_fatal(self, store: MemoryStore) -> None:
        """It runs during startup; it must never be the reason boot fails."""

        async def _boom(*_a: object, **_k: object) -> list[dict[str, object]]:
            raise RuntimeError("couch is down")

        store.find = _boom  # type: ignore[method-assign]
        assert await reclaim_local_leases(store) == 0


class TestShutdownOrdering:
    """The bug the first fix missed.

    `app.state.background_tasks` holds only API-triggered runs. Scheduled jobs
    are APScheduler's, and `shutdown(wait=False)` cancels one without waiting —
    so a backfill's lease release still landed after the store had closed:

        joblock.release_failed  Cannot send a request, as the client has been closed

    The earlier assertion checked that *some* tasks were awaited and passed
    while the real ones were not. This drives a job through the wrapper.
    """

    async def test_a_running_job_is_awaited_before_the_store_would_close(
        self, store: MemoryStore
    ) -> None:
        from podcast_agent import scheduler as scheduler_module

        released = asyncio.Event()
        started = asyncio.Event()

        async def long_job() -> None:
            async with held(store, "backfill"):
                started.set()
                try:
                    await asyncio.sleep(3600)
                finally:
                    released.set()

        job = scheduler_module._guarded("backfill", long_job)
        task = asyncio.create_task(job())
        await started.wait()
        assert await store.get(lock_id("backfill")) is not None

        # What the lifespan does: mark, cancel, then drain.
        scheduler_module.mark_shutting_down()
        try:
            task.cancel()
            drained = await scheduler_module.drain_jobs(5)
        finally:
            scheduler_module._shutting_down = False

        assert drained == 1
        assert released.is_set(), "cleanup had not run by the time drain returned"
        # And the lease is gone, which is the thing that was being lost.
        assert await store.get(lock_id("backfill")) is None

    async def test_draining_with_nothing_running_is_free(self) -> None:
        from podcast_agent import scheduler as scheduler_module

        assert await scheduler_module.drain_jobs(5) == 0

    def test_the_wait_is_bounded(self) -> None:
        """Or a job that declines to stop deadlocks every restart.

        Asserted on the source rather than by driving an uncancellable task:
        a test that has to out-stubborn the event loop is a test that hangs CI
        the first time it loses.
        """
        from pathlib import Path

        source = (Path(__file__).parent.parent / "podcast_agent/scheduler.py").read_text()
        drain = source[source.index("async def drain_jobs") :]
        assert "asyncio.timeout(grace_s)" in drain
        assert "suppress(TimeoutError)" in drain

    def test_shutdown_drains_before_closing_the_store(self) -> None:
        """The ordering the whole fix consists of."""
        from pathlib import Path

        source = (Path(__file__).parent.parent / "podcast_agent/main.py").read_text()
        assert (
            source.index("scheduler.shutdown(wait=False)")
            < source.index("await drain_jobs(SHUTDOWN_GRACE_S)")
            < source.index("await active_store.close()")
        )
