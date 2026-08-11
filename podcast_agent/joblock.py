"""Cross-process job leases held in CouchDB.

The runner already refuses to start a job that is running, using an
``asyncio.Lock`` per job. That lock covers one event loop in one process, which
was true of the whole system right up until it wasn't: a second instance started
by mistake, a one-off script, a `podcast-agent` CLI invocation against the same
database. The port check in :mod:`podcast_agent.main` closes the common case —
two servers on the same host and port — and nothing else.

Two runs of the same job against one database is not a crash, which is what
makes it worth guarding. It is two backfills paying for the same episode twice,
two digests racing to claim the same episodes, and telemetry that quietly counts
everything twice.

The lease is one document per job, taken with CouchDB's own MVCC: whoever writes
first wins and the loser sees a 409. No new dependency, no new failure mode.
It carries an expiry so a process killed mid-run cannot wedge the job forever,
and it is renewed from inside the run so a six-hour backfill does not lose the
lease it is still using.
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta

from .db import ConflictError, Doc, NotFoundError, Store
from .logging_setup import get_logger
from .utils import iso, iso_now, parse_iso, utcnow

log = get_logger(__name__)

LOCK_PREFIX = "control:lock:"

#: How long a lease survives without renewal. Long enough that a brief stall
#: does not hand the job to someone else, short enough that a `kill -9` costs
#: one window rather than a manual cleanup.
DEFAULT_TTL_S = 900

#: Renewals per lease lifetime. Three means two can be missed — to a paused
#: process, a slow database — before the lease is genuinely lost.
_RENEWALS_PER_TTL = 3


def lock_id(job: str) -> str:
    return f"{LOCK_PREFIX}{job}"


class LeaseLost(Exception):
    """The lease was taken by someone else while the job was still running."""


class JobLease:
    """One job's lease. Not reusable: construct one per run."""

    def __init__(self, store: Store, job: str, *, ttl_s: int = DEFAULT_TTL_S) -> None:
        self._store = store
        self._job = job
        self._ttl = ttl_s
        # Identifies this run, not this process: a process that crashed and came
        # back must not be able to renew or release a lease it no longer holds.
        self._owner = uuid.uuid4().hex
        self._held = False

    @property
    def owner(self) -> str:
        return self._owner

    def _doc(self) -> Doc:
        now = utcnow()
        return {
            "_id": lock_id(self._job),
            "type": "job_lock",
            "job": self._job,
            "owner": self._owner,
            "acquired_at": iso(now),
            "expires_at": iso(now + timedelta(seconds=self._ttl)),
            # Not used for decisions — only so an operator reading the document
            # can tell which machine is holding the thing they are waiting for.
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }

    async def acquire(self) -> bool:
        """Take the lease. False means someone else holds a live one."""
        existing = await self._store.get(lock_id(self._job))
        if existing is None:
            # create() is insert-only, so two processes arriving together cannot
            # both succeed — the loser gets False rather than an overwrite.
            self._held = await self._store.create(self._doc())
            if self._held:
                log.debug("joblock.acquired", job=self._job, owner=self._owner)
            return self._held

        expires = parse_iso(existing.get("expires_at"))
        if expires is not None and expires > utcnow():
            log.info(
                "joblock.busy",
                job=self._job,
                held_by_host=existing.get("host"),
                held_by_pid=existing.get("pid"),
                expires_at=existing.get("expires_at"),
            )
            return False

        # Expired, or malformed enough that its expiry cannot be read: take it
        # over, but only against the revision just read. A racing process that
        # got there first turns this into a 409, not a double acquisition.
        doc = self._doc()
        doc["_rev"] = existing["_rev"]
        doc["taken_over_from"] = existing.get("owner")
        try:
            await self._store.put(doc)
        except ConflictError:
            return False
        self._held = True
        log.warning(
            "joblock.taken_over",
            job=self._job,
            previous_host=existing.get("host"),
            previous_pid=existing.get("pid"),
            expired_at=existing.get("expires_at"),
            detail="the previous holder stopped renewing — it probably died mid-run",
        )
        return True

    async def renew(self) -> bool:
        """Extend the lease. False means it is no longer ours."""
        existing = await self._store.get(lock_id(self._job))
        if existing is None or existing.get("owner") != self._owner:
            return False
        doc = self._doc()
        doc["_rev"] = existing["_rev"]
        doc["acquired_at"] = existing.get("acquired_at") or doc["acquired_at"]
        try:
            await self._store.put(doc)
        except ConflictError:
            return False
        return True

    async def release(self) -> None:
        """Give up the lease, but only if it is still ours.

        The ownership check is the point. A process whose lease expired while it
        was stalled would otherwise delete its successor's lock on the way out,
        and the next job to ask would be told the lease is free while a run is
        very much in progress.
        """
        if not self._held:
            return
        self._held = False
        try:
            existing = await self._store.get(lock_id(self._job))
            if existing is None or existing.get("owner") != self._owner:
                log.warning(
                    "joblock.release_skipped",
                    job=self._job,
                    detail="the lease was taken over while this run was still going",
                )
                return
            await self._store.delete(existing["_id"], existing["_rev"])
        except (ConflictError, NotFoundError):
            # Someone else's write beat us to it; theirs is the current truth.
            return
        except Exception as exc:
            # Never let cleanup mask the job's own outcome. The lease expires on
            # its own, so the worst case is one TTL of unnecessary waiting.
            log.warning("joblock.release_failed", job=self._job, error=str(exc))

    async def _heartbeat(self) -> None:
        interval = max(1.0, self._ttl / _RENEWALS_PER_TTL)
        while True:
            await asyncio.sleep(interval)
            try:
                if await self.renew():
                    continue
            except Exception as exc:
                log.warning("joblock.renew_failed", job=self._job, error=str(exc))
                continue
            # Renewal was refused, which means another process now holds the
            # lease and is very likely doing this job in parallel. The run is not
            # cancelled: interrupting a summarisation mid-flight to fix a
            # bookkeeping problem trades a real cost for a theoretical one. It is
            # logged at error because it should never happen, and if it does the
            # TTL is too short for how long this job actually takes.
            log.error(
                "joblock.lease_lost",
                job=self._job,
                ttl_s=self._ttl,
                detail="another process took the lease while this run was still going",
            )
            return


@asynccontextmanager
async def held(store: Store, job: str, *, ttl_s: int = DEFAULT_TTL_S) -> AsyncIterator[JobLease]:
    """Hold the lease for the duration of the block, renewing in the background.

    Raises :class:`LeaseLost` up front when the lease cannot be taken, so callers
    can translate it into whatever "already running" means to them.
    """
    lease = JobLease(store, job, ttl_s=ttl_s)
    if not await lease.acquire():
        raise LeaseLost(f"{job} is already running in another process")
    beat = asyncio.create_task(lease._heartbeat(), name=f"joblock-{job}")
    try:
        yield lease
    finally:
        beat.cancel()
        # Awaited, not just cancelled: a task still winding down could otherwise
        # land a renewal after the release and resurrect a lease nobody holds.
        with suppress(asyncio.CancelledError):
            await beat
        await lease.release()


def _alive(pid: object) -> bool:
    """Whether ``pid`` names a running process on this machine."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process, but a process. Alive for our purposes.
        return True
    return True


async def reclaim_local_leases(store: Store) -> int:
    """Drop leases left behind by dead processes on this host. Returns the count.

    Called once at startup, before any job can run. A graceful shutdown releases
    its own leases; a `kill -9`, an OOM, or a crash during cleanup does not, and
    the job then stays locked for the rest of its TTL — long enough for the
    backfill's 20-minute poll to miss a cycle for no reason.

    Restricted to this host, and only to pids that are gone. Another machine's
    lease is not ours to judge: we cannot see whether its holder is alive, and
    guessing wrong means two processes running the same job.
    """
    host = socket.gethostname()
    reclaimed = 0
    try:
        docs = await store.find({"type": "job_lock"}, limit=50)
    except Exception as exc:
        log.warning("joblock.reclaim_failed", error=str(exc))
        return 0

    for doc in docs:
        if doc.get("host") != host or _alive(doc.get("pid")):
            continue
        try:
            await store.delete(doc["_id"], doc["_rev"])
        except (ConflictError, NotFoundError):
            continue  # someone else got there first, which is fine
        reclaimed += 1
        log.warning(
            "joblock.reclaimed",
            job=doc.get("job"),
            previous_pid=doc.get("pid"),
            expires_at=doc.get("expires_at"),
            detail="left behind by a process that died without releasing it",
        )
    return reclaimed


async def current_holders(store: Store) -> list[Doc]:
    """Every live lease, for the status endpoint. Never raises."""
    try:
        docs = await store.find({"type": "job_lock"}, limit=50)
    except Exception as exc:
        log.debug("joblock.list_failed", error=str(exc))
        return []
    now = iso_now()
    return [d for d in docs if str(d.get("expires_at") or "") > now]
