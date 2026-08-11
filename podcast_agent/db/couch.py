"""Thin async CouchDB client over httpx (§6). No ORM, no magic."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx

from ..config import CouchDBConfig
from ..logging_setup import get_logger
from .base import (
    INDEXES,
    ConflictError,
    Doc,
    NotFoundError,
    Selector,
    StoreError,
    resolve_index,
)

log = get_logger(__name__)

#: Transport failures are retried this many times before giving up. Three is
#: enough to ride out a moment of contention without hiding a database that is
#: genuinely down for a minute.
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 0.25


def _selector_shape(selector: Any) -> str:
    """A selector's structure with its values removed.

    Two queries differing only in a status value are the same missing index, so
    they should be reported once between them.
    """
    if isinstance(selector, dict):
        return (
            "{" + ",".join(f"{k}:{_selector_shape(v)}" for k, v in sorted(selector.items())) + "}"
        )
    if isinstance(selector, list):
        return "[" + ",".join(_selector_shape(v) for v in selector) + "]"
    return "?"


class CouchStore:
    """CouchDB-backed :class:`~podcast_agent.db.base.Store` implementation."""

    def __init__(self, cfg: CouchDBConfig, password: str | None, *, timeout: float = 30.0) -> None:
        self._db = cfg.db
        #: Query shapes already reported as unindexed, so each is logged once.
        self._warned: set[tuple[str, str]] = set()
        auth = (cfg.user, password) if password else None
        self._client = httpx.AsyncClient(
            base_url=cfg.url.rstrip("/"),
            auth=auth,
            timeout=httpx.Timeout(timeout),
            headers={"Accept": "application/json"},
            # CouchDB is an internal service; a redirect would mean a misconfiguration.
            follow_redirects=False,
        )

    # --- internals ----------------------------------------------------------

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        response = await self._request_with_retry(method, url, **kwargs)
        if response.status_code == 409:
            raise ConflictError(f"{method} {url}: document update conflict")
        if response.status_code == 404:
            raise NotFoundError(f"{method} {url}: not found")
        if response.status_code >= 400:
            # Response bodies from CouchDB are safe to surface (no secrets).
            raise StoreError(
                f"CouchDB {method} {url} -> {response.status_code}: {response.text[:400]}"
            )
        return response

    async def _request_with_retry(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send, retrying transport failures a few times.

        A dropped connection or a read timeout used to abort whatever was
        running: a pipeline run died with `scheduler.job_failed`, and a console
        request became a 500 with a traceback. The database is on the same host
        as everything else here, so a blip is exactly that — a blip, usually
        while the machine is busy transcribing.

        Only *transport* errors are retried. An HTTP response, whatever its
        status, is an answer: a 409 is a real conflict and a 400 is a real bad
        request, and repeating either just asks the same question twice.

        Retrying a write is safe here because CouchDB writes carry `_rev` or an
        explicit `_id`: if the first attempt did land, the retry loses to a 409
        rather than applying twice.
        """
        delay = RETRY_BASE_DELAY
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return await self._client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                if attempt == RETRY_ATTEMPTS:
                    raise StoreError(
                        f"CouchDB {method} {url} failed after {attempt} attempts: {exc}"
                    ) from exc
                log.warning(
                    "couchdb.request_retry",
                    method=method,
                    url=url,
                    attempt=attempt,
                    of=RETRY_ATTEMPTS,
                    error=str(exc)[:200],
                )
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    # --- setup --------------------------------------------------------------

    async def ensure_setup(self) -> None:
        create = await self._client.put(f"/{self._db}")
        if create.status_code in (201, 202):
            log.info("couchdb.database_created", db=self._db)
        elif create.status_code == 412:
            log.debug("couchdb.database_exists", db=self._db)
        elif create.status_code >= 400:
            raise StoreError(
                f"could not create database {self._db!r}: {create.status_code} {create.text[:300]}"
            )
        for index in INDEXES:
            await self._ensure_index(index)
        await self._drop_unpinnable_indexes()

    async def _ensure_index(self, index: dict[str, Any]) -> None:
        # CouchDB treats _index as create-or-noop when name+definition match.
        #
        # `ddoc` is named explicitly, and that is load-bearing rather than
        # cosmetic: `use_index` resolves a bare string as a *design document
        # id*, not an index name. Left to CouchDB the design document gets a
        # hash for a name, so every pinned query asked for `_design/idx-...`,
        # found nothing, and was answered by the planner's own heuristic —
        # silently, since the query still returns the right rows. Naming the
        # design document after the index is what makes the pin bind.
        await self._request(
            "POST",
            f"/{self._db}/_index",
            json={
                "index": {"fields": index["fields"]},
                "name": index["name"],
                "ddoc": index["name"],
                "type": "json",
            },
        )
        log.debug("couchdb.index_ready", name=index["name"])

    async def _drop_unpinnable_indexes(self) -> None:
        """Remove older copies of our indexes that live in hash-named documents.

        Deployments created before the design documents were named carry a
        duplicate of every index: same fields, unusable name. CouchDB would go
        on maintaining both on every write, for one that nothing can reference.
        """
        try:
            listing = await self._request("GET", f"/{self._db}/_index")
        except StoreError as exc:  # pragma: no cover - listing is not critical
            log.debug("couchdb.index_listing_failed", error=str(exc))
            return
        ours = {index["name"] for index in INDEXES}
        for entry in listing.json().get("indexes", []):
            ddoc, name = entry.get("ddoc"), entry.get("name")
            if not ddoc or name not in ours or ddoc == f"_design/{name}":
                continue
            try:
                await self._request("DELETE", f"/{self._db}/_index/{ddoc}/json/{name}")
                log.info("couchdb.stale_index_dropped", name=name, ddoc=ddoc)
            except StoreError as exc:  # pragma: no cover - best effort
                log.warning("couchdb.stale_index_not_dropped", name=name, error=str(exc))

    async def ping(self) -> bool:
        try:
            response = await self._client.get(f"/{self._db}")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    # --- documents ----------------------------------------------------------

    async def get(self, doc_id: str) -> Doc | None:
        try:
            response = await self._request("GET", f"/{self._db}/{_quote(doc_id)}")
        except NotFoundError:
            return None
        return dict(response.json())

    async def put(self, doc: Doc) -> Doc:
        doc_id = doc.get("_id")
        if not doc_id:
            raise StoreError("put() requires an _id")
        response = await self._request("PUT", f"/{self._db}/{_quote(doc_id)}", json=doc)
        body = response.json()
        # Keep the caller's in-memory doc usable for a subsequent write.
        doc["_rev"] = body["rev"]
        return {"id": body["id"], "rev": body["rev"]}

    async def create(self, doc: Doc) -> bool:
        doc_id = doc.get("_id")
        if not doc_id:
            raise StoreError("create() requires an _id")
        if "_rev" in doc:
            raise StoreError("create() must not be called with a _rev")
        try:
            await self._request("PUT", f"/{self._db}/{_quote(doc_id)}", json=doc)
        except ConflictError:
            return False
        return True

    async def delete(self, doc_id: str, rev: str) -> None:
        try:
            await self._request("DELETE", f"/{self._db}/{_quote(doc_id)}", params={"rev": rev})
        except NotFoundError:
            return

    async def find(
        self,
        selector: Selector,
        *,
        fields: list[str] | None = None,
        sort: list[dict[str, str]] | None = None,
        limit: int = 100,
        skip: int = 0,
        use_index: str | None = None,
    ) -> list[Doc]:
        body: dict[str, Any] = {"selector": selector, "limit": limit}
        if fields:
            body["fields"] = fields
        if sort:
            body["sort"] = sort
        if skip:
            body["skip"] = skip
        # Naming the index rather than leaving CouchDB's heuristic to guess it.
        if index := (use_index or resolve_index(selector, sort)):
            body["use_index"] = index
        response = await self._request("POST", f"/{self._db}/_find", json=body)
        payload = response.json()
        if warning := payload.get("warning"):
            # Usually "no matching index found" — a missing index silently turns
            # into a full scan, which is exactly what we must not ship.
            #
            # Reported once per distinct query shape. CouchDB returns it on every
            # matching call, and the pipeline runs the same handful of queries
            # every few minutes: unfiltered, one missing index buries the log in
            # thousands of copies of itself and hides everything else. Once is
            # what makes it actionable; the rest is noise.
            self._warn_once(warning, selector)
        return [dict(d) for d in payload.get("docs", [])]

    def _warn_once(self, warning: str, selector: Selector) -> None:
        shape = (warning.split(".")[0], _selector_shape(selector))
        if shape in self._warned:
            return
        self._warned.add(shape)

        # CouchDB says two different things here and only one is a defect.
        #
        # "No matching index found" means a full scan — a query nobody declared
        # an index for, which is a bug to fix. "Documents examined is high" is
        # advisory: an index *was* used, and Mango then filtered within it. That
        # happens whenever a sort forces the index choice, which for a query
        # sorted by published_at and filtered on status is unavoidable without
        # pinning an index per query. Logging it at warning put a permanent row
        # of yellow in the console for something that is working as designed.
        missing_index = "no matching index" in warning.lower()
        emit = log.warning if missing_index else log.info
        emit(
            "couchdb.find_warning" if missing_index else "couchdb.find_unindexed_filter",
            warning=warning,
            selector=selector,
            note="reported once per query shape for this process",
        )

    async def count(self, selector: Selector) -> int:
        # Mango has no COUNT; fetch ids only in pages and tally.
        total = 0
        skip = 0
        page = 1000
        while True:
            docs = await self.find(selector, fields=["_id"], limit=page, skip=skip)
            total += len(docs)
            if len(docs) < page:
                return total
            skip += page

    # --- attachments --------------------------------------------------------

    async def put_attachment(self, doc_id: str, name: str, data: bytes, content_type: str) -> None:
        doc = await self.get(doc_id)
        if doc is None:
            raise NotFoundError(doc_id)
        await self._request(
            "PUT",
            f"/{self._db}/{_quote(doc_id)}/{_quote(name)}",
            params={"rev": doc["_rev"]},
            content=data,
            headers={"Content-Type": content_type},
        )

    async def get_attachment(self, doc_id: str, name: str) -> bytes | None:
        try:
            response = await self._request("GET", f"/{self._db}/{_quote(doc_id)}/{_quote(name)}")
        except NotFoundError:
            return None
        return response.content

    async def delete_attachment(self, doc_id: str, name: str) -> None:
        doc = await self.get(doc_id)
        if doc is None:
            raise NotFoundError(doc_id)
        if name not in (doc.get("_attachments") or {}):
            return
        await self._request(
            "DELETE",
            f"/{self._db}/{_quote(doc_id)}/{_quote(name)}",
            params={"rev": doc["_rev"]},
        )

    async def close(self) -> None:
        await self._client.aclose()


def _quote(component: str) -> str:
    """Percent-encode a single path component.

    Document ids here are machine-generated (``episode:<sha256>``) but encoding
    keeps a hand-built id from escaping the path.
    """
    return quote(component, safe="")
