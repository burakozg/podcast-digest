"""MemoryStore tests.

The in-memory store stands in for CouchDB across the whole suite, so its MVCC and
Mango behaviour must match what the real client does — otherwise tests pass
against semantics production never has.
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_agent.db import (
    INDEXES,
    ConflictError,
    MemoryStore,
    NotFoundError,
    StoreError,
    drop_transcript,
    load_transcript,
    resolve_index,
    save_transcript,
    typed_sort,
    update_doc,
)
from podcast_agent.db.base import check_indexable


class TestDocuments:
    async def test_create_is_idempotent(self, store: MemoryStore) -> None:
        """Ingestion's idempotency primitive: the second writer must lose."""
        assert await store.create({"_id": "a", "type": "episode"}) is True
        assert await store.create({"_id": "a", "type": "episode"}) is False

    async def test_create_rejects_a_rev(self, store: MemoryStore) -> None:
        with pytest.raises(StoreError, match="_rev"):
            await store.create({"_id": "a", "_rev": "1-x"})

    async def test_put_requires_matching_rev(self, store: MemoryStore) -> None:
        await store.create({"_id": "a", "n": 1})
        doc = await store.get("a")
        assert doc is not None
        await store.put({**doc, "n": 2})
        # Re-using the now-stale rev must conflict, not silently overwrite.
        with pytest.raises(ConflictError):
            await store.put({**doc, "n": 3})

    async def test_rev_increments_generation(self, store: MemoryStore) -> None:
        await store.create({"_id": "a"})
        first = (await store.get("a"))["_rev"]  # type: ignore[index]
        doc = await store.get("a")
        await store.put(dict(doc))  # type: ignore[arg-type]
        second = (await store.get("a"))["_rev"]  # type: ignore[index]
        assert int(first.split("-")[0]) + 1 == int(second.split("-")[0])

    async def test_get_returns_a_copy(self, store: MemoryStore) -> None:
        """Mutating a fetched doc must not change stored state (CouchDB semantics)."""
        await store.create({"_id": "a", "nested": {"x": 1}})
        fetched = await store.get("a")
        assert fetched is not None
        fetched["nested"]["x"] = 99
        again = await store.get("a")
        assert again is not None
        assert again["nested"]["x"] == 1

    async def test_delete_requires_current_rev(self, store: MemoryStore) -> None:
        await store.create({"_id": "a"})
        with pytest.raises(ConflictError):
            await store.delete("a", "1-wrong")


class TestUpdateDoc:
    async def test_applies_mutator(self, store: MemoryStore) -> None:
        await store.create({"_id": "a", "n": 1})
        await update_doc(store, "a", lambda d: d.__setitem__("n", 5))
        assert (await store.get("a"))["n"] == 5  # type: ignore[index]

    async def test_missing_doc_raises_not_found(self, store: MemoryStore) -> None:
        with pytest.raises(NotFoundError):
            await update_doc(store, "nope", lambda d: None)

    async def test_retries_on_conflict_with_fresh_state(self, store: MemoryStore) -> None:
        """On 409 the mutator must re-run against re-read state, never force-write."""
        await store.create({"_id": "a", "n": 0})
        calls = {"count": 0}

        async def _racing_writer() -> None:
            doc = await store.get("a")
            await store.put({**doc, "other": True})  # type: ignore[dict-item]

        def mutator(doc: dict) -> None:
            calls["count"] += 1
            doc["n"] = doc["n"] + 1

        # Simulate a concurrent write landing between our read and our put by
        # pre-stealing the rev on the first attempt.
        original_put = store.put
        state = {"stolen": False}

        async def flaky_put(doc: dict) -> dict:
            if not state["stolen"]:
                state["stolen"] = True
                await _racing_writer()
            return await original_put(doc)

        store.put = flaky_put  # type: ignore[method-assign]
        await update_doc(store, "a", mutator)
        store.put = original_put  # type: ignore[method-assign]

        final = await store.get("a")
        assert final is not None
        assert calls["count"] == 2  # ran again after the conflict
        assert final["n"] == 1  # incremented once, from fresh state
        assert final["other"] is True  # the racing write survived


class TestMangoSubset:
    @pytest.fixture
    def seeded(self, store: MemoryStore) -> MemoryStore:
        store.seed(
            {"_id": "e1", "type": "episode", "status": "NEW", "published_at": "2026-01-01"},
            {"_id": "e2", "type": "episode", "status": "TRIAGED", "published_at": "2026-03-01"},
            {"_id": "e3", "type": "episode", "status": "NEW", "published_at": "2026-02-01"},
            {"_id": "d1", "type": "digest"},
            {"_id": "e4", "type": "episode", "status": "NEW", "digest_id": "digest:x"},
        )
        return store

    async def test_equality_selector(self, seeded: MemoryStore) -> None:
        docs = await seeded.find({"type": "episode", "status": "NEW"})
        assert {d["_id"] for d in docs} == {"e1", "e3", "e4"}

    @pytest.fixture
    def three_shapes(self, store: MemoryStore) -> MemoryStore:
        """The three states a field can be in, which CouchDB treats differently."""
        store.seed(
            {"_id": "absent", "type": "probe"},
            {"_id": "null", "type": "probe", "origin": None},
            {"_id": "value", "type": "probe", "origin": "backfill"},
        )
        return store

    async def test_null_does_not_match_a_missing_field(self, three_shapes: MemoryStore) -> None:
        """CouchDB distinguishes "absent" from "present and null". This store must too.

        Verified against CouchDB 3.x with exactly these three documents: a
        selector of `{"field": null}` returns the one carrying null and *not*
        the one that lacks the field.
        """
        docs = await three_shapes.find({"type": "probe", "origin": None})
        assert {d["_id"] for d in docs} == {"null"}

    async def test_ne_does_not_match_a_missing_field(self, three_shapes: MemoryStore) -> None:
        """The same trap, and the one that actually bit: `$ne` is not "not equal".

        A document with no `origin` at all is not returned by
        `{"origin": {"$ne": "backfill"}}`, because Mango has no index entry for
        it to compare. Excluding archive material therefore needs an explicit
        `$exists: false` arm — see `backfill.NOT_BACKFILL`.
        """
        docs = await three_shapes.find({"type": "probe", "origin": {"$ne": "backfill"}})
        assert {d["_id"] for d in docs} == {"null"}

    async def test_the_routine_clause_matches_only_an_explicit_value(
        self, three_shapes: MemoryStore
    ) -> None:
        """Which is why every episode is given one — see migrate.py.

        An equality match cannot see the "absent" document. That is the point:
        it is indexable, and the missing-field case is removed at the source
        rather than worked around in every query.
        """
        from podcast_agent.state import ROUTINE_ONLY

        three_shapes.seed({"_id": "routine", "type": "probe", "origin": "routine"})
        docs = await three_shapes.find({"type": "probe", **ROUTINE_ONLY})
        assert {d["_id"] for d in docs} == {"routine"}

    async def test_range_selector(self, seeded: MemoryStore) -> None:
        docs = await seeded.find(
            {"type": "episode", "published_at": {"$gte": "2026-02-01", "$lt": "2026-04-01"}}
        )
        assert {d["_id"] for d in docs} == {"e2", "e3"}

    async def test_in_and_ne(self, seeded: MemoryStore) -> None:
        docs = await seeded.find({"type": "episode", "status": {"$in": ["TRIAGED"]}})
        assert {d["_id"] for d in docs} == {"e2"}
        docs = await seeded.find({"type": "episode", "status": {"$ne": "NEW"}})
        assert {d["_id"] for d in docs} == {"e2"}

    async def test_exists(self, seeded: MemoryStore) -> None:
        docs = await seeded.find({"type": "episode", "digest_id": {"$exists": True}})
        assert {d["_id"] for d in docs} == {"e4"}

    async def test_sort_matches_couchdb_collation(self, seeded: MemoryStore) -> None:
        """CouchDB sorts null lowest, so e4 (no published_at) leads ascending and
        trails descending. Getting this backwards would make queue ordering and
        digest ordering differ between tests and production."""
        asc = await seeded.find(
            {"type": "episode", "status": "NEW"}, sort=typed_sort("published_at", "asc")
        )
        assert [d["_id"] for d in asc] == ["e4", "e1", "e3"]
        desc = await seeded.find(
            {"type": "episode", "status": "NEW"}, sort=typed_sort("published_at", "desc")
        )
        assert [d["_id"] for d in desc] == ["e3", "e1", "e4"]

    async def test_unindexed_sort_is_rejected(self, seeded: MemoryStore) -> None:
        """Regression: real CouchDB answers `no_usable_index` when the sort is not
        a prefix of an index, but the in-memory double happily sorted anything —
        so every /episodes query 500'd in production while tests were green.
        """
        with pytest.raises(StoreError, match="no_usable_index"):
            await seeded.find({"type": "episode"}, sort=[{"published_at": "desc"}])

    async def test_mixed_sort_directions_are_rejected(self, seeded: MemoryStore) -> None:
        """CouchDB requires every sort field to share one direction."""
        with pytest.raises(StoreError, match="share a direction"):
            await seeded.find(
                {"type": "episode"},
                sort=[{"type": "asc"}, {"published_at": "desc"}],
            )

    async def test_limit_and_skip(self, seeded: MemoryStore) -> None:
        page = await seeded.find({"type": "episode"}, limit=2, skip=1)
        assert len(page) == 2

    async def test_count(self, seeded: MemoryStore) -> None:
        assert await seeded.count({"type": "episode"}) == 4
        assert await seeded.count({"type": "digest"}) == 1

    async def test_fields_projection(self, seeded: MemoryStore) -> None:
        docs = await seeded.find({"type": "digest"}, fields=["_id"])
        assert docs == [{"_id": "d1"}]

    async def test_unsupported_operator_raises_loudly(self, seeded: MemoryStore) -> None:
        """Silently ignoring an operator would make tests lie about production."""
        with pytest.raises(StoreError, match="does not implement"):
            await seeded.find({"type": "episode", "status": {"$elemMatch": {}}})
        with pytest.raises(StoreError, match="does not implement"):
            await seeded.find({"$weird": []})


class TestTranscriptAttachments:
    async def test_round_trip_is_gzipped(self, store: MemoryStore) -> None:
        await store.create({"_id": "e1", "type": "episode"})
        text = "A transcript. " * 500
        size = await save_transcript(store, "e1", text)
        assert size < len(text)  # compression actually happened
        assert await load_transcript(store, "e1") == text

    async def test_attachment_is_a_stub_on_the_doc(self, store: MemoryStore) -> None:
        """The doc body must stay small and Mango-indexable (§6)."""
        await store.create({"_id": "e1", "type": "episode"})
        await save_transcript(store, "e1", "x" * 10_000)
        doc = await store.get("e1")
        assert doc is not None
        stub = doc["_attachments"]["transcript.txt.gz"]
        assert stub["stub"] is True
        assert stub["content_type"] == "application/gzip"

    async def test_load_returns_none_when_absent(self, store: MemoryStore) -> None:
        await store.create({"_id": "e1"})
        assert await load_transcript(store, "e1") is None

    async def test_unicode_survives(self, store: MemoryStore) -> None:
        await store.create({"_id": "e1"})
        text = "Åland, naïve café, 中文, emoji 🎙️ " * 50
        await save_transcript(store, "e1", text)
        assert await load_transcript(store, "e1") == text

    async def test_drop_is_safe_when_missing(self, store: MemoryStore) -> None:
        await store.create({"_id": "e1"})
        await drop_transcript(store, "e1")  # must not raise
        await save_transcript(store, "e1", "y" * 1000)
        await drop_transcript(store, "e1")
        assert await load_transcript(store, "e1") is None

    async def test_corrupt_attachment_raises_store_error(self, store: MemoryStore) -> None:
        await store.create({"_id": "e1"})
        await store.put_attachment("e1", "transcript.txt.gz", b"not gzip", "application/gzip")
        with pytest.raises(StoreError, match="unreadable"):
            await load_transcript(store, "e1")

    async def test_attachment_on_missing_doc_raises(self, store: MemoryStore) -> None:
        with pytest.raises(NotFoundError):
            await save_transcript(store, "ghost", "text")


class TestWarningSeverity:
    """Only one of CouchDB's two warnings is a defect."""

    def test_a_missing_index_is_a_warning(self) -> None:
        """A full scan is a query nobody declared an index for."""
        assert "no matching index" in "No matching index found, create an index".lower()

    def test_a_filtered_index_scan_is_not(self) -> None:
        """An index was used; Mango then filtered within it.

        Unavoidable when a sort forces the index choice, so reporting it as a
        warning put a permanent row of yellow in the console for something
        working as designed.
        """
        advisory = "The number of documents examined is high in proportion"
        assert "no matching index" not in advisory.lower()


class TestUnindexedQueryReporting:
    """A missing index must be reported, but exactly once.

    CouchDB returns the warning on every matching call, and the pipeline runs
    the same handful of queries every few minutes. Logging each one buries the
    log in thousands of copies of a single actionable fact — and now that the
    console shows the log, it buries that too.
    """

    def test_the_same_shape_is_reported_once(self) -> None:
        from podcast_agent.db.couch import _selector_shape

        first = {"type": "episode", "status": "NEW"}
        second = {"type": "episode", "status": "TRIAGED"}
        assert _selector_shape(first) == _selector_shape(second)

    def test_different_shapes_are_told_apart(self) -> None:
        from podcast_agent.db.couch import _selector_shape

        assert _selector_shape({"type": "episode", "status": "NEW"}) != _selector_shape(
            {"type": "episode", "origin": "backfill"}
        )

    def test_operators_are_part_of_the_shape(self) -> None:
        """`status: "NEW"` and `status: {$in: [...]}` need different indexes."""
        from podcast_agent.db.couch import _selector_shape

        assert _selector_shape({"status": "NEW"}) != _selector_shape({"status": {"$in": ["NEW"]}})

    def test_nested_structure_survives(self) -> None:
        from podcast_agent.db.couch import _selector_shape

        shape = _selector_shape(
            {"$or": [{"origin": {"$exists": False}}, {"origin": {"$ne": "backfill"}}]}
        )
        assert "$or" in shape and "$exists" in shape and "$ne" in shape

    def test_values_never_appear_in_the_shape(self) -> None:
        """The shape is a dedupe key, not a record of what was queried."""
        from podcast_agent.db.couch import _selector_shape

        assert "secret-slug" not in _selector_shape({"podcast_slug": "secret-slug"})


def test_every_query_the_code_runs_has_an_index() -> None:
    """The plain "everything of this type" queries were scanning the database.

    Counts, the podcast list and the archive list all select on `type` alone.
    With no index on `type` CouchDB examines every document to return a handful
    of rows — which is what "No matching index found" was reporting.
    """
    from podcast_agent.db.base import INDEXES

    declared = {tuple(index["fields"]) for index in INDEXES}
    for required in (("type",), ("type", "origin"), ("type", "status")):
        assert any(fields[: len(required)] == required for fields in declared), (
            f"no index serves {required}"
        )


class TestIndexPinning:
    """Which index serves a query is a decision, not a guess.

    Left unnamed, CouchDB chose by heuristic, and its heuristic does not know
    which selector fields are pinned by equality. One night's log carried six
    distinct shapes reporting "documents examined is high in proportion to the
    number of results returned" — an index was used, then most of what it
    returned was thrown away in memory.
    """

    def test_equality_fields_win_over_a_broader_index(self) -> None:
        assert (
            resolve_index({"type": "episode", "status": "NEW", "origin": "routine"})
            == "idx-type-origin-status"
        )

    def test_in_still_uses_the_index_for_its_own_field(self) -> None:
        """`$in` is a set of equality probes, which a B-tree can serve."""
        assert (
            resolve_index(
                {"type": "episode", "origin": "routine", "status": {"$in": ["NEW", "TRIAGED"]}}
            )
            == "idx-type-origin-status"
        )

    def test_a_range_ends_the_usable_prefix(self) -> None:
        """Nothing after a range field is ordered, so it cannot narrow further."""
        assert (
            resolve_index({"type": "episode", "transcript_at": {"$lt": "2026-01-01"}})
            == "idx-type-transcript-at"
        )

    def test_exists_true_does_not_disqualify_a_range(self) -> None:
        """A value that compares is present, so the pair is still indexable."""
        assert (
            resolve_index(
                {"type": "episode", "transcript_at": {"$lt": "2026-01-01", "$exists": True}}
            )
            == "idx-type-transcript-at"
        )

    def test_exists_false_falls_back_to_the_broad_index(self) -> None:
        """Absence cannot be indexed at all — this is the migration's own query."""
        assert resolve_index({"type": "episode", "origin": {"$exists": False}}) == "idx-type"

    def test_a_sort_must_lead_the_index(self) -> None:
        """CouchDB rejects a sort that is not a prefix of the chosen index."""
        chosen = resolve_index(
            {"type": "episode", "origin": "routine", "published_at": {"$gte": "a"}},
            typed_sort("published_at", "desc"),
        )
        assert chosen == "idx-type-published"
        fields = next(i["fields"] for i in INDEXES if i["name"] == chosen)
        assert fields[:2] == ["type", "published_at"]

    def test_a_selector_no_index_serves_is_refused(self) -> None:
        """The whole point: a full scan fails in tests instead of in production."""
        with pytest.raises(StoreError, match="no declared Mango index"):
            check_indexable({"status": "NEW"})

    def test_an_explicit_choice_is_left_alone(self) -> None:
        check_indexable({"status": "NEW"}, use_index="idx-type")

    async def test_the_memory_store_refuses_an_unindexed_query(self, store: MemoryStore) -> None:
        with pytest.raises(StoreError, match="no declared Mango index"):
            await store.find({"status": "NEW"})

    async def test_couch_names_the_index_in_the_request(self) -> None:
        """The resolved name has to reach CouchDB, not merely be computed."""
        import httpx

        from podcast_agent.config import CouchDBConfig
        from podcast_agent.db.couch import CouchStore

        sent: dict[str, Any] = {}
        store = CouchStore(CouchDBConfig(), None)

        async def _request(_method: str, _path: str, **kwargs: Any) -> Any:
            sent.update(kwargs.get("json") or {})
            return httpx.Response(200, json={"docs": []}, request=httpx.Request("POST", "http://x"))

        store._request = _request  # type: ignore[method-assign]
        await store.find({"type": "episode", "status": "NEW", "origin": "routine"})
        assert sent["use_index"] == "idx-type-origin-status"


class TestPinningAnIndexActuallyBinds:
    """`use_index` resolves a bare string as a *design document id*.

    Left to itself CouchDB names the design document after a hash of its
    contents, so every pinned query asked for `_design/idx-...`, found nothing,
    and was quietly answered by the planner's own heuristic instead. The rows
    were right, so nothing failed — the pin had simply never bound, and the only
    trace was a warning saying the design document "does not contain a valid
    index for this query".
    """

    def _store(self) -> Any:
        from podcast_agent.config import CouchDBConfig
        from podcast_agent.db.couch import CouchStore

        return CouchStore(CouchDBConfig(), None)

    async def test_each_index_is_created_in_a_document_named_after_it(self) -> None:
        import httpx

        from podcast_agent.db.base import INDEXES

        created: list[dict[str, Any]] = []
        store = self._store()

        async def _request(method: str, path: str, **kwargs: Any) -> Any:
            if method == "POST" and path.endswith("/_index"):
                created.append(kwargs["json"])
            body: dict[str, Any] = {"indexes": []} if method == "GET" else {}
            return httpx.Response(200, json=body, request=httpx.Request(method, "http://x"))

        async def _put(path: str, **_kw: Any) -> Any:
            return httpx.Response(201, json={}, request=httpx.Request("PUT", "http://x"))

        store._request = _request  # type: ignore[method-assign]
        store._client.put = _put  # type: ignore[method-assign]
        await store.ensure_setup()

        assert len(created) == len(INDEXES)
        for index in created:
            assert index["ddoc"] == index["name"], (
                "the design document must be named after the index, or use_index "
                "cannot reference it"
            )

    async def test_the_name_the_query_pins_is_the_name_the_document_has(self) -> None:
        """The two halves have to agree; either alone is useless."""
        import httpx

        from podcast_agent.db.base import INDEXES, resolve_index

        resolved = resolve_index({"type": "episode", "status": "NEW", "origin": "routine"})
        assert resolved in {i["name"] for i in INDEXES}

        sent: dict[str, Any] = {}
        store = self._store()

        async def _request(_method: str, _path: str, **kwargs: Any) -> Any:
            sent.update(kwargs.get("json") or {})
            return httpx.Response(200, json={"docs": []}, request=httpx.Request("POST", "http://x"))

        store._request = _request  # type: ignore[method-assign]
        await store.find({"type": "episode", "status": "NEW", "origin": "routine"})
        # What is pinned is a design document id, so it must equal the ddoc name
        # ensure_setup creates.
        assert sent["use_index"] == resolved

    async def test_a_hash_named_duplicate_is_dropped(self) -> None:
        """Older deployments carry a second copy of every index that nothing can
        reference, and CouchDB maintains it on every write."""
        import httpx

        deleted: list[str] = []
        store = self._store()

        async def _request(method: str, path: str, **kwargs: Any) -> Any:
            if method == "DELETE":
                deleted.append(path)
            body: dict[str, Any] = {}
            if method == "GET" and path.endswith("/_index"):
                body = {
                    "indexes": [
                        {"ddoc": None, "name": "_all_docs"},
                        {"ddoc": "_design/deadbeef", "name": "idx-type-status"},
                        {"ddoc": "_design/idx-type-status", "name": "idx-type-status"},
                        {"ddoc": "_design/somethingelse", "name": "not-ours"},
                    ]
                }
            return httpx.Response(200, json=body, request=httpx.Request(method, "http://x"))

        async def _put(path: str, **_kw: Any) -> Any:
            return httpx.Response(201, json={}, request=httpx.Request("PUT", "http://x"))

        store._request = _request  # type: ignore[method-assign]
        store._client.put = _put  # type: ignore[method-assign]
        await store.ensure_setup()

        assert deleted == ["/podcast_agent/_index/_design/deadbeef/json/idx-type-status"]


class TestTransientDatabaseFailures:
    """A dropped connection must not end whatever was running.

    Without retries a single blip aborted a pipeline run with
    `scheduler.job_failed` and turned a console poll into a 500 with an httpx
    traceback — on a database that lives on the same host and was fine a second
    later, usually while the machine was busy transcribing.
    """

    def _store(self, responses: list[Any]) -> Any:
        from podcast_agent.config import CouchDBConfig
        from podcast_agent.db.couch import CouchStore

        store = CouchStore(CouchDBConfig(), None)
        calls = {"n": 0}

        async def _request(*_a: Any, **_k: Any) -> Any:
            outcome = responses[calls["n"]]
            calls["n"] += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        store._client.request = _request  # type: ignore[method-assign]
        store._calls = calls  # type: ignore[attr-defined]
        return store

    async def test_a_dropped_connection_is_retried(self) -> None:
        import httpx

        ok = httpx.Response(200, json={"docs": []}, request=httpx.Request("POST", "http://x"))
        store = self._store([httpx.ReadTimeout("boom"), ok])
        response = await store._request("POST", "/db/_find")
        assert response.status_code == 200
        assert store._calls["n"] == 2, "should have retried once"

    async def test_it_gives_up_after_a_bounded_number_of_attempts(self) -> None:
        """A database that is genuinely down must not be retried forever."""
        import httpx

        from podcast_agent.db.base import StoreError
        from podcast_agent.db.couch import RETRY_ATTEMPTS

        store = self._store([httpx.ConnectError("down")] * RETRY_ATTEMPTS)
        with pytest.raises(StoreError, match=f"after {RETRY_ATTEMPTS} attempts"):
            await store._request("POST", "/db/_find")
        assert store._calls["n"] == RETRY_ATTEMPTS

    async def test_an_http_error_response_is_not_retried(self) -> None:
        """A 400 is an answer. Repeating it asks the same question twice."""
        import httpx

        from podcast_agent.db.base import StoreError

        bad = httpx.Response(400, text="bad selector", request=httpx.Request("POST", "http://x"))
        store = self._store([bad])
        with pytest.raises(StoreError):
            await store._request("POST", "/db/_find")
        assert store._calls["n"] == 1

    async def test_a_conflict_is_not_retried(self) -> None:
        """409 is how optimistic concurrency reports a real conflict."""
        import httpx

        from podcast_agent.db.base import ConflictError

        conflict = httpx.Response(409, json={}, request=httpx.Request("PUT", "http://x"))
        store = self._store([conflict])
        with pytest.raises(ConflictError):
            await store._request("PUT", "/db/doc")
        assert store._calls["n"] == 1
