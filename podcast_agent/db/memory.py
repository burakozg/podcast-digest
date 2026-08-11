"""In-memory :class:`Store` implementation.

Used by the test suite (no network, no testcontainers) and available for local
dry runs. Implements the subset of Mango that the pipeline actually issues; an
unsupported operator raises loudly so a query can never silently return wrong
rows in tests while working differently against real CouchDB.
"""

from __future__ import annotations

import copy
import re
import uuid
from typing import Any

from .base import (
    ConflictError,
    Doc,
    NotFoundError,
    Selector,
    StoreError,
    check_indexable,
    check_sortable,
)

_SUPPORTED_OPERATORS = frozenset(
    {"$eq", "$ne", "$in", "$nin", "$lt", "$lte", "$gt", "$gte", "$exists", "$regex"}
)


class MemoryStore:
    """Dict-backed document store with MVCC-style ``_rev`` checking."""

    def __init__(self) -> None:
        self._docs: dict[str, Doc] = {}
        self._attachments: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.setup_called = False

    # --- setup --------------------------------------------------------------

    async def ensure_setup(self) -> None:
        self.setup_called = True

    async def ping(self) -> bool:
        return True

    # --- documents ----------------------------------------------------------

    async def get(self, doc_id: str) -> Doc | None:
        doc = self._docs.get(doc_id)
        return copy.deepcopy(doc) if doc is not None else None

    async def put(self, doc: Doc) -> Doc:
        doc_id = doc.get("_id")
        if not doc_id:
            raise StoreError("put() requires an _id")
        existing = self._docs.get(doc_id)
        if existing is not None and existing.get("_rev") != doc.get("_rev"):
            raise ConflictError(f"{doc_id}: document update conflict")
        if existing is None and doc.get("_rev"):
            raise ConflictError(f"{doc_id}: _rev supplied for a non-existent document")
        stored = copy.deepcopy(doc)
        stored["_rev"] = _next_rev(existing.get("_rev") if existing else None)
        # Attachments live outside the doc body, mirroring CouchDB.
        if existing is not None and "_attachments" in existing:
            stored["_attachments"] = existing["_attachments"]
        self._docs[doc_id] = stored
        doc["_rev"] = stored["_rev"]
        return {"id": doc_id, "rev": stored["_rev"]}

    async def create(self, doc: Doc) -> bool:
        doc_id = doc.get("_id")
        if not doc_id:
            raise StoreError("create() requires an _id")
        if "_rev" in doc:
            raise StoreError("create() must not be called with a _rev")
        if doc_id in self._docs:
            return False
        await self.put(doc)
        return True

    async def delete(self, doc_id: str, rev: str) -> None:
        existing = self._docs.get(doc_id)
        if existing is None:
            return
        if existing.get("_rev") != rev:
            raise ConflictError(f"{doc_id}: document update conflict on delete")
        del self._docs[doc_id]
        for key in [k for k in self._attachments if k[0] == doc_id]:
            del self._attachments[key]

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
        # Mirror CouchDB's index requirement so an unsortable query fails here
        # too, rather than passing in tests and 400ing in production.
        check_sortable(sort)
        matched = [copy.deepcopy(d) for d in self._docs.values() if matches(d, selector)]
        # After `matches`, so a selector that is malformed *and* unindexed is
        # reported as malformed — the more useful of the two complaints.
        check_indexable(selector, sort, use_index)
        for spec in reversed(sort or []):
            ((key, direction),) = spec.items()

            def _key(doc: Doc, field: str = key) -> tuple[int, Any]:
                return _sort_key(_dig(doc, field))

            matched.sort(key=_key, reverse=direction == "desc")
        window = matched[skip : skip + limit]
        if fields:
            # CouchDB omits a requested field the document does not have, and
            # keeps one that is present and null. The distinction matters to
            # callers that check `"key" in doc`.
            window = [{k: v for k in fields if (v := _dig(d, k)) is not MISSING} for d in window]
        return window

    async def count(self, selector: Selector) -> int:
        return sum(1 for d in self._docs.values() if matches(d, selector))

    # --- attachments --------------------------------------------------------

    async def put_attachment(self, doc_id: str, name: str, data: bytes, content_type: str) -> None:
        doc = self._docs.get(doc_id)
        if doc is None:
            raise NotFoundError(doc_id)
        self._attachments[(doc_id, name)] = (data, content_type)
        doc.setdefault("_attachments", {})[name] = {
            "content_type": content_type,
            "length": len(data),
            "stub": True,
        }
        doc["_rev"] = _next_rev(doc.get("_rev"))

    async def get_attachment(self, doc_id: str, name: str) -> bytes | None:
        entry = self._attachments.get((doc_id, name))
        return entry[0] if entry else None

    async def delete_attachment(self, doc_id: str, name: str) -> None:
        doc = self._docs.get(doc_id)
        if doc is None:
            raise NotFoundError(doc_id)
        self._attachments.pop((doc_id, name), None)
        if name in (doc.get("_attachments") or {}):
            del doc["_attachments"][name]
            doc["_rev"] = _next_rev(doc.get("_rev"))

    async def close(self) -> None:
        return None

    # --- test conveniences --------------------------------------------------

    def all_docs(self) -> list[Doc]:
        return [copy.deepcopy(d) for d in self._docs.values()]

    def docs_of_type(self, doc_type: str) -> list[Doc]:
        return [copy.deepcopy(d) for d in self._docs.values() if d.get("type") == doc_type]

    def seed(self, *docs: Doc) -> None:
        """Insert documents directly, bypassing rev checks."""
        for doc in docs:
            stored = copy.deepcopy(doc)
            stored.setdefault("_rev", _next_rev(None))
            self._docs[stored["_id"]] = stored


def _next_rev(current: str | None) -> str:
    generation = int(current.split("-", 1)[0]) + 1 if current else 1
    return f"{generation}-{uuid.uuid4().hex[:16]}"


class _Missing:
    """A field the document does not have.

    Distinct from a field present and set to null, because CouchDB treats the
    two differently and this store exists to behave like CouchDB. Returning
    ``None`` for both is what let a selector pass every test and match nothing
    in production.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING = _Missing()


def _dig(doc: Any, dotted: str) -> Any:
    """Resolve a possibly dotted Mango field path, or :data:`MISSING`."""
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def _sort_key(value: Any) -> tuple[int, Any]:
    """Mirror CouchDB's collation order so sorts match production.

    CouchDB orders null < booleans < numbers < strings < everything else, which
    means a missing field sorts *first* ascending and *last* descending. Grouping
    by type rank also avoids TypeErrors when a field is mixed-typed across docs.
    """
    # A missing field and an explicit null are distinct to a *selector* but
    # collate identically: CouchDB gives both the lowest rank.
    if value is None or value is MISSING:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float)):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    return (4, str(value))


def matches(doc: Doc, selector: Selector) -> bool:
    """Evaluate a Mango selector against a document."""
    for key, condition in selector.items():
        if key == "$and":
            if not all(matches(doc, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(matches(doc, sub) for sub in condition):
                return False
        elif key == "$nor":
            if any(matches(doc, sub) for sub in condition):
                return False
        elif key.startswith("$"):
            raise StoreError(f"MemoryStore does not implement selector operator {key!r}")
        elif not _match_field(_dig(doc, key), condition):
            return False
    return True


def _match_field(value: Any, condition: Any) -> bool:
    if not isinstance(condition, dict):
        return bool(value == condition)

    unsupported = set(condition) - _SUPPORTED_OPERATORS
    if unsupported:
        raise StoreError(
            f"MemoryStore does not implement selector operator(s): {sorted(unsupported)}"
        )

    for operator, operand in condition.items():
        # CouchDB's Mango indexes hold no entry for a document that lacks the
        # field, so *every* comparison against a missing field fails — including
        # the negative ones. `{"origin": {"$ne": "backfill"}}` does not match a
        # document with no `origin` at all, which is the opposite of what Python
        # equality suggests. Only `$exists` can see a missing field.
        if value is MISSING and operator != "$exists":
            return False

        match operator:
            case "$eq":
                if value != operand:
                    return False
            case "$ne":
                if value == operand:
                    return False
            case "$in":
                if value not in operand:
                    return False
            case "$nin":
                if value in operand:
                    return False
            case "$exists":
                if (value is not MISSING) != bool(operand):
                    return False
            case "$regex":
                if not isinstance(value, str) or not re.search(operand, value):
                    return False
            case "$lt" | "$lte" | "$gt" | "$gte":
                if not _compare(value, operator, operand):
                    return False
    return True


def _compare(value: Any, operator: str, operand: Any) -> bool:
    # CouchDB sorts null below every other type; mirror that rather than raising.
    if value is None:
        return operator in ("$lt", "$lte")
    try:
        match operator:
            case "$lt":
                return bool(value < operand)
            case "$lte":
                return bool(value <= operand)
            case "$gt":
                return bool(value > operand)
            case _:
                return bool(value >= operand)
    except TypeError:
        return False
