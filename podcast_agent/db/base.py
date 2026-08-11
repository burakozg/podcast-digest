"""Storage interface shared by the CouchDB client and the in-memory test double.

Deliberately thin: a document store with Mango queries and attachments, no ORM
(§6). Pipeline code depends on this Protocol, never on httpx or CouchDB details.
"""

from __future__ import annotations

import contextlib
import gzip
from typing import Any, Protocol, runtime_checkable

#: Attachment name for the (gzipped) transcript on an episode doc (§6).
TRANSCRIPT_ATTACHMENT = "transcript.txt.gz"

Doc = dict[str, Any]
Selector = dict[str, Any]


class StoreError(Exception):
    """Any storage failure."""


class ConflictError(StoreError):
    """CouchDB MVCC conflict (HTTP 409) — caller should re-read and re-evaluate."""


class NotFoundError(StoreError):
    """Document or attachment does not exist."""


#: Mango indexes backing every query the pipeline issues (§6).
#:
#: CouchDB will only use an index whose leading fields are exactly the sort
#: fields, so a query sorting by ``published_at`` needs an index *starting* with
#: it. Every sorted query here pins ``type`` in its selector and leads the sort
#: with it (see :func:`typed_sort`), which is why these are ordered this way.
INDEXES: tuple[dict[str, Any], ...] = (
    # Plain "everything of this type" — counts, the podcast list, the archive
    # list. Without it CouchDB scans the whole database for a handful of rows,
    # which is what the "No matching index found" warnings were reporting.
    {"name": "idx-type", "fields": ["type"]},
    # Separates archive material from routine intake, which nearly every
    # pipeline and digest query needs to do.
    {"name": "idx-type-origin", "fields": ["type", "origin"]},
    {"name": "idx-type-origin-status", "fields": ["type", "origin", "status"]},
    {"name": "idx-type-status", "fields": ["type", "status"]},
    {"name": "idx-type-published", "fields": ["type", "published_at"]},
    {"name": "idx-type-generated", "fields": ["type", "generated_at"]},
    {"name": "idx-type-status-published", "fields": ["type", "status", "published_at"]},
    {"name": "idx-type-slug-published", "fields": ["type", "podcast_slug", "published_at"]},
    {"name": "idx-type-digest-published", "fields": ["type", "digest_id", "published_at"]},
    {"name": "idx-type-ts", "fields": ["type", "ts"]},
    # Job-run history, newest first, optionally narrowed to one job.
    {"name": "idx-type-at", "fields": ["type", "at"]},
    {"name": "idx-type-job-at", "fields": ["type", "job", "at"]},
    {"name": "idx-type-transcript-at", "fields": ["type", "transcript_at"]},
)


def typed_sort(field: str, direction: str = "asc") -> list[dict[str, str]]:
    """Build a sort spec for a query whose selector pins ``type``.

    Sorting by ``published_at`` alone fails against real CouchDB with
    ``no_usable_index``, even when an index on ``(type, published_at)`` exists:
    the sort must be a prefix of the index. Leading with ``type`` — a constant
    within any such result set, so the ordering is unchanged — makes the index
    usable. CouchDB also requires every sort field to share one direction.
    """
    return [{"type": direction}, {field: direction}]


#: Selector operators CouchDB can answer from an index rather than by scanning.
#: ``$exists``, ``$ne`` and ``$regex`` are absent on purpose — none of them can
#: be served by a B-tree, so a field using one ends the usable prefix.
_INDEXABLE_OPS = frozenset({"$eq", "$gt", "$gte", "$lt", "$lte", "$in"})


def _constrains(selector: Selector, field: str) -> tuple[bool, bool]:
    """``(usable, is_equality)`` for ``field`` within ``selector``."""
    if field not in selector:
        return False, False
    value = selector[field]
    if not isinstance(value, dict):
        return True, True  # plain equality
    ops = set(value)
    # `$exists: True` alongside a range is redundant — a value that compares
    # is present — so it does not stop the range using the index. `$exists:
    # False` is the unindexable one, and it keeps its meaning here.
    if value.get("$exists") is True:
        ops.discard("$exists")
    if not ops or not ops <= _INDEXABLE_OPS:
        return False, False
    return True, ops == {"$eq"}


def _prefix_score(selector: Selector, fields: list[str]) -> int:
    """How many leading index fields this selector actually narrows.

    Equality on a field lets the scan continue into the next one; a range or
    ``$in`` narrows that field but nothing after it, so it ends the prefix.
    """
    score = 0
    for field in fields:
        usable, equality = _constrains(selector, field)
        if not usable:
            break
        score += 1
        if not equality:
            break
    return score


def resolve_index(selector: Selector, sort: list[dict[str, str]] | None = None) -> str | None:
    """Name the declared index that best serves this query, or None.

    Left to itself CouchDB picks an index by heuristic, and its heuristic does
    not know which fields the selector pins by equality. In practice it kept
    choosing the broadest index and filtering the rest in memory — six distinct
    query shapes were reporting "documents examined is high in proportion to the
    number of results returned" in one night's log. Naming the index makes the
    choice deterministic and reviewable, and makes a *missing* index a test
    failure instead of a silent full scan.
    """
    if sort:
        # CouchDB can only sort from an index whose fields *begin* with the sort
        # fields, so nothing else may precede them. Among those, the shortest
        # wins: trailing fields the sort does not name add width, not selectivity.
        wanted = [next(iter(spec)) for spec in sort]
        matching = [i for i in INDEXES if i["fields"][: len(wanted)] == wanted]
        if not matching:
            return None
        return str(min(matching, key=lambda i: len(i["fields"]))["name"])

    # Unsorted: the most leading fields narrowed, then the narrowest index.
    ranked = [
        (-score, len(index["fields"]), str(index["name"]))
        for index in INDEXES
        if (score := _prefix_score(selector, index["fields"]))
    ]
    return min(ranked)[2] if ranked else None


def check_indexable(
    selector: Selector,
    sort: list[dict[str, str]] | None = None,
    use_index: str | None = None,
) -> None:
    """Raise unless some declared index can serve ``selector``.

    Enforced by the in-memory store, which is what the tests run against. A
    selector no index touches is a full database scan in production and a fast
    dict comprehension in tests — the one failure mode that gets worse the more
    real data you have and never shows up in CI. Failing here makes adding a
    query without adding its index a test failure.
    """
    if use_index or resolve_index(selector, sort):
        return
    raise StoreError(
        f"no declared Mango index serves {sorted(selector)}; CouchDB would scan "
        "the whole database (see db.base.INDEXES). Every query must pin `type`."
    )


def check_sortable(sort: list[dict[str, str]] | None) -> None:
    """Raise unless ``sort`` is one CouchDB would actually accept.

    Enforced by the in-memory store as well as documented here, so a query that
    would fail in production cannot quietly pass in tests.
    """
    if not sort:
        return
    directions = {d for spec in sort for d in spec.values()}
    if len(directions) > 1:
        raise StoreError(
            f"CouchDB requires all sort fields to share a direction, got {sorted(directions)}"
        )
    fields = [next(iter(spec)) for spec in sort]
    if not any(index["fields"][: len(fields)] == fields for index in INDEXES):
        raise StoreError(
            f"no_usable_index: no declared Mango index starts with {fields}; "
            "CouchDB would reject this query (see db.base.INDEXES / typed_sort)"
        )


@runtime_checkable
class Store(Protocol):
    """Document store operations used by the pipeline."""

    async def ensure_setup(self) -> None:
        """Create the database and Mango indexes if absent. Idempotent."""

    async def ping(self) -> bool:
        """True when the store is reachable and the database exists."""

    async def get(self, doc_id: str) -> Doc | None: ...

    async def put(self, doc: Doc) -> Doc:
        """Insert or update. ``doc`` must carry ``_id``, and ``_rev`` when updating.

        Raises ConflictError on a stale ``_rev``.
        """

    async def create(self, doc: Doc) -> bool:
        """Insert only if absent. Returns False when the doc already exists.

        This is the idempotency primitive for ingestion — two concurrent runs
        seeing the same new episode cannot both create it.
        """

    async def delete(self, doc_id: str, rev: str) -> None: ...

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
        """Query. ``use_index`` overrides the index :func:`resolve_index` picks.

        Callers should not normally pass it — the resolver reads the selector.
        It exists for the rare query whose shape misleads the resolver.
        """

    async def count(self, selector: Selector) -> int: ...

    async def put_attachment(
        self, doc_id: str, name: str, data: bytes, content_type: str
    ) -> None: ...

    async def get_attachment(self, doc_id: str, name: str) -> bytes | None: ...

    async def delete_attachment(self, doc_id: str, name: str) -> None: ...

    async def close(self) -> None: ...


async def update_doc(
    store: Store,
    doc_id: str,
    mutator: Any,
    *,
    max_retries: int = 5,
) -> Doc:
    """Read-modify-write with MVCC conflict retry (§6).

    ``mutator`` is called with the current document and mutates it in place (or
    returns a replacement). On a 409 the document is re-read and the mutator runs
    again against fresh state — never a force-overwrite.
    """
    last_error: Exception | None = None
    for _ in range(max_retries):
        doc = await store.get(doc_id)
        if doc is None:
            raise NotFoundError(doc_id)
        result = mutator(doc)
        candidate: Doc = result if isinstance(result, dict) else doc
        try:
            return await store.put(candidate)
        except ConflictError as exc:
            last_error = exc
            continue
    raise ConflictError(f"{doc_id}: still conflicting after {max_retries} attempts") from last_error


# --- Transcript helpers -----------------------------------------------------
# Transcripts are gzipped attachments so the JSON doc stays small and Mango
# indexes stay fast (§6).


async def save_transcript(store: Store, episode_id: str, text: str) -> int:
    """Store transcript text gzipped. Returns the compressed byte size."""
    blob = gzip.compress(text.encode("utf-8"), compresslevel=6)
    await store.put_attachment(episode_id, TRANSCRIPT_ATTACHMENT, blob, "application/gzip")
    return len(blob)


async def load_transcript(store: Store, episode_id: str) -> str | None:
    blob = await store.get_attachment(episode_id, TRANSCRIPT_ATTACHMENT)
    if blob is None:
        return None
    try:
        return gzip.decompress(blob).decode("utf-8", errors="replace")
    except (OSError, EOFError) as exc:  # corrupt attachment must not kill the run
        raise StoreError(f"{episode_id}: transcript attachment is unreadable ({exc})") from exc


async def drop_transcript(store: Store, episode_id: str) -> None:
    with contextlib.suppress(NotFoundError):
        await store.delete_attachment(episode_id, TRANSCRIPT_ATTACHMENT)
