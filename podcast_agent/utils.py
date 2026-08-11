"""Small shared helpers: time, IDs, token estimation."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

#: Rough characters-per-token ratio for English prose. Used only to decide
#: whether Tier-1 needs map-reduce, so an approximation is adequate and avoids a
#: tokenizer dependency that would differ per model anyway.
CHARS_PER_TOKEN = 4


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    """Serialise to UTC ISO-8601. All storage is UTC (§10.3)."""
    return dt.astimezone(UTC).isoformat()


def iso_now() -> str:
    return iso(utcnow())


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def to_local(dt: datetime, tz: ZoneInfo) -> datetime:
    """Render-time conversion; storage stays UTC."""
    return dt.astimezone(tz)


def episode_doc_id(podcast_slug: str, guid: str) -> str:
    """Stable episode ID: sha256(slug + guid) (§4 stage 1).

    Must never change for a given (show, guid) pair — it is the idempotency key
    for the entire pipeline.
    """
    digest = hashlib.sha256(f"{podcast_slug}\x00{guid}".encode()).hexdigest()
    return f"episode:{digest}"


def podcast_doc_id(slug: str) -> str:
    return f"podcast:{slug}"


def digest_doc_id(period_key: str) -> str:
    return f"digest:{period_key}"


def llm_call_doc_id() -> str:
    return f"llmcall:{uuid.uuid4()}"


def asr_run_doc_id() -> str:
    return f"asrrun:{uuid.uuid4()}"


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def run_doc_id(run_id: str) -> str:
    return f"run:{run_id}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


#: Beyond this, an episode is old enough that summarising it as current would
#: mislead — the Tier-1 prompt is told to frame it historically instead (A2).
ARCHIVE_AGE_DAYS = 90


def episode_age_days(published_at: str | None, *, now: datetime | None = None) -> int | None:
    """Whole days between publication and ``now``. None when the date is unknown."""
    published = parse_iso(published_at)
    if published is None:
        return None
    delta = (now or utcnow()) - published
    return max(0, delta.days)


def describe_age(age_days: int | None) -> str:
    """Human phrasing of an episode's age for the Tier-1 prompt (A2).

    Old episodes are labelled explicitly so the model frames dated claims as
    historical rather than current — the failure mode when summarising an
    archive is a confident summary of a world that has moved on.
    """
    if age_days is None:
        return ""
    # Past the archive threshold the label is explicit, because that is the cue
    # the prompt keys off when deciding to frame claims historically.
    suffix = " — an archive episode" if age_days >= ARCHIVE_AGE_DAYS else ""
    if age_days <= 1:
        return "published in the last day"
    if age_days < 14:
        return f"published {age_days} days ago"
    if age_days < 60:
        return f"published about {age_days // 7} weeks ago"
    if age_days < 365:
        return f"published about {age_days // 30} months ago{suffix}"
    years = age_days / 365
    plural = "s" if years >= 1.5 else ""
    return f"published about {years:.0f} year{plural} ago{suffix}"


def slug_month_label(month: str) -> str:
    """'2026-07' -> 'July 2026', for archive file headings."""
    try:
        year, mon = (int(part) for part in month.split("-"))
        return f"{MONTH_NAMES[mon - 1]} {year}"
    except (ValueError, IndexError):
        return month


MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def format_duration(seconds: int | None) -> str:
    """Human-readable duration for the digest ('62 min')."""
    if not seconds or seconds <= 0:
        return "unknown length"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, rem = divmod(minutes, 60)
    return f"{hours} h {rem:02d} min" if rem else f"{hours} h"
