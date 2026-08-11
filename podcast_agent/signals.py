"""Reader marks, written into the vault for something else to read.

Stars and wrong-call flags live on episode documents, which is fine for the
console and useless to anything outside it. This mirrors them into the vault so
a model working over it can see which episodes were actually worth the reader's
time, and which the pipeline got wrong.

**One file per week, holding only what was marked since the last run.** Marks
accumulate slowly — a handful a week — so a full snapshot rewritten weekly would
be fifty near-identical files, and a reader could not tell which lines were new.
A cursor is kept rather than trusting the calendar: a run that is late, or that
did not happen at all, still picks up everything since the one before it, where
a strict "marks in ISO week N" query would drop the gap silently.

They cannot go into the digests. Those are written once and never touched, and
the marks arrive afterwards — you read the digest, *then* star.

The hard part is not the positives. It is saying clearly that an unmarked episode
is *not* a negative: most episodes are never starred, including good ones, and a
reader that infers dislike from silence would learn the opposite of the truth.
The file says so in its own text, because whatever reads it next will not have
read this docstring.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Doc, Store, typed_sort
from .logging_setup import get_logger
from .sanitize import md_escape_inline, safe_url
from .state import EpisodeStatus
from .utils import format_duration, iso_now, utcnow

log = get_logger(__name__)

#: Where the cursor lives. One document, like the other control state.
CURSOR_DOC_ID = "control:signals"

#: Directory under the digest folder, so a week's marks sit beside the weeks.
OUTPUT_DIR = "signals"

#: Statuses meaning the episode was actually put in front of the reader. The
#: same set the precision report uses: a dropped episode was never declined,
#: it was never offered.
SURFACED = frozenset({EpisodeStatus.READY_FOR_DIGEST.value, EpisodeStatus.PUBLISHED.value})

#: Marks carried in one file. Far above a week's worth; a cap rather than an
#: expectation, so a first run over a long history cannot write something
#: nothing will read.
MAX_PER_SECTION = 200


def marked_at(episode: Doc) -> str:
    """When the reader last touched this episode, by either kind of mark."""
    feedback = episode.get("feedback") or {}
    return max(str(episode.get("starred_at") or ""), str(feedback.get("at") or ""))


async def cursor(store: Store) -> str | None:
    """When marks were last exported, or None if never."""
    doc = await store.get(CURSOR_DOC_ID)
    return str(doc["exported_at"]) if doc and doc.get("exported_at") else None


async def advance_cursor(store: Store, exported_at: str) -> None:
    existing = await store.get(CURSOR_DOC_ID)
    doc: Doc = {
        "_id": CURSOR_DOC_ID,
        "type": "control",
        "key": "signals",
        "exported_at": exported_at,
    }
    if existing:
        doc["_rev"] = existing["_rev"]
    await store.put(doc)


async def collect(store: Store, *, since: str | None) -> dict[str, list[Doc]]:
    """Episodes marked after ``since``, most recently marked first.

    ``since`` of None means everything, so the first run carries whatever
    backlog exists and every run after it carries only its own period.
    """
    starred = await store.find(
        {"type": "episode", "starred": True},
        sort=typed_sort("published_at", "desc"),
        limit=MAX_PER_SECTION * 4,
    )
    flagged = await store.find(
        {"type": "episode", "feedback": {"$exists": True}},
        sort=typed_sort("published_at", "desc"),
        limit=MAX_PER_SECTION * 4,
    )

    def fresh(docs: list[Doc]) -> list[Doc]:
        kept = [d for d in docs if not since or marked_at(d) > since]
        kept.sort(key=marked_at, reverse=True)
        return kept[:MAX_PER_SECTION]

    def verdict(doc: Doc) -> str:
        return str((doc.get("feedback") or {}).get("verdict") or "")

    return {
        # A star on something never surfaced says nothing about a ranking it
        # never got. A *flag* on one is the opposite: "should have ranked
        # higher" matters most where the episode was never shown at all.
        "starred": fresh([d for d in starred if d.get("status") in SURFACED]),
        "under": fresh([d for d in flagged if verdict(d) == "under"]),
        "over": fresh([d for d in flagged if verdict(d) == "over"]),
    }


async def surfaced_total(store: Store) -> int:
    """How many episodes were offered at all — the denominator for the marks."""
    total = 0
    for status in SURFACED:
        total += await store.count({"type": "episode", "status": status})
    return total


def _entry(episode: Doc) -> list[str]:
    tier1 = episode.get("tier1") or {}
    show = md_escape_inline(
        episode.get("podcast_name") or episode.get("podcast_slug") or "", max_chars=80
    )
    title = md_escape_inline(episode.get("title") or "(untitled)", max_chars=180)
    link = safe_url(episode.get("link"))
    head = f"[{show} — {title}]({link})" if link else f"{show} — {title}"

    facts = [(episode.get("published_at") or "")[:10]]
    if episode.get("duration_s"):
        facts.append(format_duration(episode.get("duration_s")))
    if tier1.get("relevance_score") is not None:
        facts.append(f"scored {tier1['relevance_score']}/10")
    if interests := [str(i) for i in (tier1.get("matched_interests") or []) if str(i).strip()]:
        facts.append(", ".join(interests[:6]))

    out = [f"### {head}", "", f"*{' · '.join(facts)}*", ""]
    if why := md_escape_inline(tier1.get("why_it_matters") or "", max_chars=600):
        out += [why, ""]

    feedback = episode.get("feedback") or {}
    if note := md_escape_inline(feedback.get("note") or "", max_chars=400):
        out += [f"**Reader's note:** {note}", ""]
    # What the pipeline believed when the reader disagreed. The disagreement is
    # the signal; without both numbers it is only half of one.
    judged = feedback.get("judged") or {}
    if judged.get("relevance_score") is not None:
        out += [f"*The pipeline had scored this {judged['relevance_score']}/10.*", ""]
    return out


def render(
    marks: dict[str, list[Doc]],
    *,
    since: str | None,
    until: str,
    surfaced: int,
    settings: Settings,
) -> str:
    """Markdown for the vault, addressed to whatever reads it next."""
    generated = utcnow().astimezone(settings.tz).isoformat(timespec="seconds")
    starred, under, over = marks["starred"], marks["under"], marks["over"]
    covers = f"marked since {since[:10]}" if since else "every mark made so far"

    out = [
        "---",
        "type: reading-signals",
        f"generated: {generated}",
        f"covers_from: {since or 'the beginning'}",
        f"covers_to: {until}",
        f"surfaced_total: {surfaced}",
        f"starred: {len(starred)}",
        f"ranked_too_low: {len(under)}",
        f"not_worth_it: {len(over)}",
        "tags: [reading-signals, cybersecurity]",
        "---",
        "",
        "# Reading signals",
        "",
        f"Episodes marked by hand while reading — {covers}. Preferences read from "
        "what happened, rather than guessed from a profile.",
        "",
        "## How to read this",
        "",
        "- **This file holds one period's new marks, not the whole picture.** "
        f"Earlier periods sit beside it in `{OUTPUT_DIR}/`; read them together for "
        "the full set.",
        f"- The corpus has offered {surfaced} episodes in all. These are the ones "
        "marked in this period.",
        "- **An unlisted episode is not a rejection.** Most good episodes are never "
        "starred; starring is optional and sporadic. Silence here carries no "
        "information, and treating it as dislike inverts the signal.",
        "- **The two flagged sections are the deliberate corrections** — the reader "
        "saying the ranking was wrong, in one direction or the other. They are the "
        "strongest evidence in this file, and the rarest.",
        "",
    ]

    for heading, note, episodes in (
        ("Worth the reader's time", "Starred while reading.", starred),
        (
            "Ranked too low",
            "The reader said these deserved more prominence than they were given — "
            "false negatives, the expensive kind, because nobody sees what they "
            "were not shown.",
            under,
        ),
        ("Not worth it", "Surfaced prominently and the reader disagreed.", over),
    ):
        out += [f"## {heading} ({len(episodes)})", "", note, ""]
        if not episodes:
            out += ["*Nothing marked this way in this period.*", ""]
            continue
        for episode in episodes:
            out += _entry(episode)

    out += [
        "---",
        "*Written once per period from the marks held in the database. Earlier "
        "periods are never rewritten, so a mark stays reported where it was made.*",
    ]
    return "\n".join(out) + "\n"


def period_name(settings: Settings, moment: datetime | None = None) -> str:
    """ISO week of the run, named the way the digests are."""
    local = (moment or utcnow()).astimezone(settings.tz)
    year, week, _ = local.isocalendar()
    return f"{year}-W{week:02d}"


async def export_new_marks(store: Store, settings: Settings) -> dict[str, Any]:
    """Write this period's marks, if there are any, and advance the cursor.

    Silent when nothing was marked: a file every week saying nothing teaches the
    reader that the directory is noise, and hands a model working over the vault
    a page with no content and a confident heading.
    """
    since = await cursor(store)
    until = iso_now()
    marks = await collect(store, since=since)
    total = sum(len(v) for v in marks.values())

    if not total:
        # The cursor still moves. Nothing was marked, and a later run should not
        # re-ask the same empty question over an ever-widening window.
        await advance_cursor(store, until)
        log.info("signals.nothing_new", since=since)
        return {"written": None, "marks": 0, "since": since, "until": until}

    # Imported here: digest.generate pulls in Jinja2 and the whole template
    # stack, which this module otherwise has no use for.
    from .digest.generate import _atomic_write

    body = render(
        marks, since=since, until=until, surfaced=await surfaced_total(store), settings=settings
    )
    path = _atomic_write(
        settings.output.digest_dir, Path(OUTPUT_DIR) / f"{period_name(settings)}.md", body
    )
    await advance_cursor(store, until)

    result: dict[str, Any] = {
        "written": str(path.relative_to(settings.output.digest_dir)),
        "marks": total,
        "starred": len(marks["starred"]),
        "ranked_too_low": len(marks["under"]),
        "not_worth_it": len(marks["over"]),
        "since": since,
        "until": until,
    }
    log.info("signals.written", **result)
    return result
