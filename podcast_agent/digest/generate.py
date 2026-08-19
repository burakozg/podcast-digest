"""Stage 5 — digest generation (§4, §5).

Ordering guarantees (§10.3): the full document is built in memory, written
atomically, and only then are episodes marked PUBLISHED. If the process dies
between the write and the marking, the next run finds the digest doc, sees the
file already exists, and completes the marking (reconciliation) instead of
producing a duplicate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..backfill import ROUTINE_ONLY
from ..config import Settings
from ..db import Doc, Store, typed_sort, update_doc
from ..episodes import transition
from ..llm.base import StructuredLLM
from ..logging_setup import get_logger
from ..models import WeeklySynthesis
from ..sanitize import (
    md_escape_inline,
    md_escape_table_cell,
    md_to_speech_text,
    safe_url,
    sanitize_md_block,
    slugify,
)
from ..state import AUDIT_ONLY_STATUSES, DIGESTABLE_STATUSES, EpisodeStatus
from ..utils import digest_doc_id, format_duration, iso, iso_now, parse_iso, utcnow
from .synthesis import WeeklySynthesizer, as_view, previous_theme_titles

log = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

#: Human labels for the provenance of a summary (§5 requires honest labelling).
BASIS_LABELS = {
    "transcript": "local transcription",
    "published_transcript": "published transcript",
    "description_only": "description only (no transcript available)",
}

OUTCOME_LABELS = {
    EpisodeStatus.SCORED_LOW.value: "summarized, scored below threshold",
    EpisodeStatus.DROPPED.value: "dropped at triage",
    EpisodeStatus.TRANSCRIPT_FAILED.value: "no transcript could be obtained",
    EpisodeStatus.ERROR.value: "processing error",
}


@dataclass(slots=True)
class DigestResult:
    digest_id: str
    period_key: str
    file_path: Path | None
    episode_ids: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    #: True when an existing digest doc was reconciled rather than newly written.
    reconciled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "digest_id": self.digest_id,
            "period_key": self.period_key,
            "file_path": str(self.file_path) if self.file_path else None,
            "episodes": len(self.episode_ids),
            "stats": self.stats,
            "dry_run": self.dry_run,
            "reconciled": self.reconciled,
        }


def _build_env() -> Environment:
    """Jinja environment shared by the weekly and archive digests.

    autoescape off: the output is Markdown, and every interpolated value is
    sanitised for a Markdown context before it reaches the template.
    """
    env = Environment(  # noqa: S701
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    # For the spoken script, which needs the same values with their Markdown
    # taken back off — a synthesiser reads `**bold**` aloud as punctuation.
    env.filters["speech"] = md_to_speech_text
    return env


def _interest_labels(settings: Settings, keys: list[str] | None) -> list[str]:
    by_key = {i.key: i.label for i in settings.interest_profile}
    return [md_escape_inline(by_key[k], max_chars=60) for k in (keys or []) if k in by_key]


def _episode_views(
    settings: Settings,
    episode: Doc,
    basis_labels: dict[str, str],
    *,
    audit: bool = False,
) -> dict[str, Any]:
    """Build a template-ready view of one episode.

    Every field crossing into a template is sanitised here: titles come from
    feeds and summaries come from an LLM fed by feeds (§10.2).
    """
    tier1 = episode.get("tier1") or {}
    tier0 = episode.get("tier0") or {}
    published = (episode.get("published_at") or "")[:10] or "unknown date"

    if audit:
        if tier1.get("relevance_score") is not None:
            score_display = f"{int(tier1['relevance_score'])}/10"
        elif tier0.get("relevance_guess") is not None:
            score_display = f"~{int(tier0['relevance_guess'])}/10"
        else:
            score_display = "—"
        return {
            "podcast_name": md_escape_table_cell(
                episode.get("podcast_name") or episode["podcast_slug"], max_chars=60
            ),
            "title": md_escape_table_cell(episode.get("title") or "(untitled)", max_chars=110),
            "published_date": published,
            "score_display": score_display,
            "outcome": OUTCOME_LABELS.get(str(episode.get("status")), str(episode.get("status"))),
        }

    if episode.get("status") == EpisodeStatus.DIGEST_DIRECT.value:
        return {
            "episode_id": episode["_id"],
            "podcast_name": md_escape_inline(
                episode.get("podcast_name") or episode["podcast_slug"], max_chars=80
            ),
            "title": md_escape_inline(episode.get("title") or "(untitled)", max_chars=200),
            "published_date": published,
            "score": int(tier0.get("relevance_guess") or 0),
            "link": safe_url(episode.get("link")),
            "reasoning": md_escape_inline(tier0.get("reasoning") or "", max_chars=240),
        }

    return summary_view(settings, episode, basis_labels)


def summary_view(settings: Settings, episode: Doc, basis_labels: dict[str, str]) -> dict[str, Any]:
    """The template view of a *summarised* episode.

    Split out of :func:`_episode_views` so the ad-hoc Markdown export can reach
    it directly. That path has already established the episode has a summary,
    and must not be routed through the status branching above — a DIGEST_DIRECT
    episode that was later summarised on request would otherwise come back in
    the one-liner shape, with no ``summary_md`` at all.
    """
    tier1 = episode.get("tier1") or {}
    published = (episode.get("published_at") or "")[:10] or "unknown date"
    basis = str(tier1.get("summary_basis") or "description_only")
    return {
        "episode_id": episode["_id"],
        "podcast_name": md_escape_inline(
            episode.get("podcast_name") or episode["podcast_slug"], max_chars=80
        ),
        "podcast_slug": episode["podcast_slug"],
        "title": md_escape_inline(episode.get("title") or "(untitled)", max_chars=200),
        "score": int(tier1.get("relevance_score") or 0),
        "published_date": published,
        "duration": format_duration(episode.get("duration_s")),
        "basis": basis,
        "basis_label": basis_labels.get(basis, basis),
        "link": safe_url(episode.get("link")),
        "why_it_matters": md_escape_inline(tier1.get("why_it_matters") or "", max_chars=1000),
        "summary_md": sanitize_md_block(tier1.get("summary_md") or ""),
        "key_takeaways": [str(b) for b in (tier1.get("key_takeaways") or [])],
        "entities": [str(e) for e in (tier1.get("entities") or [])][:20],
        # Wikilinks to the per-entity notes, so the vault's graph has edges from
        # both ends: the entity note lists its episodes, and the episode note
        # points back. Plain text in the weekly digest, which is read top to
        # bottom rather than navigated.
        "entity_links": [
            f"[[{slugify(str(e))}|{md_escape_inline(str(e), max_chars=80)}]]"
            for e in (tier1.get("entities") or [])[:20]
            if slugify(str(e))
        ],
        "interests": _interest_labels(settings, tier1.get("matched_interests")),
        "interest_keys": [str(k) for k in (tier1.get("matched_interests") or [])],
        "listen_anyway": bool(tier1.get("listen_anyway")),
    }


class DigestGenerator:
    def __init__(self, settings: Settings, store: Store, llm: StructuredLLM | None = None) -> None:
        self._settings = settings
        self._store = store
        self._env = _build_env()
        # Optional on purpose: the digest is a database read and a render, and
        # it stays that way when no model is wired in. Every caller that has one
        # gets the opening section; the tests that do not, do not.
        self._synthesizer = (
            WeeklySynthesizer(settings, llm)
            if llm is not None and settings.pipeline.weekly_synthesis
            else None
        )

    async def generate(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        dry_run: bool = False,
    ) -> DigestResult:
        now = utcnow()
        period_to = until or now
        period_from = since or await self._default_period_start(period_to)
        period_key = _iso_week_key(period_to.astimezone(self._settings.tz))
        digest_id = digest_doc_id(period_key)

        existing = await self._store.get(digest_id)
        if existing and not dry_run:
            # A digest doc for this period exists: finish any incomplete marking
            # rather than writing a second file for the same week.
            reconciled = await self._reconcile(existing)
            if reconciled is not None:
                return reconciled

        candidates = await self._collect(period_from, period_to)
        buckets = self._bucket(candidates, period_from)
        stats = await self._stats(candidates)
        synthesis = await self._synthesize(candidates, period_from, period_to)

        context = self._build_context(
            period_key=period_key,
            period_from=period_from,
            period_to=period_to,
            generated=now,
            buckets=buckets,
            stats=stats,
            synthesis=synthesis,
        )
        rendered = self._env.get_template("digest.md.j2").render(**context)

        episode_ids = [e["_id"] for e in candidates]
        if dry_run:
            log.info(
                "digest.dry_run",
                period_key=period_key,
                episodes=len(episode_ids),
                bytes=len(rendered.encode("utf-8")),
                **{k: v for k, v in stats.items() if k != "total_cost_usd"},
            )
            return DigestResult(
                digest_id=digest_id,
                period_key=period_key,
                file_path=None,
                episode_ids=episode_ids,
                stats=stats,
                dry_run=True,
            )

        relative = Path(str(period_to.astimezone(self._settings.tz).year)) / (
            f"podcast-digest-{period_key}.md"
        )
        written = _atomic_write(self._settings.output.digest_dir, relative, rendered)

        note_paths: list[str] = []
        if self._settings.output.episode_notes:
            note_paths = self._write_episode_notes(buckets, period_key)

        relative_path = str(written.relative_to(self._settings.output.digest_dir))
        run = {
            "file_path": relative_path,
            "period": {"from": iso(period_from), "to": iso(period_to)},
            "episode_ids": episode_ids,
            "stats": stats,
            "episode_notes": note_paths,
            "generated_at": iso_now(),
        }
        # A second generation in the same ISO week writes a second file — the
        # generator never overwrites — but the document is keyed by the week, so
        # replacing it wholesale orphaned the earlier file: still in the vault,
        # referenced by nothing, invisible to the console. Runs accumulate; the
        # top level keeps describing the most recent, which is what every
        # existing reader expects.
        previous = list((existing or {}).get("runs") or [])
        runs = [*previous, run]

        # File is on disk before any episode is marked — the ordering that makes
        # a crash here recoverable.
        await self._store.put(
            {
                "_id": digest_id,
                "type": "digest",
                "period": run["period"],
                "file_path": relative_path,
                "episode_ids": episode_ids,
                "stats": stats,
                "episode_notes": note_paths,
                # Stored so next week's "what's new" has something to compare
                # against. Themes are the only part worth carrying forward: the
                # rest is already in the file on disk.
                "synthesis": as_view(synthesis),
                "runs": runs,
                "marking_complete": False,
                "generated_at": run["generated_at"],
                **({"_rev": existing["_rev"]} if existing else {}),
            }
        )
        marked = await self._mark_published(episode_ids, digest_id)
        await self._set_marking_complete(digest_id)

        log.info(
            "digest.generated",
            period_key=period_key,
            file=str(written),
            episodes=len(episode_ids),
            marked_published=marked,
            episode_notes=len(note_paths),
            **stats,
        )
        return DigestResult(
            digest_id=digest_id,
            period_key=period_key,
            file_path=written,
            episode_ids=episode_ids,
            stats=stats,
        )

    # --- period / selection -------------------------------------------------

    async def _default_period_start(self, period_to: datetime) -> datetime:
        """Start where the last digest ended, so nothing falls between digests."""
        previous = await self._store.find(
            {"type": "digest"}, sort=typed_sort("generated_at", "desc"), limit=1
        )
        if previous:
            last_to = parse_iso((previous[0].get("period") or {}).get("to"))
            if last_to:
                return last_to
        # No history: one week back, plus the backfill window so a fresh install's
        # first digest includes what ingestion just picked up.
        lookback = max(7, self._settings.pipeline.initial_lookback_days)
        return period_to - timedelta(days=lookback)

    def _floor(self, period_from: datetime, period_to: datetime) -> datetime:
        """Earliest publication date this digest will reach back to.

        The window's own start is the wrong floor, and using it lost episodes.
        A digest leaves anything still mid-pipeline for "the next digest", but the
        next digest's window *starts where this one ended* — so an episode still
        awaiting a transcript on Friday was already behind the floor on Saturday
        and could never appear in any later window. Nothing ever claimed it and
        nothing ever reported it missing.

        Claim-once is enforced by ``digest_id``, not by this bound, so reaching
        further back cannot duplicate anything. The floor exists only to stop a
        fresh install's entire initial ingest landing in its first digest.
        """
        catch_up = period_to - timedelta(days=self._settings.pipeline.digest_catch_up_days)
        return min(period_from, catch_up)

    async def _synthesize(
        self, candidates: list[Doc], period_from: datetime, period_to: datetime
    ) -> WeeklySynthesis | None:
        """The opening section, or None. Never raises — see synthesis.py."""
        if self._synthesizer is None:
            return None
        previous = await self._store.find(
            {"type": "digest"}, sort=typed_sort("generated_at", "desc"), limit=1
        )
        return await self._synthesizer.build(
            candidates,
            period_from=iso(period_from),
            period_to=iso(period_to),
            previous_themes=previous_theme_titles(previous[0] if previous else None),
        )

    async def _collect(self, period_from: datetime, period_to: datetime) -> list[Doc]:
        """Unclaimed episodes up to ``period_to``, newest first.

        Selection keys off ``digest_id`` rather than status so that audit-only
        rows (dropped, low-scored) are also claimed exactly once and cannot be
        re-listed in a later digest.
        """
        docs = await self._store.find(
            {
                "type": "episode",
                "digest_id": None,
                # Archive material has its own per-show monthly files; letting a
                # 2019 episode into this week's digest would defeat the point.
                **ROUTINE_ONLY,
                "published_at": {
                    "$gte": iso(self._floor(period_from, period_to)),
                    "$lt": iso(period_to),
                },
            },
            sort=typed_sort("published_at", "desc"),
            limit=1000,
        )
        # Episodes still mid-pipeline are left for the next digest, which can now
        # actually reach them.
        pending = {
            EpisodeStatus.NEW.value,
            EpisodeStatus.TRIAGED.value,
            EpisodeStatus.AWAITING_TRANSCRIPT.value,
            EpisodeStatus.TRANSCRIBED.value,
            EpisodeStatus.SUMMARIZED.value,
        }
        return [d for d in docs if d.get("status") not in pending]

    def _bucket(
        self, episodes: list[Doc], period_from: datetime | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        pipeline = self._settings.pipeline
        top: list[dict[str, Any]] = []
        also: list[dict[str, Any]] = []
        maybe: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        # Older than this week but only now finished, so it is being reported for
        # the first time. Marked in the digest rather than presented as news —
        # a three-week-old episode arriving silently reads as a dating error.
        boundary = iso(period_from) if period_from else None

        for episode in episodes:
            status = episode.get("status")
            carried = bool(boundary and (episode.get("published_at") or "") < boundary)
            if status == EpisodeStatus.READY_FOR_DIGEST.value:
                view = self._summary_view(episode)
                view["carried_over"] = carried
                (top if view["score"] >= pipeline.top_pick_threshold else also).append(view)
            elif status == EpisodeStatus.DIGEST_DIRECT.value:
                view = self._one_liner_view(episode)
                view["carried_over"] = carried
                maybe.append(view)
            elif status in AUDIT_ONLY_STATUSES:
                rest.append(self._audit_view(episode))
            else:
                # PUBLISHED with digest_id unset shouldn't happen, but an
                # unexpected status must still appear somewhere auditable.
                rest.append(self._audit_view(episode))

        top.sort(key=lambda v: (-v["score"], v["podcast_name"]))
        also.sort(key=lambda v: (-v["score"], v["podcast_name"]))
        maybe.sort(key=lambda v: (-v["score"], v["podcast_name"]))
        rest.sort(key=lambda v: (v["podcast_name"], v["title"]))
        return {
            "top_picks": top,
            "also_relevant": also,
            "maybe_interesting": maybe,
            "everything_else": rest,
        }

    # --- view builders ------------------------------------------------------
    # Every field crossing into a template is sanitised here: episode titles come
    # from feeds and summaries come from an LLM fed by feeds (§10.2).

    def _interest_labels(self, keys: list[str] | None) -> list[str]:
        return _interest_labels(self._settings, keys)

    def _summary_view(self, episode: Doc) -> dict[str, Any]:
        return _episode_views(self._settings, episode, BASIS_LABELS)

    def _one_liner_view(self, episode: Doc) -> dict[str, Any]:
        return _episode_views(self._settings, episode, BASIS_LABELS)

    def _audit_view(self, episode: Doc) -> dict[str, Any]:
        return _episode_views(self._settings, episode, BASIS_LABELS, audit=True)

    async def _stats(self, episodes: list[Doc]) -> dict[str, Any]:
        summarized = sum(
            1 for e in episodes if e.get("status") == EpisodeStatus.READY_FOR_DIGEST.value
        )
        asr_runs = sum(1 for e in episodes if e.get("transcript_source") == "asr")
        cost = 0.0
        for episode in episodes:
            for tier in ("tier0", "tier1"):
                block = episode.get(tier) or {}
                cost += float(block.get("cost_usd") or 0.0)
        return {
            "scanned": len(episodes),
            "summarized": summarized,
            "asr_runs": asr_runs,
            "total_cost_usd": round(cost, 6),
        }

    def _build_context(
        self,
        *,
        period_key: str,
        period_from: datetime,
        period_to: datetime,
        generated: datetime,
        buckets: dict[str, list[dict[str, Any]]],
        stats: dict[str, Any],
        regenerated_from: str | None = None,
        synthesis: WeeklySynthesis | None = None,
    ) -> dict[str, Any]:
        tz = self._settings.tz
        local_to = period_to.astimezone(tz)
        return {
            "synthesis": as_view(synthesis),
            "week": period_key,
            "week_number": int(period_key.split("-W")[1]),
            "year": local_to.year,
            # Rendered in local time; storage stays UTC (§10.3).
            "generated_local": generated.astimezone(tz).isoformat(timespec="seconds"),
            "period_from_local": period_from.astimezone(tz).isoformat(timespec="seconds"),
            "period_to_local": local_to.isoformat(timespec="seconds"),
            "stats": stats,
            "regenerated_from": regenerated_from,
            "carried_over_count": sum(
                1 for views in buckets.values() for v in views if v.get("carried_over")
            ),
            **buckets,
        }

    # --- episode notes ------------------------------------------------------

    def _write_episode_notes(
        self, buckets: dict[str, list[dict[str, Any]]], period_key: str
    ) -> list[str]:
        template = self._env.get_template("episode.md.j2")
        written: list[str] = []
        for view in [*buckets["top_picks"], *buckets["also_relevant"]]:
            relative = (
                Path("episodes")
                / slugify(view["podcast_slug"])
                / f"{view['published_date']}-{slugify(view['title'])}.md"
            )
            rendered = template.render(e=view, week=period_key)
            path = _atomic_write(self._settings.output.digest_dir, relative, rendered)
            written.append(str(path.relative_to(self._settings.output.digest_dir)))
        return written

    # --- publishing / reconciliation ----------------------------------------

    async def _mark_published(self, episode_ids: list[str], digest_id: str) -> int:
        """Claim episodes for this digest. Failures are logged, never fatal."""
        marked = 0
        for episode_id in episode_ids:
            episode = await self._store.get(episode_id)
            if episode is None or episode.get("digest_id"):
                continue
            status = EpisodeStatus(episode["status"])

            def _apply(doc: Doc, did: str = digest_id) -> None:
                doc["digest_id"] = did

            try:
                if status in DIGESTABLE_STATUSES:
                    await transition(
                        self._store, episode_id, EpisodeStatus.PUBLISHED, mutate=_apply
                    )
                else:
                    # Audit-only rows keep their status (SCORED_LOW stays
                    # SCORED_LOW) but are claimed so they list exactly once.
                    await update_doc(self._store, episode_id, _apply)
                marked += 1
            except Exception as exc:
                log.warning(
                    "digest.mark_failed",
                    episode_id=episode_id,
                    status=episode.get("status"),
                    error=str(exc),
                )
        return marked

    async def _set_marking_complete(self, digest_id: str) -> None:
        def _apply(doc: Doc) -> None:
            doc["marking_complete"] = True

        try:
            await update_doc(self._store, digest_id, _apply)
        except Exception as exc:
            log.warning("digest.marking_flag_failed", digest_id=digest_id, error=str(exc))

    async def _reconcile(self, digest: Doc) -> DigestResult | None:
        """Finish an interrupted digest run, or report the existing one.

        Returns None when the caller should proceed with a fresh (regenerated)
        digest instead.
        """
        digest_id = digest["_id"]
        if not digest.get("marking_complete"):
            marked = await self._mark_published(digest.get("episode_ids") or [], digest_id)
            await self._set_marking_complete(digest_id)
            log.warning(
                "digest.reconciled",
                digest_id=digest_id,
                marked_published=marked,
                note="a previous run wrote the file but did not finish marking episodes",
            )
            return DigestResult(
                digest_id=digest_id,
                period_key=digest_id.split(":", 1)[1],
                file_path=self._settings.output.digest_dir / str(digest.get("file_path")),
                episode_ids=list(digest.get("episode_ids") or []),
                stats=dict(digest.get("stats") or {}),
                reconciled=True,
            )
        # Already complete: a manual re-run should produce a new -rN file.
        return None


def _iso_week_key(local_dt: datetime) -> str:
    iso_year, iso_week, _ = local_dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _atomic_write(base_dir: Path, relative: Path, content: str) -> Path:
    """Write via a temp file + os.replace so a sync client never sees a partial file (§5).

    An existing target is never overwritten: manual re-runs get ``-r2``, ``-r3``.
    """
    target = base_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    final = target
    revision = 2
    while final.exists():
        final = target.with_name(f"{target.stem}-r{revision}{target.suffix}")
        revision += 1
        if revision > 50:
            raise RuntimeError(f"too many revisions of {target}")

    # Same directory, so os.replace is an atomic rename within one filesystem.
    tmp = final.with_name(f".{final.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, final)
    return final
