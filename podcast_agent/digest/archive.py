"""Per-show monthly archive digests (roadmap A1).

One file per (show, month) under ``<digest_dir>/archive/<slug>/YYYY-MM.md``, kept
entirely separate from the weekly digest so the weekly signal stays clean.

A month is only written once every episode in it has been processed: a partially
processed month would produce a file that silently omits episodes still in the
queue, and archive files are not regenerated in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..db import Doc, Store, typed_sort, update_doc
from ..episodes import transition
from ..logging_setup import get_logger
from ..podcasts import PodcastRegistry
from ..state import AUDIT_ONLY_STATUSES, BACKFILL_ORIGIN, EpisodeStatus
from ..utils import iso_now, slug_month_label, utcnow
from .generate import BASIS_LABELS, _atomic_write, _build_env, _episode_views

log = get_logger(__name__)

#: Statuses meaning "this episode is still mid-pipeline".
PENDING_STATUSES = frozenset(
    {
        EpisodeStatus.NEW.value,
        EpisodeStatus.TRIAGED.value,
        EpisodeStatus.AWAITING_TRANSCRIPT.value,
        EpisodeStatus.TRANSCRIBED.value,
        EpisodeStatus.SUMMARIZED.value,
    }
)


@dataclass(slots=True)
class ArchiveResult:
    files_written: list[str] = field(default_factory=list)
    months_skipped_incomplete: int = 0
    episodes_published: int = 0
    #: Months whose file already existed but whose episodes were not fully
    #: claimed — an interrupted previous run, finished rather than repeated.
    reconciled: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_written": len(self.files_written),
            "paths": self.files_written[:20],
            "months_skipped_incomplete": self.months_skipped_incomplete,
            "episodes_published": self.episodes_published,
            "reconciled": self.reconciled,
        }


class ArchiveDigestGenerator:
    def __init__(
        self, settings: Settings, store: Store, registry: PodcastRegistry | None = None
    ) -> None:
        self._settings = settings
        self._store = store
        self._registry = registry or PodcastRegistry(settings)
        self._env = _build_env()

    async def generate(self, *, dry_run: bool = False) -> ArchiveResult:
        result = ArchiveResult()
        episodes = await self._store.find(
            {"type": "episode", "origin": BACKFILL_ORIGIN, "digest_id": None},
            sort=typed_sort("published_at", "desc"),
            limit=5000,
        )

        groups: dict[tuple[str, str], list[Doc]] = {}
        for episode in episodes:
            month = str(episode.get("archive_month") or "")
            slug = str(episode.get("podcast_slug") or "")
            if month and slug:
                groups.setdefault((slug, month), []).append(episode)

        for (slug, month), members in sorted(groups.items()):
            if any(e.get("status") in PENDING_STATUSES for e in members):
                # Writing now would omit episodes still being processed, and these
                # files are never rewritten.
                result.months_skipped_incomplete += 1
                continue
            await self._write_month(slug, month, members, result, dry_run)

        log.info("archive.generated", dry_run=dry_run, **result.as_dict())
        return result

    async def _write_month(
        self,
        slug: str,
        month: str,
        members: list[Doc],
        result: ArchiveResult,
        dry_run: bool,
    ) -> None:
        podcast = self._registry.podcast_by_slug(slug)
        podcast_name = podcast.name if podcast else slug

        summarized, listed, rest = [], [], []
        for episode in members:
            status = episode.get("status")
            if status == EpisodeStatus.READY_FOR_DIGEST.value:
                summarized.append(_episode_views(self._settings, episode, BASIS_LABELS))
            elif status == EpisodeStatus.DIGEST_DIRECT.value:
                listed.append(_episode_views(self._settings, episode, BASIS_LABELS))
            elif status in AUDIT_ONLY_STATUSES:
                rest.append(_episode_views(self._settings, episode, BASIS_LABELS, audit=True))

        summarized.sort(key=lambda v: (-v["score"], v["published_date"]))
        listed.sort(key=lambda v: v["published_date"])
        rest.sort(key=lambda v: v["published_date"])

        now = utcnow()
        rendered = self._env.get_template("archive.md.j2").render(
            podcast_name=podcast_name,
            month=month,
            month_label=slug_month_label(month),
            generated_local=now.astimezone(self._settings.tz).isoformat(timespec="seconds"),
            stats={"scanned": len(members), "summarized": len(summarized)},
            summarized=summarized,
            listed=listed,
            everything_else=rest,
            profile_version=self._settings.interest_profile_version(),
            threshold=self._settings.backfill.digest_threshold,
        )

        if dry_run:
            result.files_written.append(f"archive/{slug}/{month}.md (dry run)")
            return

        archive_id = f"archive:{slug}:{month}"
        existing = await self._store.get(archive_id)

        # Same ordering guarantee as the weekly digest (§10.3): the file lands
        # first, then episodes are claimed. A run stopped between the two leaves
        # an archive doc with marking_complete false, and the next run finishes
        # the claiming instead of writing the month a second time.
        if existing is not None and not existing.get("marking_complete"):
            claimed = await self._claim_all(members, archive_id)
            await self._mark_complete(archive_id)
            result.reconciled += 1
            result.episodes_published += claimed
            log.warning(
                "archive.reconciled",
                podcast=slug,
                month=month,
                claimed=claimed,
                note="previous run wrote the file but did not finish claiming",
            )
            return

        relative = Path("archive") / slug / f"{month}.md"
        written = _atomic_write(self._settings.output.digest_dir, relative, rendered)
        result.files_written.append(str(written.relative_to(self._settings.output.digest_dir)))

        await self._store.put(
            {
                "_id": archive_id,
                "type": "archive",
                "podcast_slug": slug,
                "month": month,
                "file_path": str(written.relative_to(self._settings.output.digest_dir)),
                "episode_ids": [e["_id"] for e in members],
                "stats": {"scanned": len(members), "summarized": len(summarized)},
                "marking_complete": False,
                "generated_at": iso_now(),
                **({"_rev": existing["_rev"]} if existing else {}),
            }
        )
        result.episodes_published += await self._claim_all(members, archive_id)
        await self._mark_complete(archive_id)

        log.info(
            "archive.month_written",
            podcast=slug,
            month=month,
            file=str(written),
            episodes=len(members),
            summarized=len(summarized),
        )

    async def _claim_all(self, members: list[Doc], archive_id: str) -> int:
        claimed = 0
        for episode in members:
            current = await self._store.get(episode["_id"])
            if current is None or current.get("digest_id"):
                continue
            try:
                await self._claim(current, archive_id)
                claimed += 1
            except Exception as exc:
                log.warning("archive.claim_failed", episode_id=episode["_id"], error=str(exc))
        return claimed

    async def _mark_complete(self, archive_id: str) -> None:
        def _apply(doc: Doc) -> None:
            doc["marking_complete"] = True

        try:
            await update_doc(self._store, archive_id, _apply)
        except Exception as exc:
            log.warning("archive.mark_complete_failed", archive_id=archive_id, error=str(exc))

    async def _claim(self, episode: Doc, archive_id: str) -> None:
        """Mark an episode as belonging to a written archive file.

        Mirrors the weekly digest: episodes that earned an entry become PUBLISHED,
        while audit-only rows keep their status but are claimed so they cannot be
        listed twice.
        """

        def _apply(doc: Doc) -> None:
            doc["digest_id"] = archive_id

        status = EpisodeStatus(episode["status"])
        if status in (EpisodeStatus.READY_FOR_DIGEST, EpisodeStatus.DIGEST_DIRECT):
            await transition(self._store, episode["_id"], EpisodeStatus.PUBLISHED, mutate=_apply)
        else:
            await update_doc(self._store, episode["_id"], _apply)
