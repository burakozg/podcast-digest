"""Mirroring video-digest's summaries in as read-only episodes.

Two properties carry this feature, and both are silent when broken:

* a re-import must not clobber the reader's own marks — a star or a read
  flag lost on the next poll is the feature actively working against itself;
* an imported episode must be inert in every path that publishes to the
  vault, generates the digest, or runs the pipeline. video-digest already
  wrote its own note, so a second one from here is duplication in a shared
  vault, which nothing downstream can tell apart from a real note.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from podcast_agent.config import Settings
from podcast_agent.db import MemoryStore
from podcast_agent.digest.episode_notes import summarised_episodes
from podcast_agent.digest.generate import DigestGenerator
from podcast_agent.entities import aggregate
from podcast_agent.ingest.video_digest import (
    _REFRESHED,
    SHOW_SLUG,
    VideoDigestImporter,
    _episode_doc,
)
from podcast_agent.state import (
    ACTIVE_STATUSES,
    ALLOWED_TRANSITIONS,
    DIGESTABLE_STATUSES,
    IMPORTED_ORIGIN,
    EpisodeStatus,
    IllegalTransition,
    assert_transition,
)
from podcast_agent.utils import episode_doc_id, podcast_doc_id

BASE = "http://video-digest.test:8090"


def _video(video_id: str = "vid1", **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": f"youtube:{video_id}",
        "video_id": video_id,
        "adapter": "youtube",
        "note_path": f"13 video-summaries/{video_id}.md",
        "written_at": "2026-08-01T10:00:00+00:00",
        "updated_at": "2026-08-02T10:00:00+00:00",
        "metadata": {
            "title": "A Video",
            "channel": "A Channel",
            "duration_s": 1212,
            "upload_date": "2026-07-21",
        },
        "digest": {
            "tldr": "The short version.",
            "summary_md": "Some **prose**.",
            "key_points": ["A point"],
            "entities": ["WorkOS"],
            "relevance": "high",
        },
    }
    body.update(overrides)
    return body


def _importer(settings: Settings, store: MemoryStore) -> VideoDigestImporter:
    settings.video_digest.enabled = True
    settings.video_digest.base_url = BASE
    settings.video_digest_api_key = SecretStr("test-key")
    return VideoDigestImporter(settings, store)


def _mock_export(*pages: dict[str, Any]) -> None:
    respx.get(f"{BASE}/videos").mock(side_effect=[httpx.Response(200, json=page) for page in pages])


def _page(videos: list[dict[str, Any]], nxt: dict[str, str] | None = None) -> dict[str, Any]:
    return {"videos": videos, "count": len(videos), "next": nxt}


class TestDocumentShape:
    def test_marked_imported_on_both_axes(self) -> None:
        doc = _episode_doc(_video())
        assert doc["origin"] == IMPORTED_ORIGIN
        assert doc["status"] == EpisodeStatus.IMPORTED.value

    def test_origin_is_explicit_because_migrate_would_stamp_it_routine(self) -> None:
        """`migrate.backfill_origins` gives any episode lacking an origin the
        routine one at startup — which would feed imports to the pipeline."""
        assert "origin" in _episode_doc(_video())

    def test_published_at_is_the_videos_release_date(self) -> None:
        """Not the import time: that column is the release date for every other
        show on the screen, and it sorts on it."""
        assert _episode_doc(_video())["published_at"] == "2026-07-21T00:00:00+00:00"

    def test_entities_are_empty_so_no_topic_page_can_be_touched(self) -> None:
        """Second line of defence behind entities.aggregate's OURS_ONLY."""
        assert _episode_doc(_video())["tier1"]["entities"] == []

    def test_fields_that_would_wake_other_subsystems_are_absent(self) -> None:
        doc = _episode_doc(_video())
        # tier0 -> backfill/estimate's escalation ratio.
        assert "tier0" not in doc
        # transcript_at -> retention's transcript expiry sweep.
        assert "transcript_at" not in doc
        # feedback -> signals.collect selects {"$exists": True}, which a
        # null value would still match.
        assert "feedback" not in doc

    def test_summary_is_carried_over_for_reading(self) -> None:
        tier1 = _episode_doc(_video())["tier1"]
        assert tier1["summary_md"] == "Some **prose**."
        assert tier1["why_it_matters"] == "The short version."
        assert tier1["key_takeaways"] == ["A point"]

    def test_id_is_stable_across_imports(self) -> None:
        assert _episode_doc(_video())["_id"] == _episode_doc(_video())["_id"]
        assert _episode_doc(_video())["_id"] == episode_doc_id(SHOW_SLUG, "youtube:vid1")


class TestInertness:
    def test_status_is_absent_from_every_set_that_would_act_on_it(self) -> None:
        imported = EpisodeStatus.IMPORTED
        assert imported not in ACTIVE_STATUSES
        assert imported not in DIGESTABLE_STATUSES

    def test_status_is_terminal_so_the_pipeline_cannot_move_it(self) -> None:
        """Not merely unswept — a transition raises rather than silently
        succeeding, so a future stage that forgot the origin clause fails
        loudly instead of pulling an import into the pipeline."""
        assert EpisodeStatus.IMPORTED not in ALLOWED_TRANSITIONS
        with pytest.raises(IllegalTransition):
            assert_transition(EpisodeStatus.IMPORTED, EpisodeStatus.TRIAGED)


class TestImport:
    @respx.mock
    @pytest.mark.asyncio
    async def test_creates_episodes_and_the_show(
        self, settings: Settings, store: MemoryStore
    ) -> None:
        _mock_export(_page([_video("a"), _video("b")]))

        stats = await _importer(settings, store).run()

        assert stats.created == 2
        assert await store.get(podcast_doc_id(SHOW_SLUG)) is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_the_show_is_never_polled(self, settings: Settings, store: MemoryStore) -> None:
        """A console-source podcast with enabled:true is fetched by
        FeedIngester — and this show's feed_url resolves nowhere."""
        _mock_export(_page([_video()]))

        await _importer(settings, store).run()

        show = await store.get(podcast_doc_id(SHOW_SLUG))
        assert show is not None
        assert show["overrides"]["enabled"] is False
        # Identical, so seed_podcast_docs never rewrites the doc.
        assert show["feed_url"] == show["overrides"]["feed_url"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_does_nothing_when_unconfigured(
        self, settings: Settings, store: MemoryStore
    ) -> None:
        stats = await VideoDigestImporter(settings, store).run()
        assert stats.fetched == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_follows_the_cursor_across_pages(
        self, settings: Settings, store: MemoryStore
    ) -> None:
        _mock_export(
            _page(
                [_video("a")], nxt={"since": "2026-08-01T10:00:00+00:00", "since_id": "youtube:a"}
            ),
            _page([_video("b")]),
        )

        stats = await _importer(settings, store).run()

        assert stats.fetched == 2


class TestNothingHerePublishesTheImportToTheVault:
    """The constraint the whole design exists for.

    video-digest has already written `13 video-summaries/<video>.md`. Every
    assertion below is a path that would write a *second* copy of the same
    summary into the same vault, where nothing downstream could tell the two
    apart. Each is seeded with a fully populated `tier1` — entities included —
    so that what excludes it is the origin clause and not an empty field.
    """

    def _imported(self) -> Any:
        doc = _episode_doc(_video())
        doc["tier1"] = {**doc["tier1"], "entities": ["Volt Typhoon"]}
        doc["digest_id"] = None
        return doc

    @pytest.mark.asyncio
    async def test_no_note_under_11_podcasts_episodes(self, store: MemoryStore) -> None:
        store.seed(self._imported())
        assert await summarised_episodes(store) == []

    @pytest.mark.asyncio
    async def test_no_mention_reaches_a_99_topics_note(self, store: MemoryStore) -> None:
        store.seed(self._imported())
        assert await aggregate(store) == {}

    @pytest.mark.asyncio
    async def test_the_weekly_digest_does_not_collect_it(
        self, settings: Settings, store: MemoryStore
    ) -> None:
        """`_collect` keys off `digest_id: None`, not status, so a novel status
        alone would not have excluded it — the origin clause is what does."""
        store.seed(self._imported())
        period_to = datetime(2026, 8, 8, tzinfo=UTC)

        collected = await DigestGenerator(settings, store)._collect(
            datetime(2026, 8, 1, tzinfo=UTC), period_to
        )

        assert collected == []


class TestReImportPreservesTheReadersMarks:
    """The correctness requirement of the whole feature."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_read_and_starred_survive_a_reimport(
        self, settings: Settings, store: MemoryStore
    ) -> None:
        _mock_export(_page([_video()]), _page([_video()]))
        importer = _importer(settings, store)
        await importer.run()

        doc_id = episode_doc_id(SHOW_SLUG, "youtube:vid1")
        marked = await store.get(doc_id)
        assert marked is not None
        marked["read_at"] = "2026-09-01T00:00:00+00:00"
        marked["starred"] = True
        marked["starred_at"] = "2026-09-01T00:00:00+00:00"
        await store.put(marked)

        await importer.run()

        after = await store.get(doc_id)
        assert after is not None
        assert after["read_at"] == "2026-09-01T00:00:00+00:00", "a re-import unread the episode"
        assert after["starred"] is True, "a re-import lost the star"
        assert after["starred_at"] == "2026-09-01T00:00:00+00:00"

    @respx.mock
    @pytest.mark.asyncio
    async def test_a_changed_summary_is_refreshed(
        self, settings: Settings, store: MemoryStore
    ) -> None:
        rewritten = _video()
        rewritten["digest"] = {**rewritten["digest"], "summary_md": "Rewritten prose."}
        _mock_export(_page([_video()]), _page([rewritten]))
        importer = _importer(settings, store)
        await importer.run()

        stats = await importer.run()

        doc = await store.get(episode_doc_id(SHOW_SLUG, "youtube:vid1"))
        assert doc is not None
        assert doc["tier1"]["summary_md"] == "Rewritten prose."
        assert stats.updated == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_an_unchanged_reimport_writes_nothing(
        self, settings: Settings, store: MemoryStore
    ) -> None:
        _mock_export(_page([_video()]), _page([_video()]))
        importer = _importer(settings, store)
        await importer.run()

        stats = await importer.run()

        assert stats.unchanged == 1
        assert stats.updated == 0


class TestTheManualTrigger:
    """`POST /runs/video-digest`, so a pull need not wait for the cron."""

    def _client(self, tmp_path: Any, store: MemoryStore, **over: Any) -> Any:
        from fastapi.testclient import TestClient
        from helpers import FakeLLM, make_settings

        from podcast_agent.main import build_app

        return TestClient(build_app(make_settings(tmp_path, **over), store=store, llm=FakeLLM()))

    def test_unconfigured_is_a_409_not_a_zero_result(
        self, tmp_path: Any, store: MemoryStore
    ) -> None:
        """Zero fetched is also what a healthy run over an empty export
        returns, so a missing key must not be reported as "nothing new"."""
        with self._client(tmp_path, store) as client:
            resp = client.post("/api/v1/runs/video-digest")

        assert resp.status_code == 409

    @respx.mock
    def test_a_configured_pull_reports_what_it_did(self, tmp_path: Any, store: MemoryStore) -> None:
        _mock_export(_page([_video("a"), _video("b")]))
        over = {
            "video_digest": {"enabled": True, "base_url": BASE},
            "video_digest_api_key": SecretStr("test-key"),
        }

        with self._client(tmp_path, store, **over) as client:
            body = client.post("/api/v1/runs/video-digest").json()

        assert body["job"] == "video_digest_import"
        assert body["result"]["created"] == 2


class TestTheDateShownIsTheVideosOwn:
    """`published_at` drives the console's date column and its sort order.

    Import time would flatten real history — the first live import spanned
    thirteen months of uploads and two days of imports — and would make one
    column mean two different things across shows.
    """

    def _at(self, **metadata: Any) -> Any:
        video = _video()
        video["metadata"] = {**video["metadata"], **metadata}
        return _episode_doc(video)["published_at"]

    def test_a_bare_date_is_widened_to_midnight_utc(self) -> None:
        """A short string would sort against full timestamps as text, putting
        every video below every podcast episode published the same day."""
        assert self._at(upload_date="2026-03-25") == "2026-03-25T00:00:00+00:00"

    def test_yt_dlps_own_compact_form_is_accepted(self) -> None:
        assert self._at(upload_date="20260325") == "2026-03-25T00:00:00+00:00"

    @pytest.mark.parametrize("bad", ["", "banana", "2026-03", "20260231", "not-a-date"])
    def test_an_unusable_date_falls_back_to_the_write_time(self, bad: str) -> None:
        """Never epoch, and never null. Either would bury the episode at the
        bottom of a screen sorted by this field, where nobody would find it —
        a silent failure, since the row still exists and looks fine."""
        assert self._at(upload_date=bad) == "2026-08-01T10:00:00+00:00"

    def test_a_missing_upload_date_falls_back_too(self) -> None:
        video = _video()
        video["metadata"] = {k: v for k, v in video["metadata"].items() if k != "upload_date"}
        assert _episode_doc(video)["published_at"] == "2026-08-01T10:00:00+00:00"

    def test_an_old_video_keeps_its_old_date(self) -> None:
        """The whole point: a two-year-old talk reads as two years old."""
        assert self._at(upload_date="2024-05-02") == "2024-05-02T00:00:00+00:00"

    def test_a_reimport_refreshes_it(self) -> None:
        """`published_at` is in `_REFRESHED`, so the 28 already-imported rows
        are corrected by the next poll rather than needing a migration."""
        assert "published_at" in _REFRESHED
