"""Ingestion tests (§4 stage 1). All HTTP is respx-mocked; no network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from helpers import make_settings

from podcast_agent.db import MemoryStore
from podcast_agent.ingest.feeds import (
    CIRCUIT_BREAKER_THRESHOLD,
    FEED_DESCRIPTION_CHARS,
    Ingestor,
    _duration_seconds,
    _stable_guid,
)
from podcast_agent.net import UrlGuard, build_client
from podcast_agent.state import EpisodeStatus
from podcast_agent.utils import episode_doc_id, iso, podcast_doc_id

FEED = (Path(__file__).parent / "fixtures" / "feed_basic.xml").read_text()
FEED_URL = "https://example.com/feed.xml"
PRIORITY_FEED_URL = "https://priority-show.net/feed.xml"


@pytest.fixture
def one_show_settings(tmp_path: Path):
    return make_settings(
        tmp_path,
        podcasts=[
            {
                "slug": "test-show",
                "name": "Test Show",
                "feed_url": FEED_URL,
                "priority": "med",
            }
        ],
        # The fixture's pubDates are fixed, so a lookback window would make these
        # tests drift into failure as real time passes. Backfill behaviour has its
        # own tests below, with dates relative to now.
        pipeline={"initial_lookback_days": 0},
    )


async def run_ingest(settings, store: MemoryStore) -> object:
    async with build_client() as client:
        ingestor = Ingestor(settings, store, client, UrlGuard(settings.security))
        await ingestor.seed_podcast_docs()
        return await ingestor.run()


class TestBasicIngestion:
    @respx.mock
    async def test_creates_episodes_from_feed(self, one_show_settings, store: MemoryStore) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        stats = await run_ingest(one_show_settings, store)

        episodes = store.docs_of_type("episode")
        titles = {e["title"] for e in episodes}
        # Ep 3 and Ep 2 have audio on allowed hosts; Ep 1 has no enclosure and
        # Ep 0's enclosure host is neither the feed's domain nor allowlisted.
        assert len(episodes) == 2
        assert any("PLC malware" in t for t in titles)
        assert any("Agent security" in t for t in titles)
        assert stats.episodes_created == 2  # type: ignore[attr-defined]
        # Ep 1 (no enclosure) and Ep 0 (unlisted host) are both unsupported.
        assert stats.entries_unsupported == 2  # type: ignore[attr-defined]

    @respx.mock
    async def test_all_episodes_start_as_new(self, one_show_settings, store: MemoryStore) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        assert all(e["status"] == EpisodeStatus.NEW.value for e in store.docs_of_type("episode"))

    @respx.mock
    async def test_html_is_stripped_from_descriptions(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        ep3 = next(e for e in store.docs_of_type("episode") if "PLC" in e["title"])
        assert "<b>" not in ep3["description_raw"]
        assert "<p>" not in ep3["description_raw"]
        # Entities are decoded, not left as &amp;.
        assert "Modbus abuse & segmentation" in ep3["description_raw"]

    @respx.mock
    async def test_captures_feed_transcripts_in_preference_order(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        """All transcript elements must survive, best MIME type first.

        feedparser keeps only the last <podcast:transcript> per item, so this also
        pins the raw-XML recovery path — losing it would silently discard a
        fallback format for shows that publish several.
        """
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        ep2 = next(e for e in store.docs_of_type("episode") if "Agent" in e["title"])
        transcripts = ep2["feed_transcripts"]
        assert len(transcripts) == 2
        assert transcripts[0]["type"] == "text/plain"

    @respx.mock
    async def test_parses_both_duration_formats(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        by_title = {e["title"]: e for e in store.docs_of_type("episode")}
        ep3 = next(v for k, v in by_title.items() if "PLC" in k)
        ep2 = next(v for k, v in by_title.items() if "Agent" in k)
        assert ep3["duration_s"] == 3720  # "1:02:00"
        assert ep2["duration_s"] == 2400  # bare seconds

    @respx.mock
    async def test_rejects_enclosure_on_unlisted_domain(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        """§10.2: enclosure must share the feed's domain or be allowlisted."""
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        assert not any("unexpected" in e["title"] for e in store.docs_of_type("episode"))


class TestIdempotency:
    @respx.mock
    async def test_rerunning_creates_nothing_new(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        first = {e["_id"]: e["_rev"] for e in store.docs_of_type("episode")}

        stats = await run_ingest(one_show_settings, store)
        second = {e["_id"]: e["_rev"] for e in store.docs_of_type("episode")}

        assert stats.episodes_created == 0  # type: ignore[attr-defined]
        assert stats.episodes_existing == 2  # type: ignore[attr-defined]
        # Existing docs are not even rewritten, so revs are untouched.
        assert first == second

    @respx.mock
    async def test_episode_id_is_stable_across_runs(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        assert episode_doc_id("test-show", "ep-3-guid") in {
            e["_id"] for e in store.docs_of_type("episode")
        }

    @respx.mock
    async def test_reprocessed_episode_keeps_its_pipeline_state(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        """Re-ingesting must never reset an episode that has moved on."""
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        target = episode_doc_id("test-show", "ep-3-guid")
        doc = await store.get(target)
        assert doc is not None
        doc["status"] = EpisodeStatus.PUBLISHED.value
        await store.put(doc)

        await run_ingest(one_show_settings, store)
        after = await store.get(target)
        assert after is not None
        assert after["status"] == EpisodeStatus.PUBLISHED.value


class TestConditionalGet:
    @respx.mock
    async def test_stores_and_replays_validators(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        route = respx.get(FEED_URL).mock(
            return_value=httpx.Response(
                200,
                text=FEED,
                headers={"ETag": '"abc"', "Last-Modified": "Wed, 29 Jul 2026 00:00:00 GMT"},
            )
        )
        await run_ingest(one_show_settings, store)
        pdoc = await store.get(podcast_doc_id("test-show"))
        assert pdoc is not None
        assert pdoc["etag"] == '"abc"'

        await run_ingest(one_show_settings, store)
        assert route.calls[-1].request.headers["If-None-Match"] == '"abc"'
        assert "If-Modified-Since" in route.calls[-1].request.headers

    @respx.mock
    async def test_304_is_a_cheap_no_op(self, one_show_settings, store: MemoryStore) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(304))
        stats = await run_ingest(one_show_settings, store)
        assert stats.feeds_unchanged == 1  # type: ignore[attr-defined]
        assert store.docs_of_type("episode") == []

    @respx.mock
    async def test_changed_feed_url_resets_validators(self, tmp_path: Path) -> None:
        """Otherwise the new feed gets 304'd against the old feed's ETag."""
        store = MemoryStore()
        first = make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "T", "feed_url": FEED_URL}],
        )
        respx.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"ETag": '"old"'})
        )
        await run_ingest(first, store)

        moved = make_settings(
            tmp_path,
            podcasts=[
                {"slug": "test-show", "name": "T", "feed_url": "https://example.com/new.xml"}
            ],
        )
        route = respx.get("https://example.com/new.xml").mock(
            return_value=httpx.Response(200, text=FEED)
        )
        await run_ingest(moved, store)
        assert "If-None-Match" not in route.calls[-1].request.headers


class TestBackfillGuard:
    @respx.mock
    async def test_old_episodes_skipped_on_a_fresh_show(self, tmp_path: Path) -> None:
        """§10.4: a fresh install must not summarise years of archives."""
        store = MemoryStore()
        settings = make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "T", "feed_url": FEED_URL}],
            pipeline={"initial_lookback_days": 1},
        )
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        stats = await run_ingest(settings, store)
        assert stats.episodes_created == 0  # type: ignore[attr-defined]
        # All three enclosure-bearing entries predate the 1-day window.
        assert stats.entries_too_old == 3  # type: ignore[attr-defined]

    @respx.mock
    async def test_recent_episodes_are_ingested(self, tmp_path: Path) -> None:
        recent = datetime.now(UTC) - timedelta(days=1)
        feed = f"""<?xml version="1.0"?>
        <rss version="2.0"><channel><title>T</title>
        <item><title>Fresh</title><guid>fresh-1</guid>
        <pubDate>{recent.strftime("%a, %d %b %Y %H:%M:%S +0000")}</pubDate>
        <description>New enough.</description>
        <enclosure url="https://cdn-host.net/f.mp3" length="100" type="audio/mpeg"/>
        </item></channel></rss>"""
        store = MemoryStore()
        settings = make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "T", "feed_url": FEED_URL}],
            pipeline={"initial_lookback_days": 7},
        )
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed))
        stats = await run_ingest(settings, store)
        assert stats.episodes_created == 1  # type: ignore[attr-defined]

    @respx.mock
    async def test_cutoff_does_not_apply_once_a_show_has_history(self, tmp_path: Path) -> None:
        """A slow publisher backfilling an older episode must still be picked up."""
        store = MemoryStore()
        settings = make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "T", "feed_url": FEED_URL}],
            pipeline={"initial_lookback_days": 1},
        )
        # Pre-existing history for this show.
        store.seed(
            {
                "_id": episode_doc_id("test-show", "old"),
                "type": "episode",
                "podcast_slug": "test-show",
                "status": "PUBLISHED",
                "published_at": iso(datetime(2026, 1, 1, tzinfo=UTC)),
            }
        )
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        stats = await run_ingest(settings, store)
        assert stats.episodes_created == 2  # type: ignore[attr-defined]


class TestFailureHandling:
    @respx.mock
    async def test_one_bad_feed_does_not_stop_the_others(self, settings) -> None:
        store = MemoryStore()
        respx.get(FEED_URL).mock(return_value=httpx.Response(500))
        respx.get(PRIORITY_FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        stats = await run_ingest(settings, store)
        assert stats.feeds_failed == 1  # type: ignore[attr-defined]
        # The healthy feed still produced episodes.
        assert len(store.docs_of_type("episode")) == 2

    @respx.mock
    async def test_failure_increments_the_circuit_breaker(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(503))
        await run_ingest(one_show_settings, store)
        pdoc = await store.get(podcast_doc_id("test-show"))
        assert pdoc is not None
        assert pdoc["consecutive_failures"] == 1
        assert pdoc["last_error"]

    @respx.mock
    async def test_open_circuit_skips_the_poll(self, one_show_settings, store: MemoryStore) -> None:
        """§10.3: a persistently broken feed backs off to daily attempts."""
        route = respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)  # seeds the podcast doc
        calls_before = len(route.calls)

        pdoc = await store.get(podcast_doc_id("test-show"))
        assert pdoc is not None
        pdoc["consecutive_failures"] = CIRCUIT_BREAKER_THRESHOLD
        await store.put(pdoc)

        stats = await run_ingest(one_show_settings, store)
        assert stats.feeds_skipped_backoff == 1  # type: ignore[attr-defined]
        assert len(route.calls) == calls_before  # no HTTP call made

    @respx.mock
    async def test_success_resets_the_breaker(self, one_show_settings, store: MemoryStore) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=FEED))
        await run_ingest(one_show_settings, store)
        pdoc = await store.get(podcast_doc_id("test-show"))
        assert pdoc is not None
        pdoc["consecutive_failures"] = 3
        pdoc["last_polled_at"] = None
        await store.put(pdoc)

        await run_ingest(one_show_settings, store)
        pdoc = await store.get(podcast_doc_id("test-show"))
        assert pdoc is not None
        assert pdoc["consecutive_failures"] == 0
        assert pdoc["last_error"] is None

    @respx.mock
    async def test_garbage_body_is_reported_not_crashed_on(
        self, one_show_settings, store: MemoryStore
    ) -> None:
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text="<<not xml at all"))
        stats = await run_ingest(one_show_settings, store)
        assert stats.feeds_failed == 1  # type: ignore[attr-defined]
        assert store.docs_of_type("episode") == []

    @respx.mock
    async def test_empty_feed_is_fine(self, one_show_settings, store: MemoryStore) -> None:
        empty = '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title></channel></rss>'
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=empty))
        stats = await run_ingest(one_show_settings, store)
        assert stats.feeds_failed == 0  # type: ignore[attr-defined]
        assert stats.episodes_created == 0  # type: ignore[attr-defined]


class TestEntryParsingHelpers:
    def test_guid_prefers_explicit_id(self) -> None:
        assert _stable_guid({"id": "the-id"}, "https://x/a.mp3") == "the-id"

    def test_guid_falls_back_to_enclosure_url(self) -> None:
        assert _stable_guid({}, "https://x/a.mp3") == "https://x/a.mp3"

    def test_guid_last_resort_is_title_and_date(self) -> None:
        guid = _stable_guid({"title": "T", "published": "yesterday"}, None)
        assert guid == "T|yesterday"

    def test_guid_none_when_nothing_usable(self) -> None:
        assert _stable_guid({}, None) is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("3720", 3720),
            ("62:00", 3720),
            ("1:02:00", 3720),
            ("6:59", 419),
            ("", None),
            ("nonsense", None),
            ("1:2:3:4", None),
            (None, None),
        ],
    )
    def test_duration_parsing(self, raw: str | None, expected: int | None) -> None:
        assert _duration_seconds({"itunes_duration": raw} if raw is not None else {}) == expected


class TestFeedMetadataIsActuallyCaptured:
    """Regression: descriptions and cadence never arrived in production.

    They are read from the response body, and every already-polled feed has a
    stored etag, so every subsequent poll returned 304 with no body. A podcast
    would have waited for its next *episode* to gain a description — up to a
    month for a monthly show, and forever for one that had stopped publishing.
    """

    def _feed(self, items: int = 4) -> str:
        entries = "".join(
            f"""<item><title>Ep {i}</title><guid>ep-{i}</guid>
            <pubDate>{(datetime.now(UTC) - timedelta(days=7 * i)).strftime("%a, %d %b %Y %H:%M:%S +0000")}</pubDate>
            <description>Body.</description>
            <enclosure url="https://cdn-host.net/{i}.mp3" length="1" type="audio/mpeg"/>
            </item>"""
            for i in range(items)
        )
        return f"""<?xml version="1.0"?><rss version="2.0"><channel>
        <title>T</title><description>Weekly infosec news.</description>
        {entries}</channel></rss>"""

    def _settings(self, tmp_path: Path):
        return make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "T", "feed_url": FEED_URL}],
            pipeline={"initial_lookback_days": 400},
        )

    @respx.mock
    async def test_a_feed_with_stored_validators_is_still_fetched_in_full(
        self, tmp_path: Path
    ) -> None:
        """The actual bug: an etag from a previous version blocked metadata."""
        store = MemoryStore()
        settings = self._settings(tmp_path)
        store.seed(
            {
                "_id": podcast_doc_id("test-show"),
                "type": "podcast",
                "slug": "test-show",
                "etag": 'W/"abc"',
                "last_modified": "Wed, 01 Jul 2026 00:00:00 GMT",
                # No feed_metadata_at: nothing has ever been captured.
            }
        )
        route = respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=self._feed()))
        await run_ingest(settings, store)

        sent = route.calls[0].request.headers
        assert "if-none-match" not in sent, "conditional GET blocked the first capture"
        assert "if-modified-since" not in sent
        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        assert doc["description"] == "Weekly infosec news."
        assert doc["feed_metadata_at"]

    @respx.mock
    async def test_once_captured_it_goes_back_to_conditional_requests(self, tmp_path: Path) -> None:
        """The full fetch is a one-off, not a permanent loss of bandwidth."""
        store = MemoryStore()
        settings = self._settings(tmp_path)
        route = respx.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=self._feed(), headers={"etag": 'W/"xyz"'})
        )
        await run_ingest(settings, store)
        await run_ingest(settings, store)

        assert "if-none-match" not in route.calls[0].request.headers
        assert route.calls[1].request.headers.get("if-none-match") == 'W/"xyz"'

    @respx.mock
    async def test_cadence_is_measured_from_the_feed(self, tmp_path: Path) -> None:
        """Not from held episodes: a new podcast holds too few to measure."""
        store = MemoryStore()
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=self._feed(6)))
        await run_ingest(self._settings(tmp_path), store)

        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        assert doc["feed_cadence"] == "~weekly"
        assert "median" in doc["feed_cadence_detail"]

    @respx.mock
    async def test_transcript_coverage_is_counted(self, tmp_path: Path) -> None:
        """The number that decides whether ASR is worth turning on."""
        store = MemoryStore()
        feed = """<?xml version="1.0"?>
        <rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">
        <channel><title>T</title>
        <item><title>A</title><guid>a</guid>
          <pubDate>Mon, 27 Jul 2026 00:00:00 +0000</pubDate>
          <podcast:transcript url="https://cdn-host.net/a.txt" type="text/plain"/>
          <enclosure url="https://cdn-host.net/a.mp3" length="1" type="audio/mpeg"/></item>
        <item><title>B</title><guid>b</guid>
          <pubDate>Mon, 20 Jul 2026 00:00:00 +0000</pubDate>
          <enclosure url="https://cdn-host.net/b.mp3" length="1" type="audio/mpeg"/></item>
        </channel></rss>"""
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed))
        await run_ingest(self._settings(tmp_path), store)

        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        assert doc["feed_entries_seen"] == 2
        assert doc["feed_transcripts_seen"] == 1

    @respx.mock
    async def test_a_304_does_not_blank_what_was_captured(self, tmp_path: Path) -> None:
        store = MemoryStore()
        settings = self._settings(tmp_path)
        respx.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=self._feed(), headers={"etag": 'W/"1"'})
        )
        await run_ingest(settings, store)

        respx.get(FEED_URL).mock(return_value=httpx.Response(304))
        await run_ingest(settings, store)

        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        assert doc["description"] == "Weekly infosec news."
        assert doc["feed_cadence"] == "~weekly"


class TestFeedDescription:
    """The podcast's own blurb, captured at poll time for the console."""

    def _feed(self, channel_extra: str) -> str:
        recent = datetime.now(UTC) - timedelta(days=1)
        return f"""<?xml version="1.0"?>
        <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
        <channel><title>T</title>{channel_extra}
        <item><title>Fresh</title><guid>fresh-1</guid>
        <pubDate>{recent.strftime("%a, %d %b %Y %H:%M:%S +0000")}</pubDate>
        <description>New enough.</description>
        <enclosure url="https://cdn-host.net/f.mp3" length="100" type="audio/mpeg"/>
        </item></channel></rss>"""

    async def _poll(self, tmp_path: Path, feed: str, store: MemoryStore | None = None):
        store = store or MemoryStore()
        settings = make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "T", "feed_url": FEED_URL}],
            pipeline={"initial_lookback_days": 7},
        )
        respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=feed))
        await run_ingest(settings, store)
        doc = await store.get(podcast_doc_id("test-show"))
        assert doc is not None
        return store, doc

    @respx.mock
    async def test_the_channel_description_is_stored(self, tmp_path: Path) -> None:
        _, doc = await self._poll(
            tmp_path, self._feed("<description>Weekly infosec news.</description>")
        )
        assert doc["description"] == "Weekly infosec news."

    @respx.mock
    async def test_an_itunes_only_feed_still_yields_a_description(self, tmp_path: Path) -> None:
        """Plenty of podcast feeds carry itunes:summary and no channel description."""
        _, doc = await self._poll(
            tmp_path,
            self._feed("<itunes:summary>Security, weekly.</itunes:summary>"),
        )
        assert doc["description"] == "Security, weekly."

    @respx.mock
    async def test_html_is_reduced_to_text(self, tmp_path: Path) -> None:
        """Feed content is untrusted and reaches the console."""
        _, doc = await self._poll(
            tmp_path,
            self._feed(
                "<description>&lt;p&gt;Hosted by &lt;b&gt;two&lt;/b&gt; people."
                "&lt;/p&gt;&lt;script&gt;alert(1)&lt;/script&gt;</description>"
            ),
        )
        assert "<" not in doc["description"]
        assert "alert(1)" not in doc["description"]
        assert "Hosted by two people." in doc["description"]

    @respx.mock
    async def test_a_long_description_is_capped(self, tmp_path: Path) -> None:
        """A table cell, not an about page."""
        _, doc = await self._poll(
            tmp_path, self._feed(f"<description>{'word ' * 400}</description>")
        )
        assert 0 < len(doc["description"]) <= FEED_DESCRIPTION_CHARS + 3

    @respx.mock
    async def test_a_feed_with_no_description_stores_nothing(self, tmp_path: Path) -> None:
        _, doc = await self._poll(tmp_path, self._feed(""))
        assert not doc.get("description")

    @respx.mock
    async def test_an_unchanged_feed_does_not_blank_the_stored_description(
        self, tmp_path: Path
    ) -> None:
        """A 304 carries no body, so there is nothing to re-read it from."""
        store, doc = await self._poll(
            tmp_path, self._feed("<description>Weekly infosec news.</description>")
        )
        assert doc["description"] == "Weekly infosec news."

        settings = make_settings(
            tmp_path,
            podcasts=[{"slug": "test-show", "name": "T", "feed_url": FEED_URL}],
            pipeline={"initial_lookback_days": 7},
        )
        respx.get(FEED_URL).mock(return_value=httpx.Response(304))
        await run_ingest(settings, store)
        after = await store.get(podcast_doc_id("test-show"))
        assert after is not None
        assert after["description"] == "Weekly infosec news."


class TestAWideningAllowlistIsActuallyApplied:
    """A 304 means the entries are never re-read.

    So widening the CDN allowlist changed nothing for any feed that had not
    published since: entries whose enclosure was rejected under the old rules
    stayed rejected, because the body they would be re-read from never
    arrived. A podcast sat at zero episodes across a fix that was deployed and
    correct, and the only symptom was a quiet feed.
    """

    def _settings(self, tmp_path: Path, allowlist: list[str]):
        return make_settings(
            tmp_path,
            podcasts=[
                {"slug": "test-show", "name": "Test Show", "feed_url": FEED_URL, "priority": "med"}
            ],
            pipeline={"initial_lookback_days": 0},
            security={"enforce_domain_allowlist": True, "cdn_allowlist": allowlist},
        )

    @respx.mock
    async def test_a_changed_allowlist_forces_one_full_fetch(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        route = respx.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"ETag": '"abc"'})
        )
        await run_ingest(self._settings(tmp_path, ["cdn-host.net"]), store)
        assert "If-None-Match" not in route.calls[0].request.headers

        # Same rules: the validator is replayed, as it should be.
        await run_ingest(self._settings(tmp_path, ["cdn-host.net"]), store)
        assert route.calls[-1].request.headers["If-None-Match"] == '"abc"'

        # Rules changed: ask for the body, not a 304.
        await run_ingest(self._settings(tmp_path, ["cdn-host.net", "newly-allowed.net"]), store)
        assert "If-None-Match" not in route.calls[-1].request.headers

    @respx.mock
    async def test_it_returns_to_conditional_once_re_read(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """One full fetch per feed after a change, not a permanent one."""
        route = respx.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"ETag": '"abc"'})
        )
        await run_ingest(self._settings(tmp_path, ["cdn-host.net"]), store)
        wider = self._settings(tmp_path, ["cdn-host.net", "newly-allowed.net"])
        await run_ingest(wider, store)
        await run_ingest(wider, store)
        assert route.calls[-1].request.headers["If-None-Match"] == '"abc"'

    @respx.mock
    async def test_a_feed_never_read_under_these_rules_is_fetched_in_full(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """Covers the deployment that hit this: podcast documents predate the
        marker entirely, so every feed is due one honest re-read."""
        respx.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"ETag": '"abc"'})
        )
        settings = self._settings(tmp_path, ["cdn-host.net"])
        await run_ingest(settings, store)
        pdoc = await store.get(podcast_doc_id("test-show"))
        assert pdoc is not None
        del pdoc["intake_rules"]
        await store.put(pdoc)

        route = respx.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"ETag": '"abc"'})
        )
        await run_ingest(settings, store)
        assert "If-None-Match" not in route.calls[-1].request.headers

    @respx.mock
    async def test_the_marker_is_written_even_when_unchanged(
        self, tmp_path: Path, store: MemoryStore
    ) -> None:
        """A 304 confirms the feed is as it was when last read in full, so the
        marker is current — otherwise every poll would force a full fetch."""
        settings = self._settings(tmp_path, ["cdn-host.net"])
        respx.get(FEED_URL).mock(
            return_value=httpx.Response(200, text=FEED, headers={"ETag": '"abc"'})
        )
        await run_ingest(settings, store)
        respx.get(FEED_URL).mock(return_value=httpx.Response(304))
        await run_ingest(settings, store)
        pdoc = await store.get(podcast_doc_id("test-show"))
        assert pdoc is not None
        assert pdoc["intake_rules"] == UrlGuard(settings.security).fingerprint

    def test_the_fingerprint_ignores_ordering_and_case(self, tmp_path: Path) -> None:
        """Rewriting the list in a different order is not a rule change."""
        one = UrlGuard(self._settings(tmp_path, ["a.net", "B.net"]).security)
        two = UrlGuard(self._settings(tmp_path, ["b.net", "A.net"]).security)
        assert one.fingerprint == two.fingerprint

    def test_the_fingerprint_moves_when_enforcement_is_switched_off(self, tmp_path: Path) -> None:
        strict = make_settings(
            tmp_path, security={"enforce_domain_allowlist": True, "cdn_allowlist": ["a.net"]}
        )
        lax = make_settings(
            tmp_path, security={"enforce_domain_allowlist": False, "cdn_allowlist": ["a.net"]}
        )
        assert UrlGuard(strict.security).fingerprint != UrlGuard(lax.security).fingerprint
