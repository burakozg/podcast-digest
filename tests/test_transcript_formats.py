"""Transcript normalisation tests (§4 stage 3).

Format samples mirror what the configured shows actually serve: PRX-style WebVTT
(Darknet Diaries), plain text served as HTML (SANS), SRT and Podcasting 2.0 JSON.
"""

from __future__ import annotations

from podcast_agent.summarize.chunking import (
    chunk_transcript,
    needs_map_reduce,
    truncate_to_tokens,
)
from podcast_agent.transcripts.acquire import transcript_page_url
from podcast_agent.transcripts.normalize import normalize_transcript

VTT = """WEBVTT

NOTE this is a comment block
that spans lines

00:00:00.000 --> 00:00:06.080
JACK: There's a funny story from 1888. There was this undertaker

00:00:06.080 --> 00:00:10.560
named Strowger, and business was doing well for him.

00:00:10.560 --> 00:00:16.880
<v Guest>But then he started receiving fewer calls.</v>
"""

SRT = """1
00:00:00,000 --> 00:00:04,000
First subtitle line.

2
00:00:04,000 --> 00:00:08,000
Second subtitle line.
"""

VTT_WITH_CUE_IDS = """WEBVTT

7393207e-0c73-4100
00:00:01.000 --> 00:00:03.000
Line after a UUID cue id.

2
00:00:03.000 --> 00:00:05.000
Line after a numeric cue id.
"""

PODCAST_JSON = """{
  "version": "1.0.0",
  "segments": [
    {"speaker": "Alice", "startTime": 0.0, "body": "Welcome to the show."},
    {"speaker": "Alice", "startTime": 2.0, "body": "Today we discuss OT security."},
    {"speaker": "Bob", "startTime": 5.0, "body": "Thanks for having me."}
  ]
}"""


class TestVtt:
    def test_strips_header_timestamps_and_notes(self) -> None:
        result = normalize_transcript(VTT, "text/vtt", "https://x.example/a.vtt")
        assert "WEBVTT" not in result
        assert "-->" not in result
        assert "NOTE" not in result
        assert "00:00:06" not in result
        assert "undertaker" in result
        assert "Strowger" in result

    def test_removes_inline_cue_tags_but_keeps_speech(self) -> None:
        result = normalize_transcript(VTT, "text/vtt", "a.vtt")
        assert "<v Guest>" not in result
        assert "fewer calls" in result

    def test_keeps_speaker_prefixes(self) -> None:
        assert "JACK:" in normalize_transcript(VTT, "text/vtt", "a.vtt")

    def test_drops_cue_identifier_lines(self) -> None:
        result = normalize_transcript(VTT_WITH_CUE_IDS, "text/vtt", "a.vtt")
        assert "7393207e" not in result
        assert "Line after a UUID cue id." in result
        assert "Line after a numeric cue id." in result

    def test_detected_by_body_when_type_is_wrong(self) -> None:
        """Publishers mislabel constantly, so content is sniffed too."""
        result = normalize_transcript(VTT, "application/octet-stream", "no-extension")
        assert "-->" not in result
        assert "undertaker" in result

    def test_collapses_rollup_repetition(self) -> None:
        rollup = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nsame line\n\n00:00:02.000 --> 00:00:03.000\nsame line\n"
        assert normalize_transcript(rollup, "text/vtt", "a.vtt") == "same line"


class TestSrt:
    def test_strips_indices_and_timestamps(self) -> None:
        result = normalize_transcript(SRT, "application/x-subrip", "a.srt")
        assert "-->" not in result
        assert result == "First subtitle line. Second subtitle line."


class TestJson:
    def test_groups_consecutive_segments_by_speaker(self) -> None:
        result = normalize_transcript(PODCAST_JSON, "application/json", "a.json")
        assert "Alice: Welcome to the show. Today we discuss OT security." in result
        assert "Bob: Thanks for having me." in result

    def test_malformed_json_falls_through_to_text(self) -> None:
        result = normalize_transcript("{not json", "application/json", "a.json")
        assert "not json" in result


class TestHtmlAndPlain:
    def test_html_served_as_text_plain_is_still_stripped(self) -> None:
        """SANS declares text/plain and serves an HTML page."""
        html = "<html><body><p>Hello and welcome to the Stormcast.</p></body></html>"
        result = normalize_transcript(html, "text/plain", "https://x.example/t.html")
        assert "<p>" not in result
        assert "Hello and welcome to the Stormcast." in result

    def test_plain_text_passes_through(self) -> None:
        assert normalize_transcript("Just  plain   text.", "text/plain", "a.txt") == (
            "Just plain text."
        )

    def test_empty_input(self) -> None:
        assert normalize_transcript("", "text/plain", "a.txt") == ""
        assert normalize_transcript("   \n\n  ", "text/plain", "a.txt") == ""

    def test_strips_byte_order_mark(self) -> None:
        assert normalize_transcript("﻿Text here", "text/plain", "a.txt") == "Text here"


class TestChunking:
    def test_no_map_reduce_under_budget(self) -> None:
        assert needs_map_reduce("word " * 100, max_input_tokens=24_000) is False

    def test_map_reduce_over_budget(self) -> None:
        assert needs_map_reduce("word " * 50_000, max_input_tokens=1000) is True

    def test_splits_on_paragraph_boundaries(self) -> None:
        paragraphs = [f"Paragraph {i} " + "filler " * 50 for i in range(10)]
        chunks = chunk_transcript("\n\n".join(paragraphs), target_tokens=100)
        assert len(chunks) > 1
        # No chunk starts or ends mid-paragraph.
        for chunk in chunks:
            assert chunk.strip() == chunk

    def test_handles_transcript_with_no_paragraph_breaks(self) -> None:
        """Every ASR output is one long block — it must still chunk sensibly."""
        text = ". ".join(f"Sentence number {i} with some words" for i in range(400))
        chunks = chunk_transcript(text, target_tokens=200)
        assert len(chunks) > 1
        assert all(len(c) <= 200 * 4 + 100 for c in chunks)

    def test_no_content_is_lost(self) -> None:
        text = "\n\n".join(f"Para {i} content here" for i in range(30))
        chunks = chunk_transcript(text, target_tokens=50)
        rejoined = " ".join(chunks)
        for i in range(30):
            assert f"Para {i} content here" in rejoined

    def test_oversized_single_sentence_is_hard_split(self) -> None:
        chunks = chunk_transcript("x" * 5000, target_tokens=100)
        assert len(chunks) > 1
        assert all(len(c) <= 400 for c in chunks)

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_transcript("", target_tokens=100) == []
        assert chunk_transcript("   ", target_tokens=100) == []

    def test_truncate_marks_where_it_cut(self) -> None:
        result = truncate_to_tokens("word " * 10_000, max_tokens=100)
        assert "[transcript truncated]" in result
        assert len(result) < 1000

    def test_truncate_leaves_short_text_alone(self) -> None:
        assert truncate_to_tokens("short", max_tokens=100) == "short"


class TestTheTranscriptIsOnASiblingPage:
    """Publishers commonly link to show notes and keep the transcript elsewhere.

    CyberWire links to `…/2594/notes` and serves the transcript from
    `…/2594/transcript`. A selector applied to the linked page matches nothing,
    which is indistinguishable from a show that publishes no transcript — it
    cost that show description-only summaries while a full transcript sat one
    path segment away.
    """

    NOTES = "https://www.thecyberwire.com/podcasts/daily-podcast/2594/notes"

    def test_the_link_is_rewritten_to_the_sibling(self) -> None:
        assert transcript_page_url(self.NOTES, ("/notes", "/transcript")) == (
            "https://www.thecyberwire.com/podcasts/daily-podcast/2594/transcript"
        )

    def test_without_a_rule_the_link_is_untouched(self) -> None:
        assert transcript_page_url(self.NOTES, None) == self.NOTES

    def test_a_pattern_that_does_not_appear_leaves_it_alone(self) -> None:
        """A stale rule degrades to scraping the notes page, not to a wrong fetch."""
        assert transcript_page_url(self.NOTES, ("/episodes", "/x")) == self.NOTES

    def test_only_the_last_occurrence_is_rewritten(self) -> None:
        """So a pattern that also appears in the host cannot move the request."""
        link = "https://notes.example.com/show/notes"
        assert transcript_page_url(link, ("/notes", "/transcript")) == (
            "https://notes.example.com/show/transcript"
        )

    def test_a_rule_that_would_change_host_is_refused(self) -> None:
        """A substitution from a config file must not be able to redirect the
        fetch somewhere else entirely."""
        link = "https://www.thecyberwire.com/notes"
        assert transcript_page_url(link, ("www.thecyberwire.com/notes", "evil.example/x")) == link

    def test_no_link_stays_none(self) -> None:
        assert transcript_page_url(None, ("/notes", "/transcript")) is None
