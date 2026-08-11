"""Sanitisation tests (§10.2). Inputs here are modelled on hostile feed content."""

from __future__ import annotations

import pytest

from podcast_agent.models import Tier1Result
from podcast_agent.sanitize import (
    html_to_text,
    md_escape_inline,
    md_escape_table_cell,
    safe_url,
    sanitize_bullet,
    sanitize_md_block,
    slugify,
)


class TestHtmlToText:
    def test_strips_tags_and_unescapes_entities(self) -> None:
        assert html_to_text("<p>Tools &amp; tactics</p>") == "Tools & tactics"

    def test_removes_script_contents_entirely(self) -> None:
        """bleach's strip=True keeps inner text, so script bodies are pre-removed."""
        result = html_to_text("<p>Hello</p><script>steal(document.cookie)</script>")
        assert "steal" not in result
        assert "cookie" not in result
        assert "Hello" in result

    def test_removes_style_and_iframe_contents(self) -> None:
        assert "background" not in html_to_text("<style>body{background:red}</style>ok")
        assert "evil" not in html_to_text("<iframe>evil</iframe>ok")

    def test_double_encoded_tags_do_not_survive(self) -> None:
        """&lt;script&gt; becomes a tag after unescaping; a second pass removes it."""
        result = html_to_text("&lt;script&gt;alert(1)&lt;/script&gt;")
        assert "<script>" not in result

    def test_collapses_whitespace_including_nbsp(self) -> None:
        # NBSP written as an escape rather than a literal so the intent is
        # visible. Feeds emit them constantly; they must collapse like spaces.
        nbsp = "\u00a0"
        assert html_to_text(f"a {nbsp} b\t{nbsp}\tc") == "a b c"

    def test_truncates_to_max_chars(self) -> None:
        result = html_to_text("x" * 500, max_chars=100)
        assert len(result) == 101  # 100 chars + ellipsis
        assert result.endswith("…")

    def test_handles_none_and_empty(self) -> None:
        assert html_to_text(None) == ""
        assert html_to_text("") == ""

    def test_handles_malformed_markup(self) -> None:
        assert "text" in html_to_text("<p><b>text</i></p><<>>")


class TestMarkdownEscaping:
    def test_escapes_inline_specials(self) -> None:
        assert md_escape_inline("Title [with] *stars*") == "Title \\[with\\] \\*stars\\*"

    def test_collapses_newlines_so_headings_cannot_break(self) -> None:
        """A title with a newline would otherwise split a heading or table row."""
        assert "\n" not in md_escape_inline("Line one\nLine two")

    def test_escapes_heading_and_pipe_characters(self) -> None:
        escaped = md_escape_inline("# Not a heading | not a cell")
        assert escaped.startswith("\\#")
        assert "\\|" in escaped

    def test_table_cell_escapes_pipes(self) -> None:
        assert "\\|" in md_escape_table_cell("a|b")

    def test_bullet_strips_leading_marker(self) -> None:
        assert sanitize_bullet("- already a bullet") == "already a bullet"
        assert sanitize_bullet("1. numbered") == "numbered"


class TestSanitizeMdBlock:
    def test_preserves_markdown_formatting(self) -> None:
        result = sanitize_md_block("Some **bold** and a list:\n\n- one\n- two")
        assert "**bold**" in result
        assert "- one" in result

    def test_strips_html_from_llm_output(self) -> None:
        assert "<img" not in sanitize_md_block("text <img src=x onerror=alert(1)>")

    def test_removes_frontmatter_delimiters(self) -> None:
        """A '---' line could terminate the digest's own YAML frontmatter."""
        result = sanitize_md_block("intro\n---\ntype: evil\n---\nrest")
        assert "\n---\n" not in result
        assert "intro" in result

    def test_demotes_headings_so_document_structure_is_fixed(self) -> None:
        """A model emitting '# Top' must not outrank the digest's own headings."""
        result = sanitize_md_block("# Injected top-level\n## Also this")
        headings = [line for line in result.splitlines() if line.lstrip().startswith("#")]
        assert len(headings) == 2
        # Every heading is h4 — none can compete with the digest's h1/h2/h3.
        assert all(line.startswith("#### ") for line in headings)

    def test_truncates_runaway_output(self) -> None:
        result = sanitize_md_block("y" * 20_000, max_chars=500)
        assert len(result) <= 501


class TestSlugify:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Risky Business #789", "risky-business-789"),
            ("Åland & Ösund", "aland-osund"),
            ("  spaces  ", "spaces"),
            ("../../etc/passwd", "etc-passwd"),
            ("!!!", "untitled"),
            ("", "untitled"),
        ],
    )
    def test_produces_safe_ascii_slugs(self, raw: str, expected: str) -> None:
        assert slugify(raw) == expected

    def test_path_traversal_cannot_survive(self) -> None:
        slug = slugify("../../../root/.ssh/authorized_keys")
        assert "/" not in slug
        assert ".." not in slug

    def test_respects_max_length(self) -> None:
        assert len(slugify("a" * 200, max_len=20)) <= 20


class TestSafeUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd",
            "https://example.com/a b",
            'https://example.com/")',
            None,
            "",
        ],
    )
    def test_rejects_unsafe_urls(self, url: str | None) -> None:
        assert safe_url(url) is None

    @pytest.mark.parametrize(
        "url",
        ["https://example.com/ep1", "http://example.com/a?b=c&d=e"],
    )
    def test_allows_plain_http_urls(self, url: str) -> None:
        assert safe_url(url) == url


class TestLlmOutputIsSanitizedAtValidation:
    """Sanitisation is enforced by the model, so no caller can forget it."""

    def test_summary_html_is_stripped_on_construction(self) -> None:
        result = Tier1Result(
            relevance_score=5,
            summary_md="<script>bad()</script>Real summary",
            why_it_matters="Fine <b>here</b>",
            key_takeaways=["- one", "", "  "],
            entities=["Modbus", "modbus", "MODBUS", ""],
        )
        assert "bad()" not in result.summary_md
        assert "Real summary" in result.summary_md
        assert "<b>" not in result.why_it_matters
        # Blank bullets dropped, list marker removed.
        assert result.key_takeaways == ["one"]
        # Entities deduplicated case-insensitively, blanks dropped.
        assert result.entities == ["Modbus"]

    def test_scores_outside_range_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            Tier1Result(relevance_score=11)
        with pytest.raises(ValueError):
            Tier1Result(relevance_score=-1)
