"""Reading digests back for the console.

Two hazards here, neither hypothetical. `file_path` comes off a database
document, so it is input and could point anywhere the process can read. And the
digest body is assembled from LLM output that is itself downstream of podcast
descriptions and transcripts — it has been escaped for Markdown, which says
nothing about whether it is safe as HTML.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from podcast_agent.digest.read import (
    DigestUnreadable,
    read_digest,
    resolve_within,
    split_frontmatter,
)
from podcast_agent.sanitize import _MD_TAGS, md_to_safe_html

#: Mirrors the renderer's allowlist. Imported rather than restated so a tag
#: added there is covered here without the test needing an edit.
ALLOWED_TAGS = set(_MD_TAGS)

DIGEST = """---
type: podcast-digest
week: 2026-W31
episodes_scanned: 12
---

# Podcast Digest — Week 31, 2026

## Top picks

### Some Podcast — An Episode  `9/10`

**Why it matters:** because.
"""


class TestPathConfinement:
    @pytest.mark.parametrize(
        "relative",
        [
            "../secrets.env",
            "../../etc/passwd",
            "2026/../../../etc/passwd",
            "/etc/passwd",
            "2026/../../outside.md",
        ],
    )
    def test_paths_outside_the_digest_directory_are_refused(
        self, tmp_path: Path, relative: str
    ) -> None:
        with pytest.raises(DigestUnreadable):
            resolve_within(tmp_path / "digests", relative)

    def test_a_normal_relative_path_resolves(self, tmp_path: Path) -> None:
        base = tmp_path / "digests"
        assert resolve_within(base, "2026/week.md") == (base / "2026/week.md").resolve()

    def test_the_guard_is_what_read_digest_uses(self, tmp_path: Path) -> None:
        """The check must not be bypassable through the public entry point."""
        (tmp_path / "secret.md").write_text("password: hunter2")
        base = tmp_path / "digests"
        base.mkdir()
        with pytest.raises(DigestUnreadable):
            read_digest(base, "../secret.md")


class TestFrontmatter:
    def test_metadata_is_separated_from_the_body(self) -> None:
        meta, body = split_frontmatter(DIGEST)
        assert meta["week"] == "2026-W31"
        assert meta["episodes_scanned"] == 12
        assert body.startswith("# Podcast Digest")
        assert "type: podcast-digest" not in body

    def test_a_file_without_frontmatter_is_all_body(self) -> None:
        meta, body = split_frontmatter("# Just a heading\n")
        assert meta == {}
        assert body == "# Just a heading\n"

    def test_unparseable_frontmatter_does_not_lose_the_digest(self) -> None:
        """The body is what the reader came for; bad metadata is not fatal."""
        meta, body = split_frontmatter("---\n: : not yaml : :\n---\n\n# Heading\n")
        assert meta == {}
        assert "# Heading" in body


class TestReading:
    def _write(self, tmp_path: Path, text: str = DIGEST) -> Path:
        base = tmp_path / "digests"
        (base / "2026").mkdir(parents=True)
        (base / "2026" / "w31.md").write_text(text, encoding="utf-8")
        return base

    def test_returns_metadata_markdown_and_html(self, tmp_path: Path) -> None:
        result = read_digest(self._write(tmp_path), "2026/w31.md")
        assert result["frontmatter"]["week"] == "2026-W31"
        assert result["markdown"].startswith("# Podcast Digest")
        assert "<h1>" in result["html"]
        assert result["bytes"] > 0

    def test_a_missing_file_is_reported_not_crashed(self, tmp_path: Path) -> None:
        """Digests live in a directory the user owns and may prune or move."""
        base = tmp_path / "digests"
        base.mkdir()
        with pytest.raises(DigestUnreadable, match="no digest file"):
            read_digest(base, "2026/gone.md")

    def test_an_implausibly_large_file_is_refused(self, tmp_path: Path) -> None:
        base = self._write(tmp_path, "x" * (5 * 1024 * 1024))
        with pytest.raises(DigestUnreadable, match="implausibly large"):
            read_digest(base, "2026/w31.md")


class TestMarkdownRendering:
    """The console injects this HTML directly, so it must arrive inert."""

    @pytest.mark.parametrize(
        "hostile",
        [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "<iframe src='https://evil.example'></iframe>",
            "<a href='javascript:alert(1)'>click</a>",
            "[click](javascript:alert(1))",
            "[click](data:text/html;base64,PHNjcmlwdD4=)",
            "<svg onload=alert(1)>",
            "<style>body{display:none}</style>",
            "<div onclick='alert(1)'>x</div>",
            "<form action='https://evil.example'><input name=p></form>",
        ],
    )
    def test_hostile_markup_does_not_survive(self, hostile: str) -> None:
        """Asserted on parsed structure, not on substrings.

        Escaped output legitimately *contains* the string "onclick" as visible
        text — `&lt;div onclick=...&gt;` is inert prose. What must not exist is
        a live element or a live attribute, which only parsing can tell apart.
        """
        soup = BeautifulSoup(md_to_safe_html(hostile), "html.parser")

        tags = {t.name for t in soup.find_all()}
        assert tags <= ALLOWED_TAGS, f"unexpected elements {tags - ALLOWED_TAGS}"

        for tag in soup.find_all():
            for attr, value in tag.attrs.items():
                assert not attr.lower().startswith("on"), f"event handler {attr} on <{tag.name}>"
                assert attr.lower() != "style", f"style attribute on <{tag.name}>"
                if attr.lower() == "href":
                    scheme = str(value).split(":", 1)[0].lower() if ":" in str(value) else ""
                    assert scheme in ("", "http", "https", "mailto"), f"href scheme {scheme!r}"

    def test_ordinary_prose_still_renders(self) -> None:
        html = md_to_safe_html("# Heading\n\n**bold** and *italic* and `code`\n\n- one\n- two\n")
        assert "<h1>Heading</h1>" in html
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html
        assert "<code>code</code>" in html
        assert html.count("<li>") == 2

    def test_http_links_are_kept(self) -> None:
        """Digests link to episodes; dropping those would gut the page."""
        html = md_to_safe_html("[ep](https://example.com/ep)")
        assert '<a href="https://example.com/ep">ep</a>' in html

    def test_tables_render(self) -> None:
        html = md_to_safe_html("| a | b |\n|---|---|\n| 1 | 2 |\n")
        assert "<table>" in html and "<th>a</th>" in html

    def test_empty_input_is_empty_output(self) -> None:
        assert md_to_safe_html("") == ""
        assert md_to_safe_html(None) == ""

    def test_a_digest_read_from_disk_is_sanitised_too(self, tmp_path: Path) -> None:
        """End to end: the path the console actually takes."""
        base = tmp_path / "digests"
        base.mkdir()
        (base / "w.md").write_text(
            "---\nweek: 2026-W31\n---\n\n# Ok\n\n<script>alert(1)</script>\n",
            encoding="utf-8",
        )
        result = read_digest(base, "w.md")
        assert "<script" not in result["html"].lower()
        assert "<h1>Ok</h1>" in result["html"]
