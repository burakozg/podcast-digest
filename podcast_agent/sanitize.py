"""Sanitisation for untrusted feed content and untrusted LLM output (§10.2).

Everything here treats its input as hostile: RSS descriptions, scraped pages and
transcripts are attacker-controllable, and LLM free-text fields are downstream of
those. Nothing in this module trusts its argument.
"""

from __future__ import annotations

import html
import re
import unicodedata

import bleach

#: Control characters except tab/newline. Strip before storage and rendering.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Includes U+00A0 (no-break space), which feeds emit constantly.
_WHITESPACE_RUNS = re.compile("[ \t\u00a0]+")
_BLANK_LINE_RUNS = re.compile(r"\n{3,}")

#: Markdown control characters that must not break out of an inline context.
_MD_INLINE_SPECIALS = re.compile(r"([\\`*_\[\]<>|#~])")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

#: A line consisting only of dashes/equals opens YAML frontmatter or turns the
#: preceding line into a heading. Neutralised in LLM-authored Markdown blocks.
_FRONTMATTER_LINE = re.compile(r"^\s{0,3}(-{3,}|={3,}|\.{3,})\s*$", re.MULTILINE)

#: Elements whose *contents* are not prose. bleach's strip=True would keep the
#: inner text (leaving "bad()" behind from a <script>), so remove them wholesale.
_NON_PROSE_ELEMENTS = re.compile(
    r"<\s*(script|style|template|noscript|iframe)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def strip_control_chars(text: str) -> str:
    return _CONTROL_CHARS.sub("", text)


def html_to_text(raw: str | None, *, max_chars: int | None = None) -> str:
    """Convert untrusted HTML (an RSS description) to plain text.

    Tags are removed entirely rather than allowlisted — descriptions are only
    ever used as prompt input and one-line digest context, so no markup survives.
    """
    if not raw:
        return ""
    text = _NON_PROSE_ELEMENTS.sub(" ", raw)
    # bleach next (handles malformed markup safely), then unescape entities that
    # bleach leaves encoded, then a second strip in case unescaping revealed tags.
    text = bleach.clean(text, tags=set(), attributes={}, strip=True, strip_comments=True)
    text = html.unescape(text)
    text = bleach.clean(text, tags=set(), attributes={}, strip=True, strip_comments=True)
    text = html.unescape(text)
    text = strip_control_chars(text)
    text = _WHITESPACE_RUNS.sub(" ", text)
    text = _BLANK_LINE_RUNS.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def md_escape_inline(text: str | None, *, max_chars: int = 300) -> str:
    """Escape text destined for an inline Markdown context (headings, table cells).

    Newlines are collapsed: an episode title containing a newline would otherwise
    break table rows and heading structure.
    """
    if not text:
        return ""
    flat = " ".join(strip_control_chars(text).split())
    escaped = _MD_INLINE_SPECIALS.sub(r"\\\1", flat)
    if len(escaped) > max_chars:
        escaped = escaped[:max_chars].rstrip() + "…"
    return escaped


def md_escape_table_cell(text: str | None, *, max_chars: int = 120) -> str:
    """Inline escaping plus pipe neutralisation for Markdown table cells."""
    return md_escape_inline(text, max_chars=max_chars).replace("|", "\\|")


def sanitize_md_block(text: str | None, *, max_chars: int = 8000) -> str:
    """Clean an LLM-authored Markdown block for embedding in the digest.

    Markdown formatting is preserved (that is the point of the field) but HTML is
    stripped, frontmatter delimiters are defanged, and headings are demoted so a
    model cannot restructure the digest document.
    """
    if not text:
        return ""
    # No raw HTML in output — blocks embedded HTML and script content alike.
    text = _NON_PROSE_ELEMENTS.sub(" ", text)
    text = bleach.clean(text, tags=set(), attributes={}, strip=True, strip_comments=True)
    text = html.unescape(text)
    text = strip_control_chars(text)
    # A '---' line would terminate the digest's YAML frontmatter if the model
    # emitted one near the top of the file, and renders as an <hr> otherwise.
    text = _FRONTMATTER_LINE.sub("", text)
    # Demote any heading to h4 so summaries nest under the digest's own h2/h3.
    text = re.sub(r"^(\s{0,3})#{1,6}\s+", r"\1#### ", text, flags=re.MULTILINE)
    text = _BLANK_LINE_RUNS.sub("\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def sanitize_bullet(text: str | None, *, max_chars: int = 400) -> str:
    """Clean a single LLM-authored bullet: one line, no list markers of its own."""
    cleaned = md_escape_inline(text, max_chars=max_chars)
    # Strip a leading list marker the model may have included itself.
    return re.sub(r"^(\\?[-*+]|\d+\\?\.)\s+", "", cleaned).strip()


def slugify(text: str, *, max_len: int = 60, fallback: str = "untitled") -> str:
    """ASCII-only, lowercase, hyphenated slug for use in filenames (§5)."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_only).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or fallback


def safe_url(url: str | None) -> str | None:
    """Return ``url`` only if it is a plain http(s) URL safe to put in Markdown.

    Rejects javascript:/data: schemes and anything containing characters that
    would break out of a Markdown link target.
    """
    if not url:
        return None
    candidate = strip_control_chars(url).strip()
    if not candidate.lower().startswith(("http://", "https://")):
        return None
    if any(ch in candidate for ch in (" ", "(", ")", "<", ">", '"', "'", "\\")):
        return None
    return candidate


#: Prose elements a digest may render as. Anything else — script, style, form,
#: iframe, img — is dropped. No images: a digest is text, and an <img> is an
#: outbound request from a console that is meant to make none.
_MD_TAGS = frozenset(
    {
        "p",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "strong",
        "em",
        "del",
        "code",
        "pre",
        "blockquote",
        "ul",
        "ol",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "a",
    }
)

#: Only what a link needs. No `target`, no `on*`, no `style`.
_MD_ATTRS = {"a": ["href", "title"]}

#: `javascript:` and `data:` are the two that matter. markdown-it already refuses
#: them; this is the second lock on the same door.
_MD_PROTOCOLS = frozenset({"http", "https", "mailto"})


def md_to_safe_html(markdown: str | None) -> str:
    """Render Markdown to HTML safe to inject into the console.

    Digest Markdown is assembled from LLM output, which is itself downstream of
    podcast descriptions and transcripts — attacker-influenced at two removes. It
    reaches this function having been escaped for *Markdown*, which says nothing
    about whether it is safe as *HTML*.

    Two independent defences, because either alone is a single point of failure:
    the renderer runs with ``html=False``, so raw HTML in the source is escaped
    to text rather than passed through, and the result is then filtered against
    an allowlist. A future preset change that re-enables inline HTML would be
    caught by the second; a bug in the allowlist would be caught by the first.
    """
    if not markdown:
        return ""
    # Imported here: markdown-it costs ~30ms to import and only the console
    # rendering path needs it, not the pipeline.
    from markdown_it import MarkdownIt

    rendered = MarkdownIt("js-default", {"html": False, "linkify": False}).render(
        strip_control_chars(markdown)
    )
    return bleach.clean(
        rendered,
        tags=set(_MD_TAGS),
        attributes=_MD_ATTRS,
        protocols=set(_MD_PROTOCOLS),
        strip=True,
        strip_comments=True,
    )
