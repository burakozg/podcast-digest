"""Static checks on the console pages.

These exist because of a bug that reached the browser: a mangled template literal
in settings.html meant the whole inline script failed to parse, so the page
rendered completely blank. Every existing test passed — they checked that the
page was *served* and contained no external references, never that its JavaScript
was syntactically valid.

A page whose script does not parse is a page that does nothing, which is worse
than one that errors, because there is no message anywhere.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from console_scan import is_escaped, page_functions, scan

from podcast_agent.api.pages import NAV_ITEMS, NAV_PLACEHOLDER, STATIC

PAGES = sorted(STATIC.glob("*.html"))

#: Node is used only as a JavaScript parser. Where it is unavailable the check
#: skips rather than silently passing — a green run must not imply it ran.
NODE = shutil.which("node")


def inline_script(path: Path) -> str:
    match = re.search(r"<script>(.*?)</script>", path.read_text(encoding="utf-8"), re.S)
    return match.group(1) if match else ""


def test_pages_were_found() -> None:
    """Guards the glob: an empty list would make every check below vacuous."""
    assert len(PAGES) >= 5
    assert {p.name for p in PAGES} >= {
        "admin.html",
        "episodes.html",
        "podcasts.html",
        "backfill.html",
        "settings.html",
    }


@pytest.mark.skipif(NODE is None, reason="node is not available to parse JavaScript")
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_inline_javascript_parses(page: Path, tmp_path: Path) -> None:
    script = inline_script(page)
    assert script.strip(), f"{page.name} has no inline script"

    scratch = tmp_path / f"{page.stem}.js"
    scratch.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [str(NODE), "--check", str(scratch)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{page.name} inline script does not parse — the page would render blank:\n"
        f"{result.stderr.strip()[:800]}"
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_carries_the_nav_placeholder(page: Path) -> None:
    """Without it the page is served with no navigation at all."""
    assert NAV_PLACEHOLDER in page.read_text(encoding="utf-8")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_external_resources(page: Path) -> None:
    """This service is LAN-only and has no business fetching a CDN."""
    html = page.read_text(encoding="utf-8")
    external = re.findall(r'(?:src|href)\s*=\s*["\']https?://|@import|url\(\s*https?://', html)
    assert external == [], f"{page.name} references {external}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_do_not_define_their_own_nav_styles(page: Path) -> None:
    """The nav ships its own styles with its markup, from one definition.

    A page keeping a stale copy is how the bar drifts out of alignment between
    pages, which is exactly what injecting it was meant to prevent.
    """
    html = page.read_text(encoding="utf-8")
    head = html[: html.index("</head>")]
    assert ".mainnav" not in head, f"{page.name} still styles .mainnav itself"


def test_nav_markup_is_self_contained() -> None:
    """The injected fragment carries its own styling, so a page needs no setup."""
    from podcast_agent.api.pages import render_nav

    nav = render_nav("/admin")
    assert "<style>" in nav
    assert ".mainnav" in nav
    for path, label in NAV_ITEMS:
        assert path in nav
        assert label in nav


def visible_text(html: str) -> str:
    """Text a reader actually sees: no styles, no scripts, no attributes."""
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", html)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_the_console_calls_them_podcasts_not_shows(page: Path) -> None:
    """One word for one thing.

    The console had both, sometimes in the same table — a "Podcasts" page whose
    heading said "Shows". Only visible text is checked: the CSS class `.show`
    and JS identifiers are code, and renaming those would be churn no reader
    benefits from.
    """
    found = re.findall(r"\bshows?\b", visible_text(page.read_text(encoding="utf-8")), re.I)
    assert found == [], f"{page.name} still says {set(found)} in visible text"


@pytest.mark.parametrize("route", [p for p, _ in NAV_ITEMS])
def test_console_pages_are_not_cacheable(route: str, tmp_path: Path) -> None:
    """A cached console silently hides every change made to it.

    These pages carry their own JavaScript and nothing in the URL changes when
    they are rebuilt, so a browser holding yesterday's copy shows a console
    missing controls that exist — which surfaces as "where is the button?"
    rather than as a visible failure.
    """
    from fastapi.testclient import TestClient
    from helpers import FakeLLM, make_settings

    from podcast_agent.db import MemoryStore
    from podcast_agent.main import build_app

    with TestClient(build_app(make_settings(tmp_path), store=MemoryStore(), llm=FakeLLM())) as c:
        response = c.get(route)
    assert response.status_code == 200
    assert "no-store" in response.headers.get("cache-control", "").lower(), (
        f"{route} may be cached by the browser"
    )


#: Internal names for pipeline stages, and the words used instead. The console
#: is read by the person who owns the podcasts, not by someone holding the
#: source, so a stage is named for what it does.
JARGON = {
    r"[Tt]ier[-_]?[01]": "triage / summarise",
    r"\bASR\b": "local transcription, transcribe locally",
    r"\bDispatch(ed)?\b": "routing / routed",
}


def prose(html: str) -> str:
    """Visible text with `<code>` spans removed.

    A literal config key or value — `tier0_only`, `llm.tiers.tier0`, the `asr`
    section — is the honest thing to print when the page is telling you what to
    type. Marking it up as code is what distinguishes that from prose, so the
    rule is: inside <code>, spell it as it really is; outside, use words.
    """
    html = re.sub(r"<code\b.*?</code>", " ", html, flags=re.S)
    return visible_text(html)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_console_prefers_plain_words_to_stage_numbers(page: Path) -> None:
    """A reader should not need the source to know what a stage does.

    "Tier-0" and "ASR" name an implementation; "triage" and "local
    transcription" name the work. Checked case-insensitively, because
    `tier0_only` in a sentence is the same jargon as "Tier-0" and slipped
    through a pattern that only matched the capitalised spelling.
    """
    text = prose(page.read_text(encoding="utf-8"))

    for pattern, instead in JARGON.items():
        found = re.findall(pattern, text)
        assert not found, f"{page.name} says {set(found)}; prefer {instead}"


def test_podcasts_are_listed_in_reading_order() -> None:
    """Enabled first, then priority, then name.

    Exercised as JavaScript because that is where the sort lives — asserting
    the source contains a comparator would pass on a comparator that sorts
    wrongly.
    """
    if NODE is None:  # pragma: no cover - environment dependent
        pytest.skip("node is not available to run JavaScript")

    script = inline_script(STATIC / "podcasts.html")
    # Just the comparator and its table. Slicing a prefix of the file instead
    # cuts through the enclosing IIFE and will not parse.
    start = script.index("const PRIORITY_ORDER")
    end = script.index("function render()")
    harness = (
        script[start:end]
        + """
        const rows = [
          { slug: "z", name: "Zebra", enabled: true,  priority: "low"  },
          { slug: "a", name: "Alpha", enabled: false, priority: "high" },
          { slug: "m", name: "Middle", enabled: true, priority: "high" },
          { slug: "b", name: "Beta",  enabled: true,  priority: "high" },
          { slug: "c", name: "Gamma", enabled: true,  priority: "med"  },
        ];
        console.log(inReadingOrder(rows).map((p) => p.name).join(","));
        """
    )
    scratch = Path(__import__("tempfile").mkdtemp()) / "order.js"
    scratch.write_text(harness, encoding="utf-8")
    result = subprocess.run([str(NODE), str(scratch)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-400:]
    # Beta and Middle are both enabled+high, so they fall back to name order;
    # Zebra is enabled but low; Alpha is disabled and sinks despite being high.
    assert result.stdout.strip() == "Beta,Middle,Gamma,Zebra,Alpha"


class TestValuesReachTheMarkupEscaped:
    """The console renders API JSON into HTML with its own JavaScript.

    Digest bodies are escaped server-side and re-checked against an allowlist,
    but nothing was watching the other direction: a page building a row with
    ``innerHTML`` and an unescaped episode title would run whatever a podcast
    put in that title, in a tab holding the admin key.

    The audit that established this baseline found the pages already escaping
    consistently; the rule is here so that stays true. What it enforces is that
    every value interpolated into markup is escaped, URL-encoded, coerced to a
    number, or is markup this page built itself — the last case being scanned in
    its own right, so nothing gets in unexamined.
    """

    def test_every_page_interpolates_escaped_values_only(self) -> None:
        offenders: list[str] = []
        for page in PAGES:
            source = page.read_text(encoding="utf-8")
            local = frozenset(page_functions(source))
            offenders += [
                str(found)
                for found in scan(page.name, source)
                if not is_escaped(found.expr, local_functions=local)
            ]
        assert not offenders, "unescaped values reaching markup:\n  " + "\n  ".join(offenders)

    def test_the_scanner_finds_an_unescaped_value(self) -> None:
        """The rule above is worthless if the scanner cannot see a bad page."""
        bad = "const row = (e) => `<td>${e.title}</td>`;"
        found = scan("bad.html", bad)
        assert [f.expr for f in found] == ["e.title"]
        assert not is_escaped("e.title")

    def test_the_scanner_accepts_an_escaped_value(self) -> None:
        found = scan("good.html", "const row = (e) => `<td>${esc(e.title)}</td>`;")
        assert all(is_escaped(f.expr) for f in found)

    def test_a_value_nested_inside_a_conditional_is_still_seen(self) -> None:
        """Where a real one would hide: markup built in a ternary branch."""
        source = 'const row = (e) => `<td>${e.ok ? `<b>${e.title}</b>` : ""}</td>`;'
        assert "e.title" in [f.expr for f in scan("nested.html", source)]

    def test_every_page_defines_the_escape_helper(self) -> None:
        """A page without one has no way to comply, and the scanner would pass
        it for lack of anything to flag."""
        for page in PAGES:
            source = page.read_text(encoding="utf-8")
            assert re.search(r"\b(function esc\(|const esc\s*=)", source), page.name


class TestTheDeploymentStackHoldsItsShape:
    """docker-compose.yml is not exercised by anything else here.

    Only the two things worth pinning are checked — that the file parses, and
    that the database stays off the LAN. A compose linter is not the job.
    """

    COMPOSE = Path(__file__).parent.parent / "docker-compose.yml"

    def _stack(self) -> dict:
        import yaml

        loaded = yaml.safe_load(self.COMPOSE.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return loaded

    def test_the_database_publishes_no_port(self) -> None:
        """The one property the network layout exists for.

        Whoever can write to this database can change where model work is sent
        (`llm.tiers.*.api_base`), so it must not be reachable from the LAN at
        all — the admin key is not what is guarding it.
        """
        couch = self._stack()["services"]["couchdb-podcast"]
        assert "ports" not in couch

    def test_the_agent_publishes_its_port(self) -> None:
        """Bridge networking rather than macvlan, so one file serves every host."""
        agent = self._stack()["services"]["podcast-agent"]
        assert any(str(p).endswith(":8080") for p in agent["ports"])

    def test_the_local_model_service_is_opt_in(self) -> None:
        """A machine running Ollama natively — for a GPU a container cannot
        reach — must not have a second copy started underneath it."""
        assert self._stack()["services"]["ollama"]["profiles"] == ["ollama"]


@pytest.mark.skipif(NODE is None, reason="node is not available to run JavaScript")
def test_a_finished_archive_episode_is_not_labelled_queued() -> None:
    """Historical intake indexes some episodes rather than summarising them.

    That decision is taken after triage, so the stored route still reads
    ESCALATE. Reading the route alone described 280 finished archive episodes
    as "queued" — work pending, on episodes an archive file already lists and
    that nothing will ever pick up.
    """
    script = inline_script(STATIC / "episodes.html")
    start = script.index("function whyNoSummary")
    end = script.index("async function openEpisode")
    harness = (
        script[start:end]
        + """
        const cases = [
          // The shape that was mislabelled: downgraded by historical intake.
          { tier0: { route: "ESCALATE" }, status: "PUBLISHED",
            digest_id: "archive:show:2025-12",
            indexed_only: "no transcript published, and local transcription is off" },
          // Same, from before the reason was recorded on the document.
          { tier0: { route: "ESCALATE" }, status: "PUBLISHED",
            digest_id: "archive:show:2025-12" },
          // Unchanged behaviour for the routine grey zone.
          { tier0: { route: "DIGEST_DIRECT" }, status: "PUBLISHED" },
          // Still genuinely queued: routine, escalated, not yet summarised.
          { tier0: { route: "ESCALATE" }, status: "AWAITING_TRANSCRIPT" },
        ];
        console.log(JSON.stringify(cases.map(whyNoSummary)));
        """
    )
    scratch = Path(__import__("tempfile").mkdtemp()) / "why.js"
    scratch.write_text(harness, encoding="utf-8")
    result = subprocess.run([str(NODE), str(scratch)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-400:]
    labels = json.loads(result.stdout)

    assert labels[0].startswith("indexed only")
    assert "local transcription is off" in labels[0], "the reason should reach the reader"
    assert labels[1] == "indexed only (archive)", "older episodes still read correctly"
    assert labels[2] == "indexed only (grey zone)"
    assert labels[3] == "queued", "a routine episode still awaiting work is queued"


@pytest.mark.skipif(NODE is None, reason="node is not available to run JavaScript")
def test_opening_an_episode_does_not_reference_a_missing_variable() -> None:
    """`node --check` proves a page parses, not that it runs.

    Renaming a variable and missing one use of it is a ReferenceError, which
    parses perfectly and only fails when a person opens the drawer. It reached
    the browser exactly that way: "Can't find variable: published", rendered in
    red inside the panel because the handler catches its own errors — which is
    what kept it from being a blank page, and also what would let a test
    watching only for exceptions pass.

    So this runs the real function against fixtures and reads what it renders.
    """
    script = inline_script(STATIC / "episodes.html")
    body = script[
        script.index("async function openEpisode") : script.index("function renderSignals")
    ]

    harness = (
        """
      // A DOM stand-in that accepts any read or write and remembers the panel.
      const panel = {};
      const $ = (id) => (panel[id] = panel[id] || {});
      const esc = (v) => String(v ?? "");
      const renderEpisodes = () => {};
      const renderSignals = () => {};
      let selected = null;
      let FIXTURE = {};
      const api = async () => FIXTURE;
    """
        + body
        + """
      const cases = {
        "archive, listed without a summary": {
          _id: "e1", title: "T", status: "PUBLISHED", podcast_slug: "s",
          digest_id: "archive:s:2026-02", archive_month: "2026-02",
          tier0_full: { relevance_guess: 7, confidence: 9, route: "ESCALATE" },
        },
        "weekly digest, summarised": {
          _id: "e2", title: "T", status: "PUBLISHED", podcast_slug: "s",
          digest_id: "digest:2026-W31", has_summary: true,
          tier1_full: { summary_md: "text", relevance_score: 8 },
        },
        "published with no claim recorded": {
          _id: "e3", title: "T", status: "PUBLISHED", podcast_slug: "s",
        },
        "still awaiting a transcript": {
          _id: "e4", title: "T", status: "AWAITING_TRANSCRIPT", podcast_slug: "s",
          tier0_full: { relevance_guess: 9, confidence: 10, route: "ESCALATE" },
        },
      };

      (async () => {
        const broken = [];
        for (const [name, fixture] of Object.entries(cases)) {
          FIXTURE = fixture;
          await openEpisode(fixture._id);
          const rendered = String(panel.dBody && panel.dBody.innerHTML || "");
          // The handler catches its own errors and renders the message, so a
          // crash shows up as content rather than as a thrown exception.
          if (/is not defined|Cannot read|undefined is not/.test(rendered)) {
            broken.push(name + ": " + rendered.slice(0, 120));
          }
        }
        console.log(JSON.stringify(broken));
      })();
    """
    )
    scratch = Path(__import__("tempfile").mkdtemp()) / "drawer.js"
    scratch.write_text(harness, encoding="utf-8")
    result = subprocess.run([str(NODE), str(scratch)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-500:]
    broken = json.loads(result.stdout)
    assert not broken, "the drawer crashed while rendering:\n  " + "\n  ".join(broken)


@pytest.mark.skipif(NODE is None, reason="node is not available to run JavaScript")
def test_stage_names_from_the_api_are_shown_as_words() -> None:
    """The vocabulary rule has to survive values the page did not write.

    `test_console_prefers_plain_words_to_stage_numbers` reads the static prose,
    so it cannot see a label that arrives from the API at render time — which is
    how `tier0` and `tier1` came to be printed as the row headings in the model
    work table, on a page whose own text obeys the rule everywhere else.
    """
    script = inline_script(STATIC / "logs.html")
    body = script[script.index("const TIER_LABELS") : script.index("function renderASR")]
    harness = (
        "const esc = (v) => String(v ?? '');\n"
        + body
        + """
        const rendered = llmGroup("By tier", {
          tier0: { calls: 12, input_tokens: 1, output_tokens: 2, cost_usd: 0.1 },
          tier1: { calls: 3,  input_tokens: 1, output_tokens: 2, cost_usd: 0.2 },
        });
        // What a reader sees, with the code spans that may legitimately carry
        // the config key removed — the same rule the prose test applies.
        console.log(JSON.stringify({
          all: rendered,
          prose: rendered.replace(/<code\\b[\\s\\S]*?<\\/code>/g, " ").replace(/<[^>]+>/g, " "),
        }));
        """
    )
    scratch = Path(__import__("tempfile").mkdtemp()) / "tiers.js"
    scratch.write_text(harness, encoding="utf-8")
    result = subprocess.run([str(NODE), str(scratch)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-400:]
    out = json.loads(result.stdout)

    assert "Triage" in out["prose"]
    assert "Summarise" in out["prose"]
    assert not re.search(r"tier[01]", out["prose"]), f"stage number in prose: {out['prose']}"
    # The key stays reachable, so a row can be tied back to the setting.
    assert "<code>tier0</code>" in out["all"]


@pytest.mark.skipif(NODE is None, reason="node is not available to run JavaScript")
def test_episode_runtime_reads_as_a_length_of_time() -> None:
    """The feed gives seconds; a reader is deciding whether to spend an evening."""
    script = inline_script(STATIC / "episodes.html")
    body = script[script.index("function runtime(") : script.index("function whyNoSummary")]
    harness = (
        body
        + """
        console.log(JSON.stringify([59, 60, 90, 2040, 3540, 3600, 5640, 8820].map(
          (s) => runtime(s * 1))));
        """
    )
    scratch = Path(__import__("tempfile").mkdtemp()) / "runtime.js"
    scratch.write_text(harness, encoding="utf-8")
    result = subprocess.run([str(NODE), str(scratch)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-400:]

    assert json.loads(result.stdout) == [
        "1 min",  # rounded, not truncated to 0
        "1 min",
        "2 min",
        "34 min",  # the median episode here
        "59 min",
        "1h 00m",  # padded, so it does not read as "1h 0m"
        "1h 34m",  # the longest episode here
        "2h 27m",
    ]


def test_the_episode_table_body_matches_its_header() -> None:
    """A row with more cells than headings silently shifts every column right.

    Adding the runtime column meant touching three places — the header, the row,
    and the empty-state colspan — and getting two of the three right looks fine
    until the table has rows, or has none.
    """
    source = (STATIC / "episodes.html").read_text(encoding="utf-8")

    start = source.index('<th title="Starred">')
    header = source[start : source.index("</tr>", start)]
    headings = len(re.findall(r"<th\b", header))

    row_start = source.index('return `<tr class="ep')
    row = source[row_start : source.index("</tr>`;", row_start)]
    # The star cell is assembled above the template and interpolated in.
    cells = len(re.findall(r"<td\b", row)) + 1

    empty = int(re.search(r'colspan="(\d+)" class="sub">No episodes match', source).group(1))

    assert cells == headings, f"{cells} cells against {headings} headings"
    assert empty == headings, f"empty state spans {empty} of {headings} columns"
