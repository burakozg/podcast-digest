"""Console page assembly.

Every page is a standalone HTML file, which keeps each one readable and
self-contained — but a navigation bar copied into five files drifts the first
time a page is added. The nav is therefore defined once here, with its own
styling, and injected at serve time: a page needs nothing but the placeholder,
and no page can hold a stale copy of the layout.
"""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"

#: Marker each page carries where the navigation belongs.
NAV_PLACEHOLDER = "<!--NAV-->"

#: (path, label). Order is the order shown.
NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("/admin", "Operations"),
    ("/admin/digests", "Digests"),
    ("/admin/episodes", "Episodes"),
    ("/admin/podcasts", "Podcasts"),
    ("/admin/insights", "Insights"),
    ("/admin/backfill", "Historical intake"),
    ("/admin/logs", "Logs"),
    ("/admin/settings", "Settings"),
)

#: Width of the sidebar, and the matching inset applied to the page body.
NAV_WIDTH = "190px"

#: Styling ships with the markup so a page needs no setup, and so the bar cannot
#: drift out of alignment between pages. Fixed to the left edge with the body
#: inset to match; on a narrow screen it lies flat across the top instead.
NAV_STYLE = f"""<style>
  .mainnav {{
    position: fixed; inset: 0 auto 0 0; width: {NAV_WIDTH}; z-index: 40;
    display: flex; flex-direction: column; gap: .15rem;
    padding: 1rem .6rem; overflow-y: auto;
    background: var(--panel); border-right: 1px solid var(--line);
  }}
  .mainnav .brand {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .09em;
    color: var(--muted); font-weight: 650; padding: .1rem .65rem .9rem;
  }}
  .mainnav a {{
    padding: .45rem .65rem; border-radius: 7px; text-decoration: none;
    color: var(--ink); font-size: .9rem; font-weight: 500; line-height: 1.3;
  }}
  .mainnav a:hover {{ background: color-mix(in srgb, var(--accent) 12%, transparent); }}
  .mainnav a.here {{ background: var(--accent); color: var(--bg); font-weight: 600; }}
  body {{ padding-left: {NAV_WIDTH}; }}
  header {{ left: {NAV_WIDTH}; }}
  @media (max-width: 760px) {{
    .mainnav {{
      position: static; width: auto; flex-direction: row; flex-wrap: wrap;
      border-right: 0; border-bottom: 1px solid var(--line); padding: .6rem;
    }}
    .mainnav .brand {{ display: none; }}
    body {{ padding-left: 0; }}
    header {{ left: 0; }}
  }}
</style>"""


def render_nav(current: str) -> str:
    links = "".join(
        '<a href="{path}"{here}>{label}</a>'.format(
            path=path,
            here=" class='here'" if path == current else "",
            label=label,
        )
        for path, label in NAV_ITEMS
    )
    return f'{NAV_STYLE}<nav class="mainnav"><span class="brand">Digest agent</span>{links}</nav>'


def page(name: str, current: str) -> str:
    """Load a console page with its navigation injected."""
    html = (STATIC / name).read_text(encoding="utf-8")
    return html.replace(NAV_PLACEHOLDER, render_nav(current))
