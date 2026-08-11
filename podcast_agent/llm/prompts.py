"""Versioned prompt loading (§7).

Prompts live in ``podcast_agent/prompts/<name>_v<N>.md`` as two sections:

    ## SYSTEM
    ...
    ## USER
    ...

The version string is part of the filename and is logged with every call, so a
prompt change is always attributable. Bump the file (``tier0_v2.md``) rather than
editing a shipped version in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from ..config import InterestItem

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

_SECTION_SPLIT = re.compile(r"^##\s+(SYSTEM|USER)\s*$", re.MULTILINE)

#: autoescape stays off — these render plain text for an LLM, not HTML. Untrusted
#: content is sanitised before it reaches here and is fenced in the prompt itself.
_env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)  # noqa: S701


class PromptError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    version: str
    system: str
    user: str

    @property
    def versioned_name(self) -> str:
        return f"{self.name}_{self.version}"

    def render(self, **context: object) -> tuple[str, str]:
        """Render (system, user). Missing template variables raise, never blank out."""
        try:
            return (
                _env.from_string(self.system).render(**context).strip(),
                _env.from_string(self.user).render(**context).strip(),
            )
        except Exception as exc:
            raise PromptError(f"rendering prompt {self.versioned_name}: {exc}") from exc


def _parse(text: str, name: str, version: str) -> Prompt:
    parts = _SECTION_SPLIT.split(text)
    # split() yields [preamble, 'SYSTEM', body, 'USER', body, ...]
    sections: dict[str, str] = {}
    for label, body in zip(parts[1::2], parts[2::2], strict=True):
        sections[label] = body.strip()
    missing = {"SYSTEM", "USER"} - sections.keys()
    if missing:
        raise PromptError(f"prompt {name}_{version} is missing section(s): {sorted(missing)}")
    return Prompt(name=name, version=version, system=sections["SYSTEM"], user=sections["USER"])


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str = "v1") -> Prompt:
    path = PROMPT_DIR / f"{name}_{version}.md"
    if not path.exists():
        available = sorted(p.name for p in PROMPT_DIR.glob("*.md"))
        raise PromptError(f"no prompt file {path.name!r} (available: {available})")
    return _parse(path.read_text(encoding="utf-8"), name, version)


def format_interest_profile(profile: list[InterestItem]) -> str:
    """Render the interest profile for prompt injection (never hardcoded, §7)."""
    lines = [
        f"- `{item.key}` — **{item.label}** (weight {item.weight}/10): {item.description}"
        for item in sorted(profile, key=lambda i: -i.weight)
    ]
    return "\n".join(lines)
