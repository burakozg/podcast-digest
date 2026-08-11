"""Push notifications for exceptional episodes (roadmap E4).

Deliberately rare: only an episode scoring at or above ``min_score`` (default 9)
triggers a push, and digest availability never does. A notification that arrives
weekly stops being read; one that arrives when something genuinely matters keeps
its meaning.

Delivery targets ntfy, which needs no client library — a POST with headers. A
failure here is logged and swallowed: a missed notification must never fail a
pipeline run or lose a summary that is already safely stored.
"""

from __future__ import annotations

import unicodedata
from typing import Any

import httpx

from .config import NotificationConfig
from .logging_setup import get_logger
from .sanitize import md_escape_inline
from .utils import iso_now

log = get_logger(__name__)

#: ntfy caps header values; keep the title short and single-line.
MAX_TITLE_CHARS = 120
MAX_BODY_CHARS = 900


class Notifier:
    """Sends at-most-one push per exceptional episode."""

    def __init__(
        self, cfg: NotificationConfig, client: httpx.AsyncClient, token: str | None = None
    ) -> None:
        self._cfg = cfg
        self._client = client
        self._token = token

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled and self._cfg.ntfy_url and self._cfg.topic)

    def should_notify(self, score: int) -> bool:
        return self.enabled and score >= self._cfg.min_score

    async def notify_episode(self, episode: dict[str, Any]) -> bool:
        """Push one episode. Returns True when delivered.

        Never raises: the caller has already persisted the summary, and a dead
        notification endpoint is not a reason to fail the run.
        """
        tier1 = episode.get("tier1") or {}
        score = int(tier1.get("relevance_score") or 0)
        if not self.should_notify(score):
            return False

        show = str(episode.get("podcast_name") or episode.get("podcast_slug") or "")
        title = md_escape_inline(
            f"{score}/10 · {show} — {episode.get('title') or 'Untitled'}",
            max_chars=MAX_TITLE_CHARS,
        )
        why = str(tier1.get("why_it_matters") or "").strip()
        takeaways = [str(b) for b in (tier1.get("key_takeaways") or [])][:2]
        body_parts = [why, *(f"• {b}" for b in takeaways)]
        body = "\n".join(part for part in body_parts if part)[:MAX_BODY_CHARS]

        headers = {
            # Header values must be ASCII (httpx); ntfy reads the title from here.
            "Title": _header_safe(title),
            "Priority": self._cfg.priority,
            "Tags": ",".join(self._cfg.tags) if self._cfg.tags else "",
        }
        if link := episode.get("link"):
            headers["Click"] = str(link)
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        headers = {k: v for k, v in headers.items() if v}

        url = f"{str(self._cfg.ntfy_url).rstrip('/')}/{self._cfg.topic}"
        try:
            response = await self._client.post(
                url, content=body.encode("utf-8"), headers=headers, timeout=15.0
            )
            response.raise_for_status()
        except Exception as exc:
            log.warning(
                "notify.failed",
                episode_id=episode.get("_id"),
                error=str(exc)[:300],
                url=url,
            )
            return False

        log.info(
            "notify.sent",
            episode_id=episode.get("_id"),
            score=score,
            topic=self._cfg.topic,
            at=iso_now(),
        )
        return True


#: Separators worth keeping as their ASCII equivalent rather than dropping.
#: Written as escapes: these glyphs are indistinguishable from a hyphen in most
#: editor fonts, which makes a literal here easy to misread or mistype.
_SEPARATOR_MAP = str.maketrans(
    {
        "\u00b7": "-",  # middle dot
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2026": "...",  # horizontal ellipsis
    }
)


def _header_safe(value: str) -> str:
    """Reduce a title to ASCII for an HTTP header.

    httpx encodes header values as ASCII, so a middle dot or an accented letter
    in an episode title raises UnicodeEncodeError and loses the notification.
    Transliterate instead: "Ångström" becomes "Angstrom" rather than vanishing.
    """
    normalized = unicodedata.normalize("NFKD", value.translate(_SEPARATOR_MAP))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.split()) or "New podcast summary"
