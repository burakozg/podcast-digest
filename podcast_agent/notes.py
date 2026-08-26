"""One topic note, many writers.

A topic note in the vault — ``99 topics/anthropic.md`` — is written by more than
one thing. This agent contributes what the podcast corpus knows about an entity;
a second application does the same for clipped articles; and the person whose
vault it is writes their own thinking at the top. All three want the *same* file,
because splitting them would put two ``anthropic.md`` in the vault, leave links
resolving to whichever Obsidian picked first, and produce a graph that quietly
lies about how many things connect.

So the file is divided by ownership rather than by author:

* everything outside a marked region belongs to whoever wrote it, and is never
  touched — that includes the human's prose and any other application's section;
* a region between ``<!-- begin:<owner> -->`` and ``<!-- end:<owner> -->`` is
  owned wholly by that writer, replaced on every run;
* frontmatter keys are namespaced per owner, so two writers can both describe the
  same entity without either having to know the other's schema.

HTML comments rather than heading text as the delimiter: a heading is something a
person may rename or another writer may pick by coincidence, and getting that
wrong means silently eating someone else's work. A comment is explicit, invisible
in reading view, and belongs to nobody by accident.

**The merge happens against what is in the vault, not against the file on disk.**
That is the whole point. Our copy under ``digest_dir`` never sees a human's edit
— nothing syncs back — so merging into it would rewrite the vault from a source
that cannot know what the vault contains. See :func:`merge_owned_section`, whose
caller passes the note as the vault currently holds it.
"""

from __future__ import annotations

import re

#: Directories under ``digest_dir`` whose contents are routed to their own vault
#: folder. Named here, in a module that imports nothing of ours, because both the
#: writers and the projection need them and importing either from the other
#: closes a cycle.
ENTITIES_DIR = "entities"
EPISODES_DIR = "episodes"

#: This application's owner tag. A second application picks its own; anything it
#: writes outside its own markers is not ours to touch, and vice versa.
OWNER = "podcast-digest"

#: Frontmatter keys this writer owns are prefixed, so two writers describing one
#: entity cannot collide on `mentions` meaning two different counts.
KEY_PREFIX = "podcasts_"


def begin_marker(owner: str = OWNER) -> str:
    return f"<!-- begin:{owner} -->"


def end_marker(owner: str = OWNER) -> str:
    return f"<!-- end:{owner} -->"


def _region(owner: str) -> re.Pattern[str]:
    return re.compile(
        rf"[ \t]*{re.escape(begin_marker(owner))}.*?{re.escape(end_marker(owner))}[ \t]*",
        re.DOTALL,
    )


def wrap(body: str, owner: str = OWNER) -> str:
    """Mark a block as owned, so a later run can replace exactly this much."""
    return f"{begin_marker(owner)}\n{body.strip()}\n{end_marker(owner)}"


_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def split_frontmatter(text: str) -> tuple[list[str], str]:
    """``(frontmatter lines, body)``. No YAML parse: these notes are line-per-key
    by construction, and a real parser would reformat a human's frontmatter as
    the price of reading it."""
    match = _FRONTMATTER.match(text)
    if not match:
        return [], text
    return match.group(1).split("\n"), text[match.end() :]


def _key(line: str) -> str:
    return line.split(":", 1)[0].strip()


def merge_frontmatter(
    existing: list[str], ours: list[str], *, prefix: str = KEY_PREFIX
) -> list[str]:
    """Replace our prefixed keys; leave every other line exactly as it was.

    Two classes of key, and the difference matters:

    * **Prefixed** (``podcasts_mentions``) are ours. Replaced every run, and
      dropped when this run no longer produces them.
    * **Unprefixed** (``type``, ``title``, ``tags``) describe the note as a
      whole, so they belong to whoever created it. We supply them when the key
      is absent and never otherwise — overwriting ``tags: [topic, ai]`` with our
      own ``tags: [topic]`` would silently drop a tag the reader added, and YAML
      would not even complain, because the last duplicate key wins.
    """
    kept: list[str] = []
    for line in existing:
        if (
            not line.strip()
            or line.startswith(prefix)
            or _key(line) in _RETIRED_KEYS
            or line.strip() == _RETIRED_TYPE
        ):
            continue
        # Drop an EXACT duplicate of a line already kept. Duplicate keys are
        # invalid YAML: Obsidian gives up on the whole block and shows raw text
        # instead of the properties panel, so the note looks broken to a reader.
        #
        # Not hypothetical — 21 topic notes in this vault acquired a second copy
        # of another writer's `security_*` block. No sequence of this function can
        # produce that (it always appends the owner's keys last, and these sat
        # either side of another writer's block), and some of the affected notes
        # had no region from the writer whose keys were duplicated. The likely
        # source is LiveSync's own line-level merge of two revisions during
        # concurrent writes, which duplicates identical lines rather than
        # collapsing them.
        #
        # Only EXACT duplicates. Two lines with the same key and different values
        # are a real disagreement between writers, and quietly picking one would
        # be inventing an answer; they are left for a human, who can at least see
        # both.
        if line in kept:
            continue
        kept.append(line)
    seen = {_key(line) for line in kept}
    owned = [line for line in ours if line.startswith(prefix)]
    seeds = [line for line in ours if not line.startswith(prefix) and _key(line) not in seen]
    return kept + seeds + owned


def merge_owned_section(existing: str | None, ours: str, *, owner: str = OWNER) -> str:
    """Our section written into ``existing``, leaving everything else alone.

    ``existing`` is the note as the vault currently holds it, or None when there
    is no note yet. ``ours`` is a complete note as we would write it fresh —
    frontmatter, a title, and one marked region.

    Three cases, in the order they are checked:

    1. **No existing note** — ours becomes the file.
    2. **A marked region of ours is present** — it is replaced in place, so the
       human's prose above it and any other writer's section below it keep their
       position on the page.
    3. **No marked region** — our section is appended. This is also how a note
       from before markers existed is adopted: the legacy ``## Mentioned in``
       block we used to write is recognised and removed first, because it is
       ours and leaving it would show the same list twice, once going stale.
    """
    if not existing or not existing.strip():
        return ours

    our_front, our_body = split_frontmatter(ours)
    our_region = _region(owner).search(our_body)
    our_section = our_region.group(0).strip() if our_region else wrap(our_body.strip(), owner)

    front, body = split_frontmatter(existing)
    merged_front = merge_frontmatter(front, our_front)

    if _region(owner).search(body):
        body = _region(owner).sub(lambda _: our_section, body, count=1)
    else:
        body = _strip_legacy(body).rstrip() + "\n\n" + our_section + "\n"

    head = "---\n" + "\n".join(merged_front) + "\n---\n" if merged_front else ""
    return head + body


#: Keys an earlier version of this writer owned outright, before the prefix
#: existed. Dropped on merge, so a note from the old format stops carrying stale
#: unnamespaced counts that now live under `podcasts_`.
#:
#: `type` is deliberately absent: it describes the note rather than our part of
#: it, so another writer may legitimately own it. Only its one legacy *value* is
#: retired, below.
_RETIRED_KEYS = frozenset({"entity", "mentions", "shows", "first_seen", "last_seen"})

#: The `type` this writer used when it was the only writer.
_RETIRED_TYPE = "type: podcast-entity"

#: The heading this writer used before ownership markers existed. Recognised so
#: the 137 notes already in the vault are adopted rather than duplicated.
_LEGACY = re.compile(r"\n##\s+Mentioned in\s*\n.*?(?=\n## |\Z)", re.DOTALL)

#: The one-line summary that sat above it.
_LEGACY_SUMMARY = re.compile(r"\n\*\d+ episodes? across \d+ shows? · [^\n]*\*\n")


def _strip_legacy(body: str) -> str:
    return _LEGACY_SUMMARY.sub("\n", _LEGACY.sub("\n", body))
