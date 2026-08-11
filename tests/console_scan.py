"""Finding the places a console page puts a value into markup.

Used by :mod:`test_console_assets` to keep the pages escaped. It is a scanner,
not a JavaScript parser: it finds template literals that build HTML and reports
every ``${...}`` inside them, recursing into nested literals so an escaped
expression inside an unescaped-looking one is judged on its own terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: An expression that cannot carry markup into the page: it is escaped, it is
#: URL-encoded for an attribute, or it is coerced to a number.
SAFE_PREFIXES = ("esc(", "encodeURIComponent(")
NUMERIC = re.compile(
    r"^(num\(|Number\(|Math\.|\(?\s*[\w.$\[\]]+\s*(\|\||\?\?)\s*0\s*\)?\.toFixed\()"
)

#: A bare local name — `star`, `blurb`, `control`. These hold markup the page
#: built a few lines earlier, and that markup is scanned in its own right, so
#: requiring esc() here would demand escaping HTML into visible tag soup.
FRAGMENT_NAME = re.compile(r"^[A-Za-z_$][\w$]*$")

_STRING = re.compile(r"'[^'\\]*'|\"[^\"\\]*\"")


@dataclass(frozen=True)
class Interpolation:
    file: str
    line: int
    expr: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}  ${{{self.expr}}}"


def _literals(src: str) -> list[tuple[int, str]]:
    """Every template literal in ``src``, as (offset, body) pairs."""
    found: list[tuple[int, str]] = []
    i = 0
    while True:
        start = src.find("`", i)
        if start == -1:
            return found
        j = start + 1
        depth = 0
        while j < len(src):
            c = src[j]
            if c == "\\":
                j += 2
                continue
            if c == "$" and src[j + 1 : j + 2] == "{":
                depth += 1
                j += 2
                continue
            if c == "}" and depth:
                depth -= 1
                j += 1
                continue
            if c == "`" and depth == 0:
                break
            j += 1
        found.append((start + 1, src[start + 1 : j]))
        i = j + 1


def _interpolations(body: str) -> list[tuple[int, str]]:
    """The ``${...}`` spans of one literal body, as (offset, expression)."""
    spans: list[tuple[int, str]] = []
    i = 0
    while True:
        at = body.find("${", i)
        if at == -1:
            return spans
        j = at + 2
        depth = 1
        while j < len(body) and depth:
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            j += 1
        spans.append((at + 2, body[at + 2 : j - 1]))
        i = j


def _is_html(body: str) -> bool:
    return "<" in body and ">" in body


_DEFINED = re.compile(
    r"(?:function\s+([A-Za-z_$][\w$]*)\s*\()|(?:const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:\([^)]*\)|[\w$]+)\s*=>)"
)


def page_functions(src: str) -> set[str]:
    """Functions the page defines itself.

    A call to one is treated like a fragment variable: whatever markup it
    returns is built from literals in this same file, and those are scanned on
    their own. Requiring esc() around it would escape the page's own HTML.
    """
    return {m.group(1) or m.group(2) for m in _DEFINED.finditer(src)}


def scan(name: str, src: str) -> list[Interpolation]:
    """Interpolations that land inside markup, innermost first.

    An expression containing its own template literal is not reported; its
    inner interpolations are reported instead, which is what actually reaches
    the page.
    """
    out: list[Interpolation] = []

    def walk(offset: int, body: str, in_html: bool) -> None:
        for rel, expr in _interpolations(body):
            absolute = offset + rel
            nested = _literals(expr)
            if nested:
                # Recurse: the outer expression is glue, the inner literals are
                # what produce markup.
                for n_off, n_body in nested:
                    walk(absolute + n_off, n_body, in_html or _is_html(n_body))
                # The non-literal parts still matter when this builds markup.
                stripped = expr
                for _, n_body in nested:
                    stripped = stripped.replace(n_body, "")
                if in_html and _carries_value(stripped):
                    out.append(Interpolation(name, _line(src, absolute), stripped.strip()))
                continue
            if in_html:
                out.append(Interpolation(name, _line(src, absolute), expr.strip()))

    def _line(source: str, offset: int) -> int:
        return source.count("\n", 0, offset) + 1

    for off, body in _literals(src):
        if _is_html(body):
            walk(off, body, True)
    return out


def _carries_value(expr: str) -> bool:
    """Whether what is left of an expression could still emit a value.

    A conditional only chooses *between* its branches, so once the branches are
    gone what remains is a condition, and a condition never reaches the page.
    """
    return not _branches_are_gone(expr)


def _branches_are_gone(expr: str) -> bool:
    body = _STRING.sub("", expr)
    for operator in ("?", "&&", "||"):
        if operator in body:
            tail = body.split(operator, 1)[1]
            if not re.search(r"[\w$]", tail.replace(":", " ")):
                return True
    return False


def is_escaped(expr: str, *, local_functions: frozenset[str] = frozenset()) -> bool:
    """Whether ``expr`` cannot introduce markup."""
    expr = expr.strip()
    if expr.startswith(SAFE_PREFIXES) or NUMERIC.match(expr):
        return True
    if FRAGMENT_NAME.match(expr):
        # A local name holding markup this page built; scanned in its own right.
        return True
    call = re.match(r"^([A-Za-z_$][\w$]*)\s*\(", expr)
    if call and call.group(1) in local_functions:
        return True
    # `xs.map(x => `<li>…`).join("")` — a list of fragments. The literals it is
    # built from were scanned; the chain itself introduces no value.
    if ".map(" in expr and ".join(" in expr:
        return True
    # A conditional whose branches are all string literals decides between two
    # fixed pieces of text — the condition never reaches the page.
    return _branches_are_gone(expr)
