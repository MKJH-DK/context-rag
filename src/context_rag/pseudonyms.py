"""Pseudonym query expansion helpers."""

from __future__ import annotations

import re
from pathlib import Path


DEFAULT_PSEUDONYMS_PATH = Path("pseudonyms.yaml")


def expand_query(query: str, mappings: dict[str, str]) -> str:
    """Replace real names in a query with configured pseudonym placeholders."""
    if not query or not mappings:
        return query

    expanded = query
    ordered = sorted(
        ((str(source), str(target)) for source, target in mappings.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for source, target in ordered:
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        expanded = pattern.sub(target, expanded)
    return expanded


def load_mappings(path: Path | str = DEFAULT_PSEUDONYMS_PATH) -> dict[str, str]:
    """Load the mappings section from pseudonyms.yaml, ignoring other sections."""
    source = Path(path)
    if not source.exists():
        return {}

    mappings: dict[str, str] = {}
    in_mappings = False
    mapping_indent = 0
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent == 0:
            key = stripped.split(":", 1)[0].strip()
            in_mappings = key == "mappings"
            mapping_indent = indent
            continue
        if not in_mappings or indent <= mapping_indent:
            continue
        if ":" not in stripped:
            continue

        raw_key, raw_value = stripped.split(":", 1)
        key = _unquote(raw_key.strip())
        value = _unquote(_strip_inline_comment(raw_value).strip())
        if key and value:
            mappings[key] = value
    return mappings


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(value):
        if char in "'\"" and (idx == 0 or value[idx - 1] != "\\"):
            quote = None if quote == char else char
        elif char == "#" and quote is None and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value
