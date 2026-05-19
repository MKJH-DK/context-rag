"""Structure-aware markdown chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LOC_RE = re.compile(r"^<!--\s*loc:\s*(.+?)\s*-->\s*$")


@dataclass(frozen=True)
class Chunk:
    """A source-backed retrieval chunk."""

    id: str
    source: str
    heading_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int
    src: str | None = None
    ts: str | None = None
    ts_end: str | None = None
    page: int | None = None
    page_end: int | None = None
    chapter: str | None = None
    slide: int | None = None
    slide_end: int | None = None
    sheet: str | None = None

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["heading_path"] = list(self.heading_path)
        return data


def chunk_markdown(
    path: str | Path,
    *,
    max_chars: int = 4000,
    overlap: int = 0,
) -> list[Chunk]:
    """Chunk a markdown file on H2/H3 boundaries.

    H1 headings are kept as path context. Long sections are split on paragraph
    boundaries with optional character overlap; the default overlap is zero.
    """

    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return []
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    lines = source.read_text(encoding="utf-8").splitlines()
    sections = _sections(lines)
    chunks: list[Chunk] = []
    for start_line, end_line, heading_path in sections:
        text_lines = lines[start_line - 1 : end_line]
        text = "\n".join(text_lines).strip()
        if not text:
            continue
        chunks.extend(
            _split_section(
                source=source,
                heading_path=heading_path,
                text_lines=text_lines,
                start_line=start_line,
                max_chars=max_chars,
                overlap=overlap,
            )
        )
    return chunks


def _sections(lines: list[str]) -> list[tuple[int, int, tuple[str, ...]]]:
    sections: list[tuple[int, int, tuple[str, ...]]] = []
    headings: dict[int, str] = {}
    active_start = 1
    active_path: tuple[str, ...] = ()

    for line_no, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if not match:
            continue

        level = len(match.group(1))
        title = match.group(2).strip()
        headings[level] = title
        for stale_level in range(level + 1, 7):
            headings.pop(stale_level, None)

        if level == 1 and active_start == line_no:
            active_path = _heading_path(headings)
            continue
        if level not in {2, 3}:
            continue

        if line_no > active_start:
            sections.append((active_start, line_no - 1, active_path))
        active_start = line_no
        active_path = _heading_path(headings)

    if lines:
        sections.append((active_start, len(lines), active_path))
    return sections


def _split_section(
    *,
    source: Path,
    heading_path: tuple[str, ...],
    text_lines: list[str],
    start_line: int,
    max_chars: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[tuple[str, int, dict[str, str]]] = []
    buffer_start = start_line
    current_loc: dict[str, str] = {}

    for offset, line in enumerate(text_lines):
        line_no = start_line + offset
        marker = LOC_RE.match(line)
        if marker:
            current_loc = _parse_loc_marker(marker.group(1))
            continue

        next_text = "\n".join([entry[0] for entry in [*buffer, (line, line_no, current_loc)]]).strip()
        if buffer and len(next_text) > max_chars and not line.startswith("#"):
            chunks.append(
                _chunk(
                    source=source,
                    heading_path=heading_path,
                    text="\n".join(entry[0] for entry in buffer).strip(),
                    start_line=buffer_start,
                    end_line=buffer[-1][1],
                    loc=_chunk_loc(buffer),
                )
            )
            buffer, buffer_start = _overlap_lines(buffer, overlap, start_line + offset)
        if not buffer:
            buffer_start = line_no
        buffer.append((line, line_no, current_loc.copy()))

    if "\n".join(entry[0] for entry in buffer).strip():
        chunks.append(
            _chunk(
                source=source,
                heading_path=heading_path,
                text="\n".join(entry[0] for entry in buffer).strip(),
                start_line=buffer_start,
                end_line=buffer[-1][1],
                loc=_chunk_loc(buffer),
            )
        )
    return chunks


def _overlap_lines(
    lines: list[tuple[str, int, dict[str, str]]], overlap: int, next_line_no: int
) -> tuple[list[tuple[str, int, dict[str, str]]], int]:
    if overlap == 0:
        return [], next_line_no

    selected: list[tuple[str, int, dict[str, str]]] = []
    size = 0
    for entry in reversed(lines):
        selected.insert(0, entry)
        size += len(entry[0]) + 1
        if size >= overlap:
            break
    return selected, selected[0][1] if selected else next_line_no


def _heading_path(headings: dict[int, str]) -> tuple[str, ...]:
    return tuple(headings[level] for level in sorted(headings) if level <= 3)


def _chunk_loc(lines: list[tuple[str, int, dict[str, str]]]) -> dict[str, str]:
    """Use first marker metadata and derive ranges within the first source segment."""
    first: dict[str, str] | None = None
    first_src: str | None = None
    last_values: dict[str, str] = {}

    for _, _, loc in lines:
        if not loc:
            continue
        if first is None:
            first = {key: value for key, value in loc.items() if key != "ts_end"}
            first_src = loc.get("src")
        elif loc.get("src") != first_src:
            break
        for key in ("ts", "p", "slide"):
            if loc.get(key):
                last_values[key] = loc[key]

    if first is None:
        return {}

    for key, end_key in (
        ("ts", "ts_end"),
        ("p", "page_end"),
        ("slide", "slide_end"),
    ):
        start = first.get(key)
        end = last_values.get(key)
        if start and end and _range_value_differs(key, start, end):
            first[end_key] = end
    return first


def _chunk(
    *,
    source: Path,
    heading_path: tuple[str, ...],
    text: str,
    start_line: int,
    end_line: int,
    loc: dict[str, str] | None = None,
) -> Chunk:
    seed = f"{source.as_posix()}:{start_line}:{end_line}:{text}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    loc = loc or {}
    return Chunk(
        id=f"chunk:{digest}",
        source=str(source),
        heading_path=heading_path,
        text=text,
        start_line=start_line,
        end_line=end_line,
        src=loc.get("src"),
        ts=loc.get("ts"),
        ts_end=loc.get("ts_end"),
        page=_to_int(loc.get("p")),
        page_end=_to_int(loc.get("page_end")),
        chapter=loc.get("ch"),
        slide=_to_int(loc.get("slide")),
        slide_end=_to_int(loc.get("slide_end")),
        sheet=loc.get("sheet"),
    )


def _parse_loc_marker(body: str) -> dict[str, str]:
    loc: dict[str, str] = {}
    for part in body.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"src", "ts", "ts_end", "p", "ch", "slide", "sheet"} and value:
            loc[key] = value
    return loc


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _range_value_differs(key: str, start: str, end: str) -> bool:
    if key in {"p", "slide"}:
        return _to_int(start) is not None and _to_int(start) != _to_int(end)
    return start != end
