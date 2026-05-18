"""Structure-aware markdown chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    """A source-backed retrieval chunk."""

    id: str
    source: str
    heading_path: tuple[str, ...]
    text: str
    start_line: int
    end_line: int

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
    if len("\n".join(text_lines)) <= max_chars:
        return [
            _chunk(
                source=source,
                heading_path=heading_path,
                text="\n".join(text_lines).strip(),
                start_line=start_line,
                end_line=start_line + len(text_lines) - 1,
            )
        ]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_start = start_line
    for offset, line in enumerate(text_lines):
        next_text = "\n".join([*buffer, line]).strip()
        if buffer and len(next_text) > max_chars and not line.startswith("#"):
            chunks.append(
                _chunk(
                    source=source,
                    heading_path=heading_path,
                    text="\n".join(buffer).strip(),
                    start_line=buffer_start,
                    end_line=start_line + offset - 1,
                )
            )
            buffer, buffer_start = _overlap_lines(buffer, overlap, start_line + offset)
        buffer.append(line)

    if "\n".join(buffer).strip():
        chunks.append(
            _chunk(
                source=source,
                heading_path=heading_path,
                text="\n".join(buffer).strip(),
                start_line=buffer_start,
                end_line=start_line + len(text_lines) - 1,
            )
        )
    return chunks


def _overlap_lines(lines: list[str], overlap: int, next_line_no: int) -> tuple[list[str], int]:
    if overlap == 0:
        return [], next_line_no

    selected: list[str] = []
    size = 0
    for line in reversed(lines):
        selected.insert(0, line)
        size += len(line) + 1
        if size >= overlap:
            break
    return selected, next_line_no - len(selected)


def _heading_path(headings: dict[int, str]) -> tuple[str, ...]:
    return tuple(headings[level] for level in sorted(headings) if level <= 3)


def _chunk(
    *,
    source: Path,
    heading_path: tuple[str, ...],
    text: str,
    start_line: int,
    end_line: int,
) -> Chunk:
    seed = f"{source.as_posix()}:{start_line}:{end_line}:{text}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return Chunk(
        id=f"chunk:{digest}",
        source=str(source),
        heading_path=heading_path,
        text=text,
        start_line=start_line,
        end_line=end_line,
    )
