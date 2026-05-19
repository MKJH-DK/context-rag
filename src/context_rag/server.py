"""MCP stdio server for a markdown corpus index."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sqlite3
from typing import Any

from .embeddings import Embedder
from .pseudonyms import expand_query, load_mappings
from .query import format_citation, hybrid_search


DEFAULT_DB = Path(".context-rag/index.db")
DEFAULT_TOOL_DESCRIPTIONS = {
    "search": (
        "Search the indexed markdown corpus. Returns relevant excerpts "
        "with citations."
    ),
    "get_chunk": "Fetch the full content of a chunk by id.",
    "list_sources": "List all source files in the index.",
}


def search_index(
    db_path: str | Path,
    query: str,
    k: int = 10,
    *,
    embedder: Embedder | None = None,
    pseudonym_mappings: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Search the markdown corpus and return cited chunks."""

    active_embedder = embedder or Embedder()
    expanded_query = expand_query(query, pseudonym_mappings or {})
    return [
        _with_citation(hit)
        for hit in hybrid_search(db_path, active_embedder, expanded_query, k=k)
    ]


def get_chunk(db_path: str | Path, chunk_id: str) -> dict[str, Any] | None:
    """Return a full chunk by ID."""

    with _connect(db_path) as con:
        row = con.execute(
            """
            SELECT id, source, heading_path, text, start_line, end_line,
                   src, ts, ts_end, page, page_end, chapter,
                   slide, slide_end, sheet
            FROM chunks
            WHERE id = ?
            """,
            (chunk_id,),
        ).fetchone()
    if row is None:
        return None
    return _with_citation(_row_result(row))


def list_sources(db_path: str | Path) -> list[dict[str, Any]]:
    """List indexed markdown source files."""

    with _connect(db_path) as con:
        rows = con.execute(
            """
            SELECT source, COUNT(*) AS chunks, MIN(start_line) AS first_line,
                   MAX(end_line) AS last_line
            FROM chunks
            GROUP BY source
            ORDER BY source
            """
        ).fetchall()
    return [dict(row) for row in rows]


def serve(db_path: str | Path = DEFAULT_DB) -> None:
    """Run the MCP server over stdio."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The official mcp Python SDK is required. Install with "
            "`python -m pip install -e .`."
        ) from exc

    db = Path(db_path)
    get_chunk_fn = get_chunk
    list_sources_fn = list_sources
    descriptions = resolve_tool_descriptions()
    embedder = Embedder()
    pseudonym_mappings = load_mappings()
    mcp = FastMCP("context-rag")

    @mcp.tool(name="search", description=descriptions["search"])
    def _search_tool(query: str, k: int = 10) -> list[dict[str, Any]]:
        """Search an indexed markdown corpus and return chunks with citations."""

        return search_index(
            db,
            query,
            k=k,
            embedder=embedder,
            pseudonym_mappings=pseudonym_mappings,
        )

    @mcp.tool(name="get_chunk", description=descriptions["get_chunk"])
    def _get_chunk_tool(chunk_id: str) -> dict[str, Any] | None:
        """Return the full content and citation for one markdown corpus chunk."""

        return get_chunk_fn(db, chunk_id)

    @mcp.tool(name="list_sources", description=descriptions["list_sources"])
    def _list_sources_tool() -> list[dict[str, Any]]:
        """List source markdown files currently indexed in the corpus."""

        return list_sources_fn(db)

    mcp.run(transport="stdio")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m context_rag.server")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)
    serve(args.db)
    return 0


def _connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def resolve_tool_descriptions(config_path: Path | None = None) -> dict[str, str]:
    """Return MCP tool descriptions from config with built-in fallbacks."""

    from .cli import load_config

    configured = load_config(config_path).get("tool_descriptions", {})
    descriptions = dict(DEFAULT_TOOL_DESCRIPTIONS)
    if isinstance(configured, dict):
        for name in descriptions:
            value = configured.get(name)
            if value:
                descriptions[name] = str(value)
    return descriptions


def _row_result(row: sqlite3.Row) -> dict[str, Any]:
    result: dict[str, Any] = {
        "chunk_id": row["id"],
        "source": row["source"],
        "heading_path": json.loads(row["heading_path"]),
        "text": row["text"],
        "start_line": int(row["start_line"]),
        "end_line": int(row["end_line"]),
        "score": 1.0,
        "rank_bm25": None,
        "rank_dense": None,
    }
    for key in ("src", "ts", "ts_end", "chapter", "sheet"):
        if row[key] is not None:
            result[key] = row[key]
    for key in ("page", "page_end", "slide", "slide_end"):
        if row[key] is not None:
            result[key] = int(row[key])
    return result


def _with_citation(hit: dict[str, Any]) -> dict[str, Any]:
    return {**hit, "citation": format_citation(hit)}


if __name__ == "__main__":
    raise SystemExit(main())
