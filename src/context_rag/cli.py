"""Command line interface for context-rag."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from .chunker import chunk_markdown
from .embeddings import Embedder
from .indexer import Indexer
from .query import bm25_search, dense_search, format_citation, hybrid_search
from .server import serve


CONFIG_NAME = "context-rag.yaml"
CONFIG_TEMPLATE = """# context-rag configuration
# Customize tool_descriptions to control how MCP clients describe this corpus.
# Available MCP tools: search, get_chunk, list_sources.
corpus_root: .
database_path: .context-rag/index.db
chunk:
  max_chars: 4000
  overlap: 0
embedding:
  model: BAAI/bge-m3
  batch_size: 16
retrieval:
  default_mode: hybrid
  rrf_k: 60
tool_descriptions:
  search: "Search the indexed markdown corpus. Returns relevant excerpts with citations."
  get_chunk: "Fetch the full content of a chunk by id."
  list_sources: "List all source files in the index."
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write a context-rag.yaml template")

    index_parser = sub.add_parser("index", help="index markdown files")
    index_parser.add_argument("directory", type=Path)

    query_parser = sub.add_parser("query", help="query the index")
    query_parser.add_argument("query")
    query_parser.add_argument("--k", type=int, default=10)
    query_parser.add_argument(
        "--mode", choices=["hybrid", "bm25", "dense"], default=None
    )

    serve_parser = sub.add_parser("serve", help="start the stdio MCP server")
    serve_parser.add_argument("--db", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "init":
        return cmd_init()
    if args.command == "index":
        return cmd_index(args.directory)
    if args.command == "query":
        return cmd_query(args.query, k=args.k, mode=args.mode)
    if args.command == "serve":
        config = load_config()
        serve(args.db or Path(config["database_path"]))
        return 0
    raise AssertionError(args.command)


def cmd_init() -> int:
    path = Path(CONFIG_NAME)
    if path.exists():
        print(f"{CONFIG_NAME} already exists")
        return 0
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    print(f"wrote {CONFIG_NAME}")
    return 0


def cmd_index(directory: Path) -> int:
    config = load_config()
    root = directory.expanduser().resolve()
    db_path = Path(config["database_path"])
    chunk_config = config["chunk"]
    embed_config = config["embedding"]

    files = _markdown_files(root)
    chunks = []
    for file_path in files:
        chunks.extend(
            chunk_markdown(
                file_path,
                max_chars=int(chunk_config["max_chars"]),
                overlap=int(chunk_config["overlap"]),
            )
        )

    embedder = Embedder(
        model_name=str(embed_config["model"]),
        batch_size=int(embed_config["batch_size"]),
    )
    vectors = embedder.encode([chunk.text for chunk in chunks]) if chunks else []
    indexer = Indexer(db_path)
    indexer.add_chunks(chunks, vectors)
    stats = indexer.get_stats()
    print(
        f"indexed {len(files)} files, {stats['total_chunks']} chunks, "
        f"embeddings={embedder.model_name}, db={db_path}"
    )
    return 0


def cmd_query(query: str, *, k: int, mode: str | None) -> int:
    config = load_config()
    selected_mode = mode or str(config["retrieval"]["default_mode"])
    db_path = Path(config["database_path"])
    embedder = Embedder(model_name=str(config["embedding"]["model"]))

    if selected_mode == "bm25":
        hits = bm25_search(db_path, query, k=k)
    elif selected_mode == "dense":
        hits = dense_search(db_path, embedder, query, k=k)
    else:
        hits = hybrid_search(
            db_path,
            embedder,
            query,
            k=k,
            rrf_k=int(config["retrieval"]["rrf_k"]),
        )

    for idx, hit in enumerate(hits, 1):
        heading = " > ".join(hit["heading_path"]) or "(root)"
        citation = format_citation(hit)
        print(f"{idx}. {heading} {citation} score={hit['score']:.4f}")
        print(_preview(hit["text"]))
    return 0


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or Path(CONFIG_NAME)
    if not config_path.exists():
        return _defaults()
    data = _parse_simple_yaml(config_path)
    defaults = _defaults()
    return _merge(defaults, data)


def _defaults() -> dict[str, Any]:
    return {
        "corpus_root": ".",
        "database_path": ".context-rag/index.db",
        "chunk": {"max_chars": 4000, "overlap": 0},
        "embedding": {"model": "BAAI/bge-m3", "batch_size": 16},
        "retrieval": {"default_mode": "hybrid", "rrf_k": 60},
        "tool_descriptions": {
            "search": (
                "Search the indexed markdown corpus. Returns relevant excerpts "
                "with citations."
            ),
            "get_chunk": "Fetch the full content of a chunk by id.",
            "list_sources": "List all source files in the index.",
        },
    }


def _markdown_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if ".context-rag" not in path.parts and path.is_file()
    )


def _preview(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = raw_line.strip().split(":", 1)
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(value)
    return root


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base


def _scalar(value: str) -> Any:
    if value.isdigit():
        return int(value)
    return value.strip("'\"")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
