"""Command line interface for context-rag."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from .chunker import chunk_markdown
from .embeddings import DEFAULT_MODEL, Embedder
from .indexer import Indexer, check_embedding_model
from .query import bm25_search, dense_search, format_citation, hybrid_search
from .server import serve


CONFIG_NAME = "context-rag.yaml"
CONFIG_TEMPLATE = """# context-rag configuration
# Customize tool_descriptions to control how MCP clients describe this corpus.
# Available MCP tools: search, get_chunk, list_sources.
corpus_root: .
database_path: .context-rag/index.db
# embedding_model: BAAI/bge-m3  # default, multilingual, ~2GB
# Alternatives:
#   intfloat/multilingual-e5-base  # ~1.1GB, good Danish quality
#   intfloat/multilingual-e5-small # ~470MB, decent Danish quality
#   paraphrase-multilingual-MiniLM-L12-v2 # ~470MB, OK Danish
chunk:
  max_chars: 4000
  overlap: 0
embedding:
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
    index_parser.add_argument("--model", default=None, help="override embedding model")

    query_parser = sub.add_parser("query", help="query the index")
    query_parser.add_argument("query")
    query_parser.add_argument("--k", type=int, default=10)
    query_parser.add_argument(
        "--mode", choices=["hybrid", "bm25", "dense"], default=None
    )
    query_parser.add_argument("--model", default=None, help="override embedding model")

    serve_parser = sub.add_parser("serve", help="start the stdio MCP server")
    serve_parser.add_argument("--db", type=Path, default=None)

    install_parser = sub.add_parser(
        "install-claude-desktop",
        help="write a Claude Desktop MCP server entry",
    )
    install_parser.add_argument("--name", default="context-rag")
    install_parser.add_argument("--cwd", type=Path, default=Path("."))
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "init":
        return cmd_init()
    if args.command == "index":
        return cmd_index(args.directory, model=args.model)
    if args.command == "query":
        return cmd_query(args.query, k=args.k, mode=args.mode, model=args.model)
    if args.command == "serve":
        config = load_config()
        serve(args.db or Path(config["database_path"]))
        return 0
    if args.command == "install-claude-desktop":
        return cmd_install_claude_desktop(
            name=args.name,
            cwd=args.cwd,
            dry_run=args.dry_run,
            force=args.force,
        )
    raise AssertionError(args.command)


def cmd_init() -> int:
    path = Path(CONFIG_NAME)
    if path.exists():
        print(f"{CONFIG_NAME} already exists")
        return 0
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    print(f"wrote {CONFIG_NAME}")
    return 0


def cmd_index(directory: Path, *, model: str | None = None) -> int:
    config = load_config()
    root = directory.expanduser().resolve()
    db_path = Path(config["database_path"])
    chunk_config = config["chunk"]
    embed_config = config["embedding"]
    model_name = model or str(config["embedding_model"])

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
        model_name=model_name,
        batch_size=int(embed_config["batch_size"]),
    )
    if db_path.exists():
        check_embedding_model(db_path, embedder.model_name)
    vectors = embedder.encode([chunk.text for chunk in chunks]) if chunks else []
    indexer = Indexer(db_path)
    indexer.add_chunks(chunks, vectors, embedding_model=embedder.model_name)
    stats = indexer.get_stats()
    print(
        f"indexed {len(files)} files, {stats['total_chunks']} chunks, "
        f"embeddings={embedder.model_name}, db={db_path}"
    )
    return 0


def cmd_query(
    query: str, *, k: int, mode: str | None, model: str | None = None
) -> int:
    config = load_config()
    selected_mode = mode or str(config["retrieval"]["default_mode"])
    db_path = Path(config["database_path"])
    model_name = model or str(config["embedding_model"])
    embedder = Embedder(model_name=model_name)

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


def cmd_install_claude_desktop(
    *, name: str, cwd: Path, dry_run: bool = False, force: bool = False
) -> int:
    config_path = resolve_claude_desktop_config_path()
    config = _load_claude_desktop_config(config_path)
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        config["mcpServers"] = servers

    if name in servers and not force:
        print(
            f'MCP entry "{name}" already exists in {config_path}. '
            "Re-run with --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    servers[name] = _claude_desktop_entry(cwd)
    rendered = json.dumps(config, indent=2) + "\n"

    if dry_run:
        print(rendered, end="")
        return 0

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(rendered, encoding="utf-8")
    print(f'Wrote MCP entry "{name}" to {config_path}. Restart Claude Desktop to load.')
    return 0


def resolve_claude_desktop_config_path() -> Path:
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        return (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or Path(CONFIG_NAME)
    if not config_path.exists():
        return _defaults()
    data = _parse_simple_yaml(config_path)
    defaults = _defaults()
    merged = _merge(defaults, data)
    if "embedding_model" not in data:
        embedding = data.get("embedding", {})
        if isinstance(embedding, dict) and embedding.get("model"):
            merged["embedding_model"] = embedding["model"]
    return merged


def _load_claude_desktop_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"mcpServers": {}}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {"mcpServers": {}}
    config = json.loads(text)
    if not isinstance(config, dict):
        return {"mcpServers": {}}
    config.setdefault("mcpServers", {})
    return config


def _claude_desktop_entry(cwd: Path) -> dict[str, Any]:
    return {
        "command": sys.executable,
        "args": ["-m", "context_rag.cli", "serve"],
        "cwd": str(cwd.expanduser().resolve()),
        "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    }


def _defaults() -> dict[str, Any]:
    return {
        "corpus_root": ".",
        "database_path": ".context-rag/index.db",
        "embedding_model": DEFAULT_MODEL,
        "chunk": {"max_chars": 4000, "overlap": 0},
        "embedding": {"batch_size": 16},
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
    value = _strip_inline_comment(value).strip()
    if value.isdigit():
        return int(value)
    return value.strip("'\"")


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(value):
        if char in "'\"" and (idx == 0 or value[idx - 1] != "\\"):
            quote = None if quote == char else char
        elif (
            char == "#"
            and quote is None
            and (idx == 0 or value[idx - 1].isspace())
        ):
            return value[:idx].rstrip()
    return value


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
