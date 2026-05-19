import sys
import types
from unittest.mock import Mock

from pathlib import Path

from context_rag import server as server_mod
from context_rag.chunker import Chunk
from context_rag.indexer import Indexer
from context_rag.server import get_chunk, list_sources, resolve_tool_descriptions


def test_server_helpers_return_citations(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    chunk = Chunk(
        id="chunk:one",
        source="notes.md",
        heading_path=("Root",),
        text="tool visible citations",
        start_line=4,
        end_line=6,
        src="video.mp4",
        ts="01:20",
    )
    indexer = Indexer(db_path)
    indexer.add_chunks([chunk], [[1.0, 0.0, 0.0]])

    assert list_sources(db_path)[0]["source"] == "notes.md"
    assert get_chunk(db_path, "chunk:one")["citation"] == "[video.mp4 @ 01:20] (chunk chunk:one)"


def test_tool_description_overrides_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "context-rag.yaml"
    config_path.write_text(
        """
tool_descriptions:
  search: "custom"
""",
        encoding="utf-8",
    )

    descriptions = resolve_tool_descriptions(config_path)

    assert descriptions["search"] == "custom"
    assert descriptions["get_chunk"] == "Fetch the full content of a chunk by id."


def test_serve_preloads_embedder_before_search_tools_run(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self.tools = {}

        def tool(self, *, name: str, description: str):
            def register(fn):
                self.tools[name] = fn
                return fn

            return register

        def run(self, *, transport: str) -> None:
            assert transport == "stdio"
            self.tools["search"]("alpha", k=1)
            self.tools["search"]("beta", k=2)

    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeMCP
    server_module = types.ModuleType("mcp.server")
    server_module.fastmcp = fastmcp_module
    mcp_module = types.ModuleType("mcp")
    mcp_module.server = server_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)

    embedder = object()
    embedder_cls = Mock(return_value=embedder)
    calls = []

    def fake_hybrid_search(db_path, active_embedder, query, k=10):
        calls.append((db_path, active_embedder, query, k))
        return []

    monkeypatch.setattr(server_mod, "Embedder", embedder_cls)
    monkeypatch.setattr(server_mod, "hybrid_search", fake_hybrid_search)

    server_mod.serve(tmp_path / "index.db")

    assert embedder_cls.call_count == 1
    assert [call[2:] for call in calls] == [("alpha", 1), ("beta", 2)]
    assert all(call[1] is embedder for call in calls)
