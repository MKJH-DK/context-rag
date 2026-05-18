from pathlib import Path

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
