from pathlib import Path

from context_rag.chunker import Chunk
from context_rag.indexer import Indexer


def test_indexer_creates_schema_and_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    chunk = Chunk(
        id="chunk:one",
        source="notes.md",
        heading_path=("Root", "Topic"),
        text="retrieval augmented generation",
        start_line=1,
        end_line=3,
    )

    indexer = Indexer(db_path)
    indexer.add_chunks([chunk], [[1.0, 0.0, 0.0]])
    stats = indexer.get_stats()

    assert stats["total_chunks"] == 1
    assert stats["embedded_chunks"] == 1
    assert stats["sources"] == ["notes.md"]
