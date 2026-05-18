from pathlib import Path
import sqlite3

from context_rag.chunker import Chunk
from context_rag.indexer import Indexer, SchemaVersionError


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


def test_indexer_roundtrips_loc_metadata_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    chunk = Chunk(
        id="chunk:meta",
        source="module.md",
        heading_path=("Root",),
        text="metadata visible content",
        start_line=10,
        end_line=12,
        src="Module/video.mp4",
        ts="02:15",
        ts_end="02:30",
        page=3,
        chapter="Intro",
        slide=4,
        sheet="Data",
    )

    Indexer(db_path).add_chunks([chunk])

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT src, ts, ts_end, page, chapter, slide, sheet FROM chunks WHERE id = ?",
            ("chunk:meta",),
        ).fetchone()

    assert row == ("Module/video.mp4", "02:15", "02:30", 3, "Intro", 4, "Data")


def test_old_schema_requires_rebuild(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as con:
        con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '1')")

    try:
        Indexer(db_path)
    except SchemaVersionError as exc:
        assert "please rebuild index" in str(exc)
    else:
        raise AssertionError("expected SchemaVersionError")
