"""SQLite index for markdown chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import sqlite3
import struct
from typing import Any, Iterable, Sequence

from .chunker import Chunk


SCHEMA_VERSION = 2
REBUILD_HINT = "Rebuild: rm .context-rag/index.db && context-rag index ."


class SchemaVersionError(RuntimeError):
    """Raised when an existing database uses a different schema version."""


class EmbeddingModelMismatchError(RuntimeError):
    """Raised when index metadata does not match the requested model."""


@dataclass(frozen=True)
class IndexStats:
    database_path: str
    total_chunks: int
    embedded_chunks: int
    sources: list[str]
    vector_backend: str


class Indexer:
    """Create and populate the context-rag SQLite index."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_schema(self.db_path)

    @staticmethod
    def create_schema(db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(path) as con:
            _check_schema_version(con)
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    src TEXT,
                    ts TEXT,
                    ts_end TEXT,
                    page INTEGER,
                    chapter TEXT,
                    slide INTEGER,
                    sheet TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            con.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    id UNINDEXED,
                    source UNINDEXED,
                    heading_path UNINDEXED,
                    text
                )
                """
            )
            backend = _ensure_vector_table(con)
            con.execute(
                """
                INSERT INTO meta(key, value) VALUES ('vector_backend', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (backend,),
            )
            con.commit()

    def add_chunks(
        self,
        chunks: Iterable[Chunk],
        embeddings: Any | None = None,
        *,
        embedding_model: str | None = None,
    ) -> None:
        chunk_list = list(chunks)
        vectors = _rows(embeddings) if embeddings is not None else [None] * len(chunk_list)
        if len(vectors) != len(chunk_list):
            raise ValueError("embeddings length must match chunks length")

        with _connect(self.db_path) as con:
            _check_schema_version(con)
            if embedding_model is not None:
                _set_meta(con, "embedding_model", embedding_model)
            for chunk, vector in zip(chunk_list, vectors):
                heading_json = json.dumps(list(chunk.heading_path), ensure_ascii=False)
                con.execute(
                    """
                    INSERT INTO chunks(
                        id, source, heading_path, text, start_line, end_line,
                        src, ts, ts_end, page, chapter, slide, sheet, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        source = excluded.source,
                        heading_path = excluded.heading_path,
                        text = excluded.text,
                        start_line = excluded.start_line,
                        end_line = excluded.end_line,
                        src = excluded.src,
                        ts = excluded.ts,
                        ts_end = excluded.ts_end,
                        page = excluded.page,
                        chapter = excluded.chapter,
                        slide = excluded.slide,
                        sheet = excluded.sheet,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        chunk.id,
                        chunk.source,
                        heading_json,
                        chunk.text,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.src,
                        chunk.ts,
                        chunk.ts_end,
                        chunk.page,
                        chunk.chapter,
                        chunk.slide,
                        chunk.sheet,
                        "{}",
                    ),
                )
                con.execute("DELETE FROM chunks_fts WHERE id = ?", (chunk.id,))
                con.execute(
                    "INSERT INTO chunks_fts(id, source, heading_path, text) VALUES (?, ?, ?, ?)",
                    (chunk.id, chunk.source, heading_json, chunk.text),
                )
                if vector is not None:
                    _upsert_vector(con, chunk.id, vector)
            con.commit()

    def get_stats(self) -> dict[str, Any]:
        with _connect(self.db_path) as con:
            _check_schema_version(con)
            vector_backend = _meta(con, "vector_backend", "blob")
            total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedded = con.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
            sources = [
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT source FROM chunks ORDER BY source"
                ).fetchall()
            ]
        return IndexStats(
            database_path=str(self.db_path),
            total_chunks=int(total),
            embedded_chunks=int(embedded),
            sources=sources,
            vector_backend=vector_backend,
        ).__dict__


def unpack_vector(payload: bytes, dim: int) -> list[float]:
    if not payload or dim <= 0:
        return []
    return list(struct.unpack(f"<{dim}f", payload))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def get_embedding_model(db_path: str | Path) -> str | None:
    path = Path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(str(path)) as con:
        try:
            row = con.execute(
                "SELECT value FROM meta WHERE key = 'embedding_model'"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return str(row[0]) if row else None


def check_embedding_model(db_path: str | Path, requested_model: str) -> None:
    built_model = get_embedding_model(db_path)
    if built_model != requested_model:
        built = built_model or "unknown"
        raise EmbeddingModelMismatchError(
            f"Index built with model {built} but config requests {requested_model}. "
            f"{REBUILD_HINT}"
        )


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return con


def _check_schema_version(con: sqlite3.Connection) -> None:
    row = con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        con.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        return
    found = int(row["value"])
    if found != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Index schema version {found} is not supported by this code "
            f"(expected {SCHEMA_VERSION}). please rebuild index: "
            "rm .context-rag/index.db && context-rag index ."
        )


def _ensure_vector_table(con: sqlite3.Connection) -> str:
    try:
        import sqlite_vec

        sqlite_vec.load(con)
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(id TEXT PRIMARY KEY, embedding float[1024])"
        )
        backend = "sqlite-vec"
    except Exception:
        backend = "blob"
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_vectors (
            chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    return backend


def _upsert_vector(con: sqlite3.Connection, chunk_id: str, vector: Sequence[float]) -> None:
    values = [float(value) for value in vector]
    payload = struct.pack(f"<{len(values)}f", *values)
    con.execute(
        """
        INSERT INTO chunk_vectors(chunk_id, dim, embedding) VALUES (?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            dim = excluded.dim,
            embedding = excluded.embedding
        """,
        (chunk_id, len(values), payload),
    )
    if _meta(con, "vector_backend", "blob") == "sqlite-vec":
        con.execute("DELETE FROM chunk_vec WHERE id = ?", (chunk_id,))
        if len(values) == 1024:
            con.execute(
                "INSERT INTO chunk_vec(id, embedding) VALUES (?, ?)",
                (chunk_id, payload),
            )
        else:
            _set_meta(con, "vector_backend", "blob")


def _rows(vectors: Any) -> list[Any]:
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return list(vectors)


def _set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        """
        INSERT INTO meta(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _meta(con: sqlite3.Connection, key: str, default: str) -> str:
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else default
