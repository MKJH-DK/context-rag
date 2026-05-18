"""BM25, dense, and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path
import json
import re
import sqlite3
from typing import Any

from .indexer import cosine_similarity, unpack_vector


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "er",
    "hvad",
    "i",
    "is",
    "of",
    "og",
    "the",
    "to",
    "what",
}


def bm25_search(db: str | Path, query: str, k: int = 20) -> list[dict[str, Any]]:
    fts_query = _fts_query(query)
    if not fts_query or k <= 0:
        return []
    with _connect(db) as con:
        rows = con.execute(
            """
            SELECT c.id, c.source, c.heading_path, c.text, c.start_line, c.end_line,
                   c.src, c.ts, c.ts_end, c.page, c.chapter, c.slide, c.sheet,
                   bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            (fts_query, k),
        ).fetchall()
    return [_result(row, -float(row["bm25_score"]), rank_bm25=rank) for rank, row in enumerate(rows, 1)]


def dense_search(
    db: str | Path,
    embedder: Any,
    query: str,
    k: int = 20,
) -> list[dict[str, Any]]:
    if not query.strip() or k <= 0:
        return []
    query_vector = _first_vector(embedder.encode([query]))
    with _connect(db) as con:
        rows = con.execute(
            """
            SELECT c.id, c.source, c.heading_path, c.text, c.start_line, c.end_line,
                   c.src, c.ts, c.ts_end, c.page, c.chapter, c.slide, c.sheet,
                   v.embedding, v.dim
            FROM chunk_vectors v
            JOIN chunks c ON c.id = v.chunk_id
            """
        ).fetchall()
    scored: list[dict[str, Any]] = []
    for row in rows:
        vector = unpack_vector(row["embedding"], int(row["dim"]))
        scored.append(_result(row, cosine_similarity(query_vector, vector), rank_dense=0))
    scored.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(scored[:k], 1):
        item["rank_dense"] = rank
    return scored[:k]


def hybrid_search(
    db: str | Path,
    embedder: Any,
    query: str,
    k: int = 10,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    pool = max(k * 4, 20)
    bm25_hits = bm25_search(db, query, k=pool)
    dense_hits = dense_search(db, embedder, query, k=pool)
    return rrf_fuse(bm25_hits, dense_hits, k=k, rrf_k=rrf_k)


def rrf_fuse(
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    *,
    k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}

    for field, hits in (("rank_bm25", bm25_hits), ("rank_dense", dense_hits)):
        for rank, hit in enumerate(hits, 1):
            chunk_id = str(hit["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            item = merged.setdefault(chunk_id, dict(hit))
            item[field] = rank

    fused = []
    for chunk_id, item in merged.items():
        item["score"] = scores[chunk_id]
        item.setdefault("rank_bm25", None)
        item.setdefault("rank_dense", None)
        fused.append(item)
    fused.sort(key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
    return fused[:k]


def format_citation(hit: dict[str, Any]) -> str:
    """Return a user-facing citation for a search hit."""
    chunk_id = hit["chunk_id"]
    src = hit.get("src") or hit["source"]
    if hit.get("ts"):
        location = f"{src} @ {hit['ts']}"
    elif hit.get("page") is not None:
        location = f"{src} p.{hit['page']}"
    elif hit.get("slide") is not None:
        location = f"{src} slide {hit['slide']}"
    elif hit.get("sheet"):
        location = f"{src} sheet {hit['sheet']}"
    else:
        location = f"{hit['source']}:{hit['start_line']}-{hit['end_line']}"
    return f"[{location}] (chunk {chunk_id})"


def _result(
    row: sqlite3.Row,
    score: float,
    *,
    rank_bm25: int | None = None,
    rank_dense: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "chunk_id": row["id"],
        "source": row["source"],
        "heading_path": json.loads(row["heading_path"]),
        "text": row["text"],
        "start_line": int(row["start_line"]),
        "end_line": int(row["end_line"]),
        "score": score,
        "rank_bm25": rank_bm25,
        "rank_dense": rank_dense,
    }
    for key in ("src", "ts", "ts_end", "chapter", "sheet"):
        if row[key] is not None:
            result[key] = row[key]
    for key in ("page", "slide"):
        if row[key] is not None:
            result[key] = int(row[key])
    return result


def _connect(db: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    return con


def _fts_query(query: str) -> str:
    terms = [
        term
        for term in re.findall(r"[\w]+", query, flags=re.UNICODE)
        if term.lower() not in STOPWORDS
    ]
    return " OR ".join(f'"{term}"' for term in terms)


def _first_vector(vectors: Any) -> list[float]:
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return [float(value) for value in vectors[0]]
