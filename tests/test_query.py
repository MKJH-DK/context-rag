from pathlib import Path

from context_rag.chunker import Chunk
from context_rag.indexer import Indexer
from context_rag.query import _fts_query, bm25_search, format_citation, rrf_fuse


def test_rrf_fusion_combines_ranks_deterministically() -> None:
    bm25 = [
        {"chunk_id": "a", "score": 10.0},
        {"chunk_id": "b", "score": 8.0},
    ]
    dense = [
        {"chunk_id": "b", "score": 0.9},
        {"chunk_id": "c", "score": 0.8},
    ]

    fused = rrf_fuse(bm25, dense, k=3, rrf_k=60)

    assert [hit["chunk_id"] for hit in fused] == ["b", "a", "c"]
    assert fused[0]["rank_bm25"] == 2
    assert fused[0]["rank_dense"] == 1
    assert fused[1]["rank_bm25"] == 1
    assert fused[1]["rank_dense"] is None


def test_fts_query_drops_common_danish_stopwords() -> None:
    assert _fts_query("hvad er RAG") == '"RAG"'


def test_bm25_results_include_non_null_loc_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    chunk = Chunk(
        id="chunk:one",
        source="module.md",
        heading_path=("Root",),
        text="retrieval citations",
        start_line=1,
        end_line=2,
        src="Module/video.mp4",
        ts="02:15",
    )
    Indexer(db_path).add_chunks([chunk])

    hit = bm25_search(db_path, "retrieval", k=1)[0]

    assert hit["src"] == "Module/video.mp4"
    assert hit["ts"] == "02:15"
    assert "page" not in hit


def test_format_citation_prefers_location_metadata() -> None:
    base = {
        "chunk_id": "chunk:one",
        "source": "module.md",
        "start_line": 4,
        "end_line": 8,
    }

    assert (
        format_citation({**base, "src": "Module/video.mp4", "ts": "02:15"})
        == "[Module/video.mp4 @ 02:15] (chunk chunk:one)"
    )
    assert (
        format_citation(
            {
                **base,
                "src": "video.mp4",
                "ts": "01:07",
                "ts_end": "01:40",
            }
        )
        == "[video.mp4 @ 01:07-01:40] (chunk chunk:one)"
    )
    assert (
        format_citation({**base, "src": "Module/book.pdf", "page": 3})
        == "[Module/book.pdf p.3] (chunk chunk:one)"
    )
    assert (
        format_citation({**base, "src": "bog.pdf", "page": 3, "page_end": 5})
        == "[bog.pdf p.3-5] (chunk chunk:one)"
    )
    assert (
        format_citation({**base, "src": "Module/slides.pptx", "slide": 4})
        == "[Module/slides.pptx slide 4] (chunk chunk:one)"
    )
    assert (
        format_citation({**base, "src": "deck.pptx", "slide": 4, "slide_end": 7})
        == "[deck.pptx slide 4-7] (chunk chunk:one)"
    )
    assert (
        format_citation({**base, "src": "Module/data.xlsx", "sheet": "Budget"})
        == "[Module/data.xlsx sheet Budget] (chunk chunk:one)"
    )
    assert format_citation(base) == "[module.md:4-8] (chunk chunk:one)"
