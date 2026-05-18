from context_rag.query import _fts_query, rrf_fuse


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
