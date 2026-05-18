import math

from context_rag.embeddings import Embedder, FALLBACK_MODEL


def test_fallback_embedder_is_normalized() -> None:
    embedder = Embedder()

    vectors = embedder.encode(["retrieval augmented generation"])
    vector = vectors[0]

    assert embedder.model_name in {"BAAI/bge-m3", FALLBACK_MODEL}
    assert math.isclose(sum(value * value for value in vector), 1.0)
