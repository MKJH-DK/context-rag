import math
import sys
import types

from context_rag.embeddings import Embedder, FALLBACK_MODEL


def test_fallback_embedder_is_normalized() -> None:
    embedder = Embedder()

    vectors = embedder.encode(["retrieval augmented generation"])
    vector = vectors[0]

    assert embedder.model_name in {"BAAI/bge-m3", FALLBACK_MODEL}
    assert math.isclose(sum(value * value for value in vector), 1.0)


def test_embedder_reads_model_name_from_config(tmp_path, monkeypatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs) -> None:
            self.model_name = model_name

        def get_sentence_embedding_dimension(self) -> int:
            return 3

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "context-rag.yaml").write_text(
        "embedding_model: intfloat/multilingual-e5-small  # compact\n",
        encoding="utf-8",
    )

    embedder = Embedder()

    assert embedder.requested_model_name == "intfloat/multilingual-e5-small"
    assert embedder.model_name == "intfloat/multilingual-e5-small"
