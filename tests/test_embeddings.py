import math
import sys
import types
from unittest.mock import Mock

from context_rag.embeddings import Embedder, FALLBACK_MODEL


def _install_fake_sentence_transformer(monkeypatch, encode_return=None):
    if encode_return is None:
        encode_return = [[1.0, 0.0, 0.0]]

    class FakeSentenceTransformer:
        instances = []

        def __init__(self, model_name: str, **kwargs) -> None:
            self.model_name = model_name
            self.kwargs = kwargs
            self.encode = Mock(return_value=encode_return)
            self.half = Mock(return_value=self)
            FakeSentenceTransformer.instances.append(self)

        def get_sentence_embedding_dimension(self) -> int:
            return 3

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeSentenceTransformer


def _install_fake_torch(monkeypatch, *, cuda_available: bool):
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=Mock(return_value=cuda_available))
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_fallback_embedder_is_normalized() -> None:
    embedder = Embedder()

    vectors = embedder.encode(["retrieval augmented generation"])
    vector = vectors[0]

    assert embedder.model_name in {"BAAI/bge-m3", FALLBACK_MODEL}
    assert math.isclose(sum(value * value for value in vector), 1.0)


def test_embedder_reads_model_name_from_config(tmp_path, monkeypatch) -> None:
    fake_cls = _install_fake_sentence_transformer(monkeypatch)
    monkeypatch.setenv("CONTEXT_RAG_DEVICE", "cpu")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "context-rag.yaml").write_text(
        "embedding_model: intfloat/multilingual-e5-small  # compact\n",
        encoding="utf-8",
    )

    embedder = Embedder()

    assert embedder.requested_model_name == "intfloat/multilingual-e5-small"
    assert embedder.model_name == "intfloat/multilingual-e5-small"
    assert fake_cls.instances[0].kwargs["device"] == "cpu"


def test_embedder_auto_detects_cpu_when_cuda_unavailable(monkeypatch) -> None:
    fake_cls = _install_fake_sentence_transformer(monkeypatch)
    _install_fake_torch(monkeypatch, cuda_available=False)
    monkeypatch.delenv("CONTEXT_RAG_DEVICE", raising=False)

    embedder = Embedder(model_name="test-model")

    assert embedder.device == "cpu"
    assert fake_cls.instances[0].kwargs["device"] == "cpu"


def test_context_rag_device_override_wins(monkeypatch) -> None:
    fake_cls = _install_fake_sentence_transformer(monkeypatch)
    _install_fake_torch(monkeypatch, cuda_available=True)
    monkeypatch.setenv("CONTEXT_RAG_DEVICE", "cpu")

    embedder = Embedder(model_name="test-model")

    assert embedder.device == "cpu"
    assert fake_cls.instances[0].kwargs["device"] == "cpu"


def test_embed_query_cached_reuses_underlying_encode(monkeypatch) -> None:
    fake_cls = _install_fake_sentence_transformer(
        monkeypatch,
        encode_return=[[0.25, 0.5, 0.75]],
    )
    monkeypatch.setenv("CONTEXT_RAG_DEVICE", "cpu")
    embedder = Embedder(model_name="test-model")

    first = embedder.embed_query_cached("hvad er RAG")
    second = embedder.embed_query_cached("hvad er RAG")
    third = embedder.embed_query_cached("hvad er RAG")

    assert list(first) == list(second)
    assert list(second) == list(third)
    assert fake_cls.instances[0].encode.call_count == 1
    assert first is not second

    first[0] = 9.0
    assert list(second) == [0.25, 0.5, 0.75]
