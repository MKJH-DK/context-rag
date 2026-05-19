from __future__ import annotations

from context_rag import cli
from context_rag.embeddings import ensure_model_cached, is_model_cached


class FakeEmbedder:
    def __init__(self, model_name=None, *, batch_size=16, no_download=False, **kwargs):
        self.model_name = model_name or "test-model"
        self.batch_size = batch_size
        self.no_download = no_download

    def encode(self, texts):
        return []


def test_is_model_cached_finds_huggingface_hub_cache(tmp_path) -> None:
    cached = tmp_path / "hub" / "models--BAAI--bge-m3"
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}", encoding="utf-8")

    assert is_model_cached("BAAI/bge-m3", cache_folder=str(tmp_path))


def test_ensure_model_cached_raises_clear_error_when_missing(tmp_path) -> None:
    try:
        ensure_model_cached("missing/model", cache_folder=str(tmp_path))
    except RuntimeError as exc:
        message = str(exc)
        assert message.startswith("Model missing/model not cached and --no-download passed.")
        assert "Remove flag or pre-download with:" in message
    else:
        raise AssertionError("expected RuntimeError")


def test_index_no_download_checks_cache_before_embedder(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "ensure_model_cached", lambda model: calls.append(model))
    monkeypatch.setattr(cli, "Embedder", FakeEmbedder)

    assert cli.main(["index", str(corpus), "--model", "test-model", "--no-download"]) == 0
    assert calls == ["test-model"]


def test_index_without_no_download_skips_cache_check(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    def fail_cache_check(model):
        raise AssertionError("cache check should not run")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "ensure_model_cached", fail_cache_check)
    monkeypatch.setattr(cli, "Embedder", FakeEmbedder)

    assert cli.main(["index", str(corpus), "--model", "test-model"]) == 0


def test_query_no_download_checks_cache_before_search(tmp_path, monkeypatch) -> None:
    calls = []
    queries = []

    def fake_hybrid_search(db_path, embedder, query, k=10, rrf_k=60):
        queries.append(query)
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "ensure_model_cached", lambda model: calls.append(model))
    monkeypatch.setattr(cli, "Embedder", FakeEmbedder)
    monkeypatch.setattr(cli, "hybrid_search", fake_hybrid_search)

    assert cli.main(["query", "hello", "--model", "test-model", "--no-download"]) == 0
    assert calls == ["test-model"]
    assert queries == ["hello"]
