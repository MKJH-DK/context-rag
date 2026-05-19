from __future__ import annotations

from context_rag.pseudonyms import expand_query, load_mappings
from context_rag import server as server_mod


def test_expand_query_replaces_real_names() -> None:
    query = "hvad sagde Kristian Andersen om RAG"

    expanded = expand_query(query, {"Kristian Andersen": "Underviser_01"})

    assert expanded == "hvad sagde Underviser_01 om RAG"


def test_missing_pseudonyms_file_is_noop(tmp_path) -> None:
    query = "hvad sagde Kristian Andersen om RAG"

    mappings = load_mappings(tmp_path / "missing.yaml")

    assert mappings == {}
    assert expand_query(query, mappings) == query


def test_expand_query_respects_word_boundaries() -> None:
    query = "Kristian Andersen besøgte Kristianborg"

    expanded = expand_query(query, {"Kristian Andersen": "Underviser_01"})

    assert expanded == "Underviser_01 besøgte Kristianborg"


def test_load_mappings_ignores_allowlist_and_suggestions(tmp_path) -> None:
    path = tmp_path / "pseudonyms.yaml"
    path.write_text(
        """
mappings:
  "Kristian Andersen": Underviser_01
org_allowlist:
  - Microsoft
suggestions:
  "Jens Hansen":
    category: Person
""".lstrip(),
        encoding="utf-8",
    )

    assert load_mappings(path) == {"Kristian Andersen": "Underviser_01"}


def test_search_index_expands_query_before_retrieval(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_hybrid_search(db_path, embedder, query, k=10):
        calls.append((db_path, embedder, query, k))
        return []

    monkeypatch.setattr(server_mod, "hybrid_search", fake_hybrid_search)

    server_mod.search_index(
        tmp_path / "index.db",
        "hvad sagde Kristian Andersen",
        k=3,
        embedder=object(),
        pseudonym_mappings={"Kristian Andersen": "Underviser_01"},
    )

    assert calls[0][2] == "hvad sagde Underviser_01"
