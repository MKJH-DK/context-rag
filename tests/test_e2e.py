from pathlib import Path

from context_rag.cli import main


def test_cli_e2e_indexes_and_queries_modes(tmp_path: Path, monkeypatch, capsys) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "rag.md").write_text(
        "\n".join(
            [
                "# AI Course",
                "## Retrieval",
                "RAG means retrieval augmented generation with citations.",
                "## Prompting",
                "Prompt templates shape model behavior.",
            ]
        ),
        encoding="utf-8",
    )
    (corpus / "agents.md").write_text(
        "\n".join(
            [
                "# Agents",
                "## Tools",
                "Tool calls make actions visible in the interface.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert main(["index", str(corpus)]) == 0

    for mode in ("hybrid", "bm25", "dense"):
        assert main(["query", "retrieval augmented generation", "--k", "3", "--mode", mode]) == 0
        output = capsys.readouterr().out
        assert "Retrieval" in output
        assert "rag.md:" in output
