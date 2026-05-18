from pathlib import Path

from context_rag.chunker import chunk_markdown


def test_empty_markdown_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")

    assert chunk_markdown(path) == []


def test_chunks_on_h2_and_h3_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(
        "\n".join(
            [
                "# Course",
                "preface",
                "## Retrieval",
                "bm25 text",
                "### Dense",
                "embedding text",
                "#### Detail",
                "kept with dense",
                "## Agents",
                "tool text",
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_markdown(path)

    assert [chunk.heading_path for chunk in chunks] == [
        ("Course",),
        ("Course", "Retrieval"),
        ("Course", "Retrieval", "Dense"),
        ("Course", "Agents"),
    ]
    assert [chunk.start_line for chunk in chunks] == [1, 3, 5, 9]
    assert [chunk.end_line for chunk in chunks] == [2, 4, 8, 10]
    assert chunks[2].text.startswith("### Dense")
    assert "#### Detail" in chunks[2].text


def test_overlap_defaults_to_zero_when_splitting(tmp_path: Path) -> None:
    path = tmp_path / "long.md"
    path.write_text(
        "\n".join(["## Topic", *[f"line {idx} " + ("x" * 80) for idx in range(20)]]),
        encoding="utf-8",
    )

    chunks = chunk_markdown(path, max_chars=500)

    assert len(chunks) > 1
    assert chunks[0].end_line + 1 == chunks[1].start_line


def test_configurable_overlap_repeats_tail_lines(tmp_path: Path) -> None:
    path = tmp_path / "long.md"
    path.write_text(
        "\n".join(["## Topic", *[f"line {idx} " + ("x" * 80) for idx in range(20)]]),
        encoding="utf-8",
    )

    chunks = chunk_markdown(path, max_chars=500, overlap=120)

    assert len(chunks) > 1
    assert chunks[1].start_line <= chunks[0].end_line
    assert chunks[0].text.splitlines()[-1] in chunks[1].text
