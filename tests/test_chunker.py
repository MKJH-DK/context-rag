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


def test_loc_markers_are_removed_and_stamped_on_chunks(tmp_path: Path) -> None:
    path = tmp_path / "loc.md"
    path.write_text(
        "\n".join(
            [
                "# Course",
                "## Video",
                "<!-- loc: src=Module/video.mp4; ts=00:12; ts_end=00:20 -->",
                "Segment text",
                "## Reading",
                "<!-- loc: src=Module/book.pdf; p=4 -->",
                "Page text",
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_markdown(path)

    assert chunks[1].src == "Module/video.mp4"
    assert chunks[1].ts == "00:12"
    assert chunks[1].ts_end is None
    assert "<!-- loc:" not in chunks[1].text
    assert chunks[1].text.startswith("## Video")
    assert chunks[2].src == "Module/book.pdf"
    assert chunks[2].page == 4
    assert chunks[2].text.endswith("Page text")


def test_ts_range_uses_last_marker_with_same_src(tmp_path: Path) -> None:
    path = tmp_path / "range.md"
    path.write_text(
        "\n".join(
            [
                "# Course",
                "## Video",
                "<!-- loc: src=video.mp4; ts=00:00 -->",
                "Opening",
                "<!-- loc: src=video.mp4; ts=00:15 -->",
                "Middle",
                "<!-- loc: src=video.mp4; ts=00:30 -->",
                "Closing",
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_markdown(path)

    assert chunks[1].ts == "00:00"
    assert chunks[1].ts_end == "00:30"


def test_ts_range_omits_end_when_markers_share_ts(tmp_path: Path) -> None:
    path = tmp_path / "same-ts.md"
    path.write_text(
        "\n".join(
            [
                "# Course",
                "## Video",
                "<!-- loc: src=video.mp4; ts=00:00 -->",
                "Opening",
                "<!-- loc: src=video.mp4; ts=00:00 -->",
                "Still same timestamp",
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_markdown(path)

    assert chunks[1].ts == "00:00"
    assert chunks[1].ts_end is None
