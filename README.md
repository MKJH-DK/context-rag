---
generated_by: openai/gpt-5-codex
generated_at: 2026-05-18T15:39:57Z
agents_md_version: "3.0"
orchestrator: codex
---

# context-rag

Standalone, local-first hybrid RAG for markdown corpora. It chunks markdown by
document structure, indexes chunks in SQLite with FTS5 and vector embeddings,
then serves retrieval through a CLI or stdio MCP server with file and line
citations.

## Status

- **Phase**: MVP
- **Stability**: usable, experimental
- **Default retrieval**: hybrid BM25 + dense with RRF fusion
- **Default embedding model**: `BAAI/bge-m3`

## Install

```bash
git clone git@github.com:MKJH-DK/context-rag.git
cd context-rag
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest -x
```

## Quickstart

```bash
cd /path/to/markdown-corpus
context-rag init
context-rag index .
context-rag query "what is retrieval augmented generation?" --k 5
```

The index is written to `.context-rag/index.db` by default.

## CLI

```bash
context-rag init
context-rag index <dir>
context-rag query "question" --k 10 --mode hybrid
context-rag query "exact term" --mode bm25
context-rag query "paraphrased question" --mode dense
context-rag serve
```

## MCP Integration

Project-local `.mcp.json`:

```json
{
  "mcpServers": {
    "context-rag": {
      "command": "context-rag",
      "args": ["serve"],
      "cwd": "/path/to/markdown-corpus"
    }
  }
}
```

Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "context-rag": {
      "command": "context-rag",
      "args": ["serve"],
      "cwd": "/path/to/markdown-corpus"
    }
  }
}
```

MCP tools:

- `search(query: str, k: int = 10)` returns ranked chunks with citations.
- `get_chunk(chunk_id: str)` returns full chunk content.
- `list_sources()` returns indexed source files.

## Architecture

```text
Markdown files
    |
    v
Structure chunker (H2/H3, line ranges, zero overlap by default)
    |
    v
Embedder (BAAI/bge-m3 via sentence-transformers)
    |
    v
SQLite index
    |-- FTS5 table for BM25
    |-- sqlite-vec when available, BLOB vector fallback otherwise
    |
    v
Query layer (BM25 + dense + RRF)
    |
    v
CLI and stdio MCP server
```

## Configuration

`context-rag init` writes:

```yaml
corpus_root: .
database_path: .context-rag/index.db
chunk:
  max_chars: 4000
  overlap: 0
embedding:
  model: BAAI/bge-m3
  batch_size: 16
retrieval:
  default_mode: hybrid
  rrf_k: 60
```


## Privacy and GDPR

`context-rag` runs index, query, and local search against files and SQLite data
on the local machine. It makes no outbound API calls during normal
index/query/search execution, sends no telemetry, and includes no analytics.

The only built-in network activity is the optional first-use Hugging Face model
download performed by `sentence-transformers` when the configured embedding
model is not already cached. Use `--no-download` on `index` or `query` to fail
instead of fetching model weights:

```bash
context-rag index . --no-download
context-rag query "question" --no-download
```

Pseudonym query expansion loads `pseudonyms.yaml` from the corpus working
directory. Users can type real names in a query, and configured names are
expanded to the same placeholders that were indexed, for example
`Kristian Andersen` to `Underviser_01`. The original query text is not written
to the index.

When `context-rag` is used through Claude Desktop or another LLM client, that
client is the third-party processor for any text it receives from tool results.
That processing is outside this tool; `context-rag` only returns local retrieval
results to the caller.

The user remains the data controller. This tool is designed for privacy by
default, but it is not legal advice and does not replace a concrete GDPR
assessment of the source material and processing purpose.

## Limitations

- The first bge-m3 run downloads local model weights through
  `sentence-transformers`.
- If `sqlite-vec` is unavailable, dense vectors are stored as SQLite BLOBs and
  scored in Python.
- If embedding dependencies are unavailable, the CLI can use a deterministic
  local hashing fallback for tests and smoke runs, but bge-m3 is the intended
  retrieval model.
- Index refresh is explicit. There is no watcher in v1.

## Non-goals

- No graph RAG.
- No agentic retrieval loops.
- No automatic chunk summarization.
- No web UI.
- No cloud embedding provider in the default path.

## License

See [LICENSE](./LICENSE).
