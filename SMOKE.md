---
generated_by: openai/gpt-5-codex
generated_at: 2026-05-18T15:39:57Z
agents_md_version: "3.0"
orchestrator: codex
---

# Smoke Test: Programmering af AI Corpus

Corpus: `programmering-af-ai.md` from context-compiler output.
Size: 512 KB.
Environment: Termux / Android, Python 3.13.
Workspace: `/data/data/com.termux/files/usr/tmp/cr-smoke`.

Note: `/tmp/cr-smoke` could not be created in this environment because `/tmp`
is not writable by this app user. The same commands were run from Termux's
writable `$TMPDIR`.

## Commands

```bash
PYTHONPATH=/storage/emulated/0/vaultBU/03-common/03-repositories/02-maintained/context-rag/src python -m context_rag.cli init
PYTHONPATH=/storage/emulated/0/vaultBU/03-common/03-repositories/02-maintained/context-rag/src python -m context_rag.cli index .
PYTHONPATH=/storage/emulated/0/vaultBU/03-common/03-repositories/02-maintained/context-rag/src python -m context_rag.cli query "hvad er RAG" --k 5
```

## Metrics

- Indexing time: 0.885 seconds.
- Indexed files: 1.
- Indexed chunks: 728.
- Query latency: 0.531 seconds.
- Embeddings used: `local-hashing-1024` fallback because
  `sentence-transformers`, `torch`, and `numpy` are not installed in this
  Termux session.

## Top-5 Sanity Check

Query: `hvad er RAG`

1. `We Tried and Tested 10 Best Vector Databases for RAG Pipelines > Our Evaluation Criteria to Pick the Top Vector Databases for RAG Pipelines`
2. `We Tried and Tested 10 Best Vector Databases for RAG Pipelines > Building Advanced Search, Retrieval, and Recommendation Systems with LLMs > We Tried and Tested 10 Best Vector Databases for RAG Pipelines - ZenML Blog`
3. `Run a responses API call > Hvad betyder felterne?`
4. `Run a responses API call > Modul 3 - Chatbot (prompt engineering) > Manuel historik`
5. `FelderBot - ChatGPT-samtale med Blazor og OpenAI Responses API > Languages > We Tried and Tested 10 Best Vector Databases for RAG Pipelines - ZenML Blog`

Assessment: top-5 is partly relevant. Results 1, 2, and 5 are clearly RAG
related; result 3 matches the Danish question wording; result 4 is weaker and
is likely from the local hashing fallback. BM25-only top-5 was strongly RAG
focused after stopword filtering.
