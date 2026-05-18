---
generated_by: openai/gpt-5-codex
generated_at: 2026-04-30T05:35:46Z
agents_md_version: "2.1"
orchestrator: codex
task_id: lesson-project-vault-package-4-20260430
bee_name: "context-rag"
bee_version: 0.1.0
queen_template_version: "2.1"
inherits: ../../../../AGENTS.md
---

# AGENTS.md — context-rag

**Read the Queen's `AGENTS.md` at the vault root FIRST.**
Path: `../../../../AGENTS.md` (four levels up).

All vault-level rules apply. This file extends them with bee-specific
context and overrides.

---

## Bee identity

- **Name**: context-rag
- **Bucket**: 02-maintained  (01-active | 02-maintained | 03-archive)
- **Status**: paused   (active | paused | done | archived)
- **Primary language(s)**: python
- **One-line purpose**: standalone hybrid RAG (BM25+dense+RRF) over markdown corpora

## What this bee does

_(2-3 sentences. Replace before committing.)_

## Bee-specific rules

_(Add rules here that apply ONLY to this bee. If a rule would apply to
all bees, propose it for Queen's AGENTS.md instead.)_

- Example: "This bee uses Python 3.12+. Do not introduce syntax
  incompatible with 3.12."
- Example: "All API calls must go through src/api_client.py — do not
  call requests directly."
- Example: "Tests must run in <5 seconds. Slow tests go to a separate
  suite under tests/slow/."

_(Delete the examples when you add real rules.)_

## Cost-tier overrides

_(Most bees inherit vault-level `03-common/00-config/routing.yaml`.
Override here with `.bee-config.yaml` only
if this bee has special needs.)_

Example override:
- This bee requires reasoning that typically works best on Tier 2.
  Route all tasks here to Claude/Gemini-Pro by default.

_(Delete if no overrides.)_

## Context pointers

- State: `memory/CONTEXT.md`
- Prior learnings: `memory/lessons.jsonl`
- Side observations: `memory/observations.jsonl`
- Issues: use GitHub Issues on this bee's remote; cross-bee issues
  go to Queen's `02-agents/01-knowledge/ISSUES.md`

## Invocation

Use the Queen's wrapper to ensure AGENTS.md (Queen + bee) is loaded
as system prompt:

```bash
# From inside this bee directory:
bash ../../../../02-agents/03-tools/ask.sh claude "your task"
```

The wrapper handles AGENTS.md injection. Do not call CLIs directly
without the wrapper — bypassing it loses Queen's rules.

## When done with a task

1. Update `memory/CONTEXT.md` (STATUS, NEXT, CHANGED)
2. Append lesson to `memory/lessons.jsonl` if non-trivial
3. Append observations if you noticed something outside task scope
4. Commit (bee has its own git; the vault tracks only pollen files)

## Running tests / CI

_(Document how to run tests for this bee.)_

```bash
# Example:
# make test
# pytest tests/
# npm test
```

## Deployment

_(If this bee gets deployed somewhere, describe how. Otherwise delete
this section.)_

---

*Template version: 2.1. When the vault's bee standard changes, this
file may be updated via `bash 02-agents/03-tools/retrofit-bees.sh`.*
