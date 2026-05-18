---
template_version: 2.1
adapter_for: gemini-cli
inherits: ./AGENTS.md
---

# GEMINI.md — context-rag (Gemini CLI adapter)

**Read [AGENTS.md](./AGENTS.md) first. It is the source of truth for this bee.**
**The vault root [AGENTS.md](../../../../AGENTS.md) is the source of truth for everything.**

## Hierarchy (read carefully)

1. **Vault root `AGENTS.md`** — non-overridable. Rules #0–13 apply
   everywhere, including inside this bee. A bee CANNOT relax safety,
   ledger discipline, signing, or destructive-op rules.
2. **Vault root `GEMINI.md`** — Gemini-specific vault defaults.
   Applies unless the bee explicitly overrides for files inside its scope.
3. **This file (`./GEMINI.md`)** — bee-specific Gemini context. Wins
   over vault-root `GEMINI.md` for files inside this bee, ONLY for
   scope-local matters: language choice, conventions, tool preferences.
4. **`./AGENTS.md`** (bee-level) — bee-specific rules and context.

In any conflict over **rules**, AGENTS.md (root) wins.
In any conflict over **scope-local context** for files inside this bee,
the bee-level adapter wins.

## Bee-specific Gemini notes

(Add bee-specific guidance here.)

- Language(s): python
- Test command: {{TEST_COMMAND}}
- Bee-specific style notes: {{NOTES}}

## Model self-knowledge

You do NOT know which Gemini model you are running unless
`$VAULT_AGENT_MODEL` is set or `02-agents/01-knowledge/CLI_REGISTRY.md`
documents it. Do not guess from training data — say "unknown" instead.
