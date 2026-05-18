---
template_version: 2.1
adapter_for: codex-cli
inherits: ./AGENTS.md
---

# CODEX.md — context-rag (Codex CLI adapter)

**Read [AGENTS.md](./AGENTS.md) first. It is the source of truth for this bee.**
**The vault root [AGENTS.md](../../../../AGENTS.md) is the source of truth for everything.**

## Hierarchy (read carefully)

1. **Vault root `AGENTS.md`** — non-overridable. Rules #0–13 apply
   everywhere, including inside this bee. A bee CANNOT relax safety,
   ledger discipline, signing, or destructive-op rules.
2. **Vault root `CODEX.md`** — Codex-specific vault defaults.
   Applies unless the bee explicitly overrides for files inside its scope.
3. **This file (`./CODEX.md`)** — bee-specific Codex context. Wins
   over vault-root `CODEX.md` for files inside this bee, ONLY for
   scope-local matters: language choice, conventions, tool preferences.
4. **`./AGENTS.md`** (bee-level) — bee-specific rules and context.

In any conflict over **rules**, AGENTS.md (root) wins.
In any conflict over **scope-local context** for files inside this bee,
the bee-level adapter wins.

## Bee-specific Codex notes

(Add bee-specific guidance here.)

- Language(s): python
- Test command: {{TEST_COMMAND}}
- Bee-specific style notes: {{NOTES}}
