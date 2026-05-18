---
template_version: 2.1
adapter_for: claude-code
inherits: ./AGENTS.md
---

# CLAUDE.md — context-rag (Claude Code adapter)

**Read [AGENTS.md](./AGENTS.md) first. It is the source of truth for this bee.**
**The vault root [AGENTS.md](../../../../AGENTS.md) is the source of truth for everything.**

## Hierarchy (read carefully)

1. **Vault root `AGENTS.md`** — non-overridable. Rules #0–13 apply
   everywhere, including inside this bee. A bee CANNOT relax safety,
   ledger discipline, signing, or destructive-op rules.
2. **Vault root `CLAUDE.md`** — Claude-specific vault defaults.
   Applies unless the bee explicitly overrides for files inside its scope.
3. **This file (`./CLAUDE.md`)** — bee-specific Claude context. Wins
   over vault-root `CLAUDE.md` for files inside this bee, ONLY for
   scope-local matters: language choice, conventions, tool preferences.
4. **`./AGENTS.md`** (bee-level) — bee-specific rules and context.

In any conflict over **rules**, AGENTS.md (root) wins.
In any conflict over **scope-local context** for files inside this bee,
the bee-level adapter wins.

## Bee-specific Claude notes

(Add bee-specific guidance here. Examples: preferred libraries, test
runner, file layout conventions that differ from vault default.)

- Language(s): python
- Test command: {{TEST_COMMAND}}
- Bee-specific style notes: {{NOTES}}

## Danish

User writes Danish. Claude responds in Danish for conversation, English
for code/commits/docs. Code comments: English.
