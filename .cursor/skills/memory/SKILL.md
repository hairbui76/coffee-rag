---
name: memory
description: >-
  Persistent project memory that tracks decisions, progress, and context across
  sessions. MUST read at session start. MUST update after any meaningful change
  (new feature, bug fix, architecture decision, config change, blockers).
---

# Project Memory

## When to read

Read `MEMORY.md` (in this same directory) at the **start of every session** or
whenever you need project context you may have lost between conversations.

## When to update

Append to `MEMORY.md` immediately after any of these events:

- A task or phase is completed (e.g. "P0 data cleaning done")
- An important decision is made (e.g. "Chose FAISS over ChromaDB because…")
- A new file/module is created or significantly refactored
- A bug is found and fixed
- A blocker or open question is identified
- Config / environment changes (new dependency, env var, API key setup)
- The user explicitly asks to remember something

## Format

Each entry in `MEMORY.md` follows this template:

```
### YYYY-MM-DD — Short title

- **Category:** decision | progress | bugfix | blocker | config | note
- **Details:** 1-3 sentences describing what happened and why.
- **Files:** list of key files touched (if applicable)
```

## Rules

1. **Never delete** existing entries — memory is append-only.
2. Keep each entry **concise** (≤ 5 lines). Link to files instead of pasting code.
3. Use **newest-first** order (latest entry at the top, after the header).
4. If `MEMORY.md` doesn't exist yet, create it with the header shown below.

## MEMORY.md header template

```markdown
# Project Memory — Coffee RAG

> Auto-maintained by the memory skill. Newest entries first.

---
```
