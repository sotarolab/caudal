---
name: adr
description: Write a one-page, dated architecture decision record in docs/decisions/ in the same session the decision is made. Use when the user says "/adr", "write the ADR", "record this decision", or after a decision is taken in review and the user asks to record it.
---

# ADR

One decision, one page, written the day it is made. Never backfilled from memory.

## File

`docs/decisions/NNNN-<kebab-title>.md`, where NNNN is the next number (start at 0001).
Create `docs/decisions/` if it does not exist.

## Template

```markdown
# NNNN. <Title in the imperative, e.g. "Key observations by received_at">

Date: YYYY-MM-DD
Status: accepted | superseded by NNNN

## Context
What forced the decision. Two to four sentences. Point at the plan or the code.

## Decision
What was decided, stated so it can be checked against the code.

## Consequences
What becomes easier, what becomes harder, what must now exist (a test, a column, a gate).

## Alternatives rejected
One line each, with the reason.
```

## Rules

- Under 300 words. If it needs more, it is two decisions.
- No metrics in prose. A number belongs in a table written by a script at a commit.
- Plain language, no hedging words, no em dashes. The repo is public.
- If the decision changes `docs/ARCHITECTURE.md`, edit that page in the same turn.
- Tell the user the file path and ask nothing. They commit it themselves.
