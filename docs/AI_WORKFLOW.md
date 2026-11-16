# How this repository is built with Claude

Status: proposal, written 2026-11-15, not yet approved.

Most of the code here is written by Claude Code, an AI coding agent, working under the rules on
this page. The method and its core functions are written by the repository's owner. This page
says which is which and how to check it.

## Who writes what

| Part | Written by | How to check |
|---|---|---|
| `caudal/science/`: the loss, the scores, the map logic, the resampling, the routing baseline | The owner | `git log caudal/science` shows no Claude co-author line |
| `tests/science/`: the tests of those functions | Drafted by Claude, approved by the owner, who moves them into place | Same |
| Everything else: data loaders, features, training plumbing, scripts, CI, documentation | Claude, reviewed by the owner | Commits end with the Claude co-author line |
| The architecture, the decisions, the pre-registration | The owner decides; Claude drafts on request | `docs/decisions/`, each dated |

## The four gates

1. **An approved architecture.** `docs/ARCHITECTURE.md` fixes the layers and what may import
   what. A test enforces it, and changing it takes a decision record.
2. **A plan before code.** Each piece of work starts as a written plan the owner approves.
3. **Tests as the contract.** The owner approves the tests of a core function before its body
   exists. Claude's settings refuse edits to the core and to its approved tests.
4. **Automatic checks and a second reviewer.** `make qc` runs when Claude finishes a turn that
   changed code, and in CI on every pull request. A separate review pass reads each change before
   the owner does. The owner merges.

## The tools

| Tool | What it does | Where |
|---|---|---|
| `CLAUDE.md` | The standing rules Claude reads at the start of every session | Repository root |
| `/scaffold` | Prepares a core function for the owner: signature, docstring with the equation and reference, failing tests. Never the body | `.claude/skills/scaffold/` |
| `/adr` | Writes a one-page decision record the day a decision is made | `.claude/skills/adr/` |
| Deny rules | Refuse Claude's edits under `caudal/science/` and `tests/science/` | `.claude/settings.json` |
| Stop hook | Runs `make qc` before a turn ends when Python files changed; a failure goes back to Claude | `.claude/hooks/qc-on-stop.sh` |
| ponytail | A review plugin: a review of each change, an audit of the whole repository before a release, and a ledger of `shortcut:` comments | Claude Code plugin |
| CI | `make qc` on every pull request and every merge | `.github/workflows/qc.yml` |

## Limits, stated plainly

- The deny rules stop Claude's file-editing tools. A shell command could still write to a
  protected path. `CLAUDE.md` forbids that, and the commit history would show it.
- The Stop hook runs only when uncommitted Python files exist, so it does not catch a broken
  commit. CI does.
- A reviewer that is also an AI shares blind spots with the author. The owner's review and the
  public scores in hapi are the checks that do not.
- Tests show that code does what the tests say. Whether the tests say the right thing is the
  owner's judgment, which is why the owner approves them first.
