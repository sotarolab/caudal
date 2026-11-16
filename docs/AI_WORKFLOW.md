# How this repository is built with Claude

Status: proposal, written 2026-11-15, revised 2026-11-16, not yet approved.

Most of the code here is written by Claude Code, an AI coding agent, working under the rules on
this page. The method and its core functions are written by the repository's owner. This page
says which is which, how the work moves, and how to check both.

## Who writes what

| Part | Written by | How to check |
|---|---|---|
| `caudal/science/`: the loss, the quantile head, the windows, the scores, the map logic, the resampling, the routing baseline | The owner | `git log caudal/science` shows no Claude co-author line; CI fails a pull request where one appears |
| `tests/science/`: the tests of those functions | Drafted by Claude with `/scaffold`, approved by the owner, who moves them into place | Same |
| Everything else: data loaders, features, training plumbing, products, scripts, CI, documentation | Claude, reviewed by the owner | Commits end with the Claude co-author line |
| The architecture, the decisions, the scoring plan of a frozen model | The owner decides; Claude drafts on request | `docs/decisions/`, each dated |

## The loop

Every change follows the same seven steps, and each step leaves a trace.

1. **Plan.** Anything beyond a one-file fix starts as a plan the owner approves.
2. **Issue.** `/issue` opens one GitHub issue from the template: goal, done-when, the tests to
   write, out of scope, milestone, label. The owner approves the test list by leaving it in
   the issue.
3. **Branch.** Work happens on `<issue number>-<slug>`, in the open checkout, one stage at a
   time, each ending with the diff in view. The Stop hook runs `make qc` whenever Claude ends a
   turn that changed Python.
4. **Pull request.** `/pr` runs `make qc`, then the `ponytail-review` skill, writes the pull
   request from the template (what and why, tests, review findings and their outcome, and what
   was not checked), pushes the branch and opens the pull request on its milestone, then runs
   the built-in code review so its findings land as inline comments.
5. **Checks.** CI runs `make qc`, the authorship check and the documentation build on the pull
   request. The default branch is protected: no merge without a pull request and green checks.
6. **Merge.** The owner reads the reviews and merges. Claude never merges.
7. **Decision.** Anything decided in review is recorded the same day with `/adr`.

A milestone closes with `/milestone`: the `ponytail-audit` skill over the repository, the
`ponytail-debt` ledger of `shortcut:` comments, their findings filed as issues in the next
milestone, release notes from the merged pull requests, and a tag.

## The tools

| Tool | What it does | Where |
|---|---|---|
| `CLAUDE.md` | The standing rules Claude reads at the start of every session | Repository root |
| `/issue`, `/pr`, `/milestone` | The loop above, as skills | `.claude/skills/` |
| `/scaffold` | Prepares a core function for the owner: signature, docstring with the equation and reference, failing tests. Never the body | `.claude/skills/scaffold/` |
| `/adr` | Writes a one-page decision record the day a decision is made | `.claude/skills/adr/` |
| Templates | The issue and the pull request have fixed sections, so a missing one is visible | `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md` |
| Deny rules | Refuse Claude's edits under `caudal/science/` and `tests/science/`, and refuse `gh pr merge` and force pushes | `.claude/settings.json` |
| Stop hook | Runs `make qc` before a turn ends when Python files changed; a failure goes back to Claude | `.claude/hooks/qc-on-stop.sh` |
| ponytail | A review plugin: the review of each change, the audit of the whole repository at each milestone, and the ledger of `shortcut:` comments | Claude Code plugin |
| CI | `make qc`, the authorship check and the documentation build on every pull request and every merge | `.github/workflows/` |
| Branch protection | No direct push to the default branch, no merge with a red check | GitHub setting, applied by the owner |

## What a reader can check, and where

| Question | Answer lives in |
|---|---|
| Who wrote this line? | The commit's co-author line; the authorship check on `caudal/science/` |
| Why is it built this way? | `docs/decisions/`, and the issue the pull request closes |
| Was it tested? | The tests named in the issue, green in the pull request's checks |
| Was it reviewed? | The review findings in the pull request body, the inline comments, the merge |
| What was skipped? | The Not checked section of every pull request |
| What shortcuts exist? | `shortcut:` comments in the code, listed at each milestone |
| When was a forecast issued, by which model? | The CI commit on hapi's `data` branch, and `forecasts/<model>/MODEL.json` |

## Limits, stated plainly

- The deny rules stop Claude's own tools, including shell redirects. A script Claude writes
  could still write to a protected path. `CLAUDE.md` forbids that, and the authorship check
  would show the commit.
- The Stop hook runs only when uncommitted Python files exist, so it does not catch a broken
  commit. CI does.
- A reviewer that is also an AI shares blind spots with the author. The owner's review and the
  public scores in hapi are the checks that do not.
- Tests show that code does what the tests say. Whether the tests say the right thing is the
  owner's judgment, which is why the owner approves them first.
