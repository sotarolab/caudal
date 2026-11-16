---
name: milestone
description: Open or close a GitHub milestone. Closing runs the ponytail audit and the shortcut debt ledger, files their findings as issues in the next milestone, writes release notes from the merged pull requests and proposes a tag. Use when the user says "/milestone", "close the milestone", "set up the milestones", "what is left in M1".
---

# Milestone

A milestone is a promise with a date and a done-when. Closing one is an audit, not a
celebration: what did it leave behind?

## Open

1. `gh api -X POST repos/{owner}/{repo}/milestones -f title=... -f due_on=... -f description=...`
   The description is the done-when, one to three observable sentences.
2. Nothing else. Issues are opened with `/issue` when their work starts.

## Status

`gh issue list --milestone "<title>" --state all` and `gh pr list --search 'milestone:"<title>"' --state all`.
Report open, merged and unmerged in one table, and the done-when next to what is true today.

## Close

1. **Done-when.** Check each sentence of the milestone's description against the repository
   and the live record. A sentence that is not true is reported, and the milestone does not
   close until the owner says it does.
2. **Audit.** Run the `ponytail-audit` skill on the whole repository. Each finding worth
   fixing becomes one issue with `/issue`, label `audit`, in the next milestone.
3. **Debt.** Run the `ponytail-debt` skill. Each `shortcut:` comment whose trigger has arrived,
   and each one with no trigger, becomes one issue, label `debt`, in the next milestone. A
   shortcut whose trigger has not arrived is only listed.
4. **Release notes.** From the merged pull requests of the milestone: one line each, grouped by
   label, with the pull request number. Write them to `docs/releases/<tag>.md` and show them.
5. **Tag.** Propose the version (semantic: a new product or a changed contract raises the
   minor number before 1.0). The owner tags and pushes the tag; the release workflow does the
   rest.
6. Show the summary and wait for a yes, then close the milestone:
   `gh api -X PATCH repos/{owner}/{repo}/milestones/<number> -f state=closed`.

## Rules

- Findings are filed, not fixed, during a close. Fixing happens on an issue branch.
- No number in the release notes that is not in a file written by a script at a commit.
- Plain language, no em dashes. The repository is public.
