# caudal

The forecasting package behind [hapi](https://github.com/sotarolab/hapi): probabilistic forecasts
of river flow, reservoir inflow and flood extent for California, written as files into hapi and
scored there in public against the official forecasts. The design of record is
`docs/ARCHITECTURE.md`, decisions are `docs/decisions/`, and how Claude is used here is
`docs/AI_WORKFLOW.md`.

## Rules

- **One path.** The function that issues today's forecast is the function the backtest calls.
- **Issued once, never revised.** A changed model is a new product with a new name and a new
  manifest.
- **A point is a row** in `caudal/points.py`. No per-point code.
- **The science is written by a person.** Never write or edit a function body in
  `caudal/science/`, or a test in `tests/science/`. The settings deny it; do not work around the
  rule with a shell command. Use `/scaffold` to prepare a function for the owner.
- **Layers import downward only** (`docs/ARCHITECTURE.md`). `science` does no I/O; a `sources`
  module computes no result.
- **Published methods only.** Any hydrologic, hydrodynamic or routing method must be one that can
  be cited, and the docstring cites it.
- **Numbers exist only as script output.** A result is a row written by a script at a commit.
  Nothing is quoted from a notebook or from memory.
- **A change ships with its test** in the same pull request.
- **A known shortcut is marked** with a `shortcut:` comment that states the limit and when to
  lift it. `ponytail-debt` lists them.
- **Plain prose**: no em dashes, no hedging, no claim without a number or a reference behind it.
  Keys and credentials never enter the repository. The repository is public.

## Commands

    pip install -e ".[dev]"            # the conda environment is still named embalse
    make qc                            # ruff + pytest: what CI runs and what the Stop hook runs
    python scripts/train.py            # trains the daily inflow model; writes data/models, data/results
    python scripts/issue.py            # issues today's forecasts into ../hapi/data (or HAPI_DATA)

Data lives under `data/` at the repository root (or `CAUDAL_DATA`) and is not in git.

## Process

The owner owns the process.

- **Plan first.** Anything beyond a one-file fix starts as a plan the owner approves.
- **Small pull requests, one layer each**, opened as an issue plus a linked pull request. The
  owner merges.
- **Pushing.** Commit to a branch and stop. The owner pushes and opens the pull request, unless
  they have said in the session that Claude may.
- **Review before the owner reads.** Run `ponytail-review` on the branch and put its findings in
  the pull request description. Run `ponytail-audit` before a release.
- **Decisions are recorded the day they are made**, one page, with `/adr`.
- **Attribution.** A commit Claude wrote ends with the Claude co-author line. A commit that
  touches only `caudal/science/` or `tests/science/` is the owner's and carries none.
- **Say what was not checked.** End a piece of work with what was skipped and what is uncertain.
