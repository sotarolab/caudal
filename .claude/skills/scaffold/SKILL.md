---
name: scaffold
description: Scaffold a core science function for the owner to write. Produces the signature, a docstring carrying the equation and its reference, and failing tests, in scaffolds/, and never the body. Use when the user says "/scaffold <function>", "scaffold the pinball loss", or asks for a core function in caudal/science/ to be set up for them to write.
---

# Scaffold

The functions in `caudal/science/` are written by the repository's owner. This skill prepares one
for them. It never writes the body.

## What to produce

Two files under `scaffolds/`, mirroring where they will live:

- `scaffolds/science/<module>.py`: the function with its full signature and type hints, a NumPy
  style docstring, and a body that is exactly `raise NotImplementedError`.
- `scaffolds/tests/science/test_<module>.py`: tests that fail now and pass once the body is right.

Claude cannot write to `caudal/science/` or `tests/science/`: `.claude/settings.json` denies it.
The owner moves the two files there, which is the act of approving the tests, and then writes the
body.

## The docstring

- One sentence saying what the function computes and when it is used.
- The equation, in LaTeX between two lines of `$$`, with every symbol defined. The
  documentation site renders `$$` in docstrings; it does not render `.. math::` blocks.
- Parameters and returns with shapes and units.
- `References`: the paper the equation comes from, with year and equation number. If the
  reference is from memory and not checked, say so in a `TODO(owner): confirm reference` line.
- `TODO(owner)` notes for each real design decision the body has to take (an interpolation rule,
  a tie-break, what to do with missing values). State the options; do not choose.

## The tests

- A hand-computed case small enough to check on paper, with the arithmetic in a comment.
- The edge cases the docstring names: empty input, one member, missing values, ties.
- A property the method must satisfy (a score is zero for a perfect forecast; an extent never
  shrinks as the stage rises; a probability lies in [0, 1] and falls as the threshold rises).
- Where an independent implementation exists (scipy, properscoring, hapi's `score.py`), a test
  that the two agree.
- No test that only restates the implementation.

## Rules

- Never write, complete, or suggest the body, in the file or in chat, before the owner has made
  an attempt. After an attempt, review it: say what is wrong and why, not the corrected line.
- If the owner asks for the answer outright, give it. It is their repository. Say that the
  commit should then carry the Claude attribution line.
- One function per invocation. Finish by telling the owner the two file paths and the `git mv`
  commands that move them into place.
- Plain language, no em dashes.
