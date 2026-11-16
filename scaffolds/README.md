# scaffolds

Staging for the core functions. `/scaffold` writes a function's signature, docstring and failing
tests here. The owner moves them into `caudal/science/` and `tests/science/` and writes the body.
Claude cannot write to those two folders (`.claude/settings.json`). Nothing here is imported by
the package or collected by the tests.
