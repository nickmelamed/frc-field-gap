---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "notebooks/**"
---

# Python standards

- Python 3.11, managed with `uv`. `uv.lock` is committed, and
  `requirements.txt` is exported from it for Colab.
- `ruff` for lint and format with rules `E, F, W, I, B, UP, N, SIM, RUF, D`
  (Google docstring convention), line length 100.
- `mypy --strict` on `src/`. `# type: ignore` only with an error code and a
  reason.
- Type hints everywhere, including return types. Frozen dataclasses or
  Pydantic models for configs and records, not loose dicts.
- Small pure functions for logic (mapping, splitting, metrics), with I/O at
  the edges, so core logic is testable without data or network.
- `logging` with one logger per module, never `print`. CLI entry points take
  `--log-level`.
- Fail loudly and specifically with the exceptions in
  `src/frc_xdata/errors.py` (`UnmappedLabelError`, `DataLeakError`,
  `ConfigError`, and so on). No bare `except`, and never swallow an exception.
- `pathlib` for paths. No hardcoded absolute paths.
- Every tunable (dataset versions, splits, seeds, thresholds, model IDs) lives
  in `configs/*.yaml` and is validated on load. No magic numbers.
- CLIs use `argparse` or `typer` and are exposed as console scripts and Make
  targets. Add a console script and its Make target in the same commit as the
  module it runs, never as a stub.
- Notebooks are thin and call the package. Outputs are stripped by
  nbstripout. A rendered copy with outputs may be published outside the repo,
  as a gist or release asset, after the same face check as any published
  image.
- Run metadata, manifests, and seeds follow SPEC section 3.5. Refuse to write
  results from a dirty tree unless `--allow-dirty` is passed, and flag it.
