# Decisions

One entry per decision that someone might later question. Newest last.
Write these yourself or edit Claude's drafts into your own words, since
reviewers read this file to see how you think.

## Carried over from a previous attempt, re-verify during Task 1

These came from an earlier setup of this project. Library versions may have
moved since, so check each one against what is installed before relying on it
(rule 5).

## D-001: Use supervision 0.29.x and make inference optional (2026-09-26)

Context. `inference` 1.7.x requires `supervision>=0.29,<0.30` and
`pydantic<2.12`, so the latest supervision (0.30) cannot be installed next to
it.

Decision. Use supervision 0.29.x. Keep runtime ranges in `pyproject.toml`
loose so the `infer` extra co-resolves, and let `uv.lock` hold the exact pins.
Ship `inference` as an optional extra (`uv sync --extra infer`).

Why. Evaluation and edge deployment both need `inference`, so its pins win
over having the newest supervision. Making it optional keeps CI light and
offline.

Consequences. Anything written against supervision has to use the 0.29 API.
Code that needs `inference` only works after `uv sync --extra infer`.

## D-002: Add console scripts with their modules (2026-09-26)

Context. `pyproject.toml` and the Makefile could list every planned console
script and pipeline target up front.

Decision. Add each console script and its Make target in the same commit as
the module it runs, never as a stub.

Why. A stubbed entry point is broken until its module lands, and a reader
cloning the repo mid-way would hit it.

Consequences. The Makefile and `pyproject.toml` grow task by task, and every
target that exists works.

## D-003: Read package versions with importlib.metadata (2026-09-26)

Context. Run metadata records package versions (SPEC section 3.5), which the
spec describes as `uv pip freeze`.

Decision. Read them with `importlib.metadata`.

Why. It gives the same information without spawning a subprocess or
depending on `uv` being on the path at run time.

Consequences. Metadata still lists every installed distribution and its
version, in a format the package controls.

## D-004: Rebase-merge only, with branch protection after first CI run (2026-09-26)

Context. SPEC section 3.2 asks for atomic commits landing linearly on `main`
and merges only with CI green.

Decision. Limit GitHub merges to rebase-merge. Add branch protection that
requires the `lint, typecheck, test` check on `main` once that check has run
at least once.

Why. GitHub only offers a check as required after it has reported on the
repo, so protection has to wait for the first CI run.

Consequences. Between the first PR and enabling protection, CI green is
enforced by habit only.
