# Decisions

One entry per decision that someone might later question. Newest last.

## Carried over from a previous attempt

D-001 to D-004 came from an earlier setup of this project. All four were
re-checked on 2026-09-26 during Task 1 and still apply.

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

Re-verified 2026-09-26. `inference` 1.7.2 still pins supervision below 0.30
and pydantic below 2.12, while supervision 0.30.5 is out. `uv.lock` resolves
the `infer` extra with supervision 0.29.1 and pydantic 2.11.10.

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

## D-005: Support Python 3.11 to 3.13 only (2026-09-26)

Context. The project targets Python 3.11. `inference` 1.7.2 and its
`inference-models` dependency require Python below 3.14.

Decision. Set `requires-python = ">=3.11,<3.14"` and pin 3.11 in
`.python-version`.

Why. uv resolves one lock for every Python version the project allows, and
`inference` has no release that installs on 3.14.

Consequences. The project will not install on 3.14 until `inference`
supports it. Check the Colab Python version when the notebook lands.

## D-006: Exclude uv.lock from the large-file hook (2026-09-26)

Context. pre-commit rejects files over 500 KB so data and weights never get
committed. With the `infer` extra resolved, `uv.lock` is about 780 KB.

Decision. Exclude `uv.lock` from `check-added-large-files` by exact path.
Every other file keeps the limit.

Why. The lock is generated text that has to be committed (SPEC section 3.1).
It is not what the limit protects against.

Consequences. A large lock diff shows up in review instead of being blocked.

## D-007: Make supervision, roboflow, and imagehash core dependencies (2026-09-26)

Context. Task 2 downloads datasets with the `roboflow` SDK, draws sample
grids with `supervision`, and finds near-duplicate images with perceptual
hashes. Until now supervision only came in through the `infer` extra.

Decision. Add `roboflow>=1.5,<1.6`, `supervision>=0.29,<0.30`, and
`imagehash>=4.3,<5` to the core dependencies, along with `numpy` and
`pillow`, which the code imports directly.

Why. Download and inspection run without `inference`, so these packages
can't live in the extra. The supervision range matches what `inference`
pins (D-001), so the extra still co-resolves. `imagehash` is the standard
perceptual-hash implementation and only adds PyWavelets, since scipy is
already in the lock. numpy stays below 2.4 because `roboflow` 1.5.1 requires
it.

Consequences. `uv.lock` resolves roboflow 1.5.1, supervision 0.29.1, and
imagehash 4.3.2. `requirements.txt` grows, since Colab needs the same
packages to download data.

## D-008: Pin the candidate dataset versions (2026-09-26)

Context. SPEC section 5 lists five candidates with unverified slugs,
versions, and classes. `make download ARGS=--resolve` looked each one up on
2026-09-26 and wrote `reports/dataset_resolution.json`.

Decision. Pin the latest published version of each project:
`marswars` version 5, `testingfrfr` version 1, `scorekeeper` version 1,
`robotzftp2` (`rebuilt-dataset-hcmwl`) version 1, and a sixth key,
`robotzftp2_fuel` (`frc-2026-fuel-ndrbj`) version 1. Leave `lava`
unpinned.

Why. The `robot-zftp2` workspace holds three REBUILT projects.
`rebuilt-dataset-hcmwl` is the largest one with a published version.
`frc-2026-fuel-ndrbj` is the only other one with a version, so it is kept
as a separate key and inspection decides whether it is worth using.
`2026-rebuilt-ezy58` has no version, and its classes and image count match
`marswars` version 3, so it looks like a copy. The rest of the workspace is
unrelated to FRC. `lava` has no published version, and Roboflow only
exports versions.

Consequences. `make download` reports `lava` as a failure with the reason
"project or version not pinned". The spec's class list for `marswars` was
missing `red_active`. `testingfrfr` and `scorekeeper` versions hold far
more images than their source projects (see the resolution file), which
suggests augmented copies in the train split. Task 3 has to check this before
either is used for evaluation.
