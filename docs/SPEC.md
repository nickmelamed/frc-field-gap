# FRC REBUILT cross-dataset generalization: design spec

Design spec for the project. Section numbers match the original brief and are referenced from code and docs.

---

## 1. Project in one paragraph

Game-piece and robot detection for the 2026 FRC game (REBUILT), built around **cross-dataset generalization**. We train a detector on one team's public Roboflow Universe dataset, show how it degrades on other teams' footage, diagnose why, fix it by harmonizing and merging datasets, then deploy to a Raspberry Pi with measured edge performance. A held-out **field test set** of our own REBUILT footage (FRC Team 1515, MorTorq) is used only for evaluation, never training.

**Narrative arc:** baseline → generalization failure → diagnosis → fix → edge performance.

**Audience:** Roboflow hiring teams (Forward Deployed Engineer, Implementation Engineer, Customer Success Engineer) and FRC students/teams who want to reuse the work. Write every doc as if a student or a customer's engineer will follow it.

**Hard deadline:** a public repo with baseline + cross-dataset results and a draft write-up, tagged `v0.1.0`, by **Wed Sept 30, 2026, 11:59pm PT**. Edge deployment, video, and polish follow in later releases.

---

## 2. Non-negotiable rules

Moved to CLAUDE.md.

---

## 3. Engineering standards

These apply to every change. They exist so that a reviewer skimming the repo sees production habits, and so that every result is reproducible.

### 3.1 Tooling (set up in Task 1)

| Concern | Tool | Notes |
|---|---|---|
| Environment + lockfile | `uv` (`pyproject.toml` + `uv.lock`) | Python 3.11. Lockfile is committed. `requirements.txt` is exported from the lock for Colab. |
| Lint + format | `ruff` (lint and `ruff format`) | Rules: `E, F, W, I, B, UP, N, SIM, RUF, D` (Google docstring convention). Line length 100. |
| Type checking | `mypy --strict` on `src/` | No untyped defs. `# type: ignore` only with an error code and a reason comment. |
| Tests | `pytest` + `pytest-cov` | Coverage target: ≥ 80% on `src/` core logic (harmonize, merge, splits, metrics). |
| Hooks | `pre-commit` | ruff, ruff-format, mypy, end-of-file/trailing-whitespace, check-yaml, check-added-large-files (500 KB), `gitleaks`, `nbstripout`. |
| CI | GitHub Actions | On every push and PR: install from lock, ruff, mypy, pytest with coverage. Status badge in README. CI must never need the API key or network data. |
| Task runner | `Makefile` | `make setup`, `make lint`, `make typecheck`, `make test`, `make check` (all three), and one target per pipeline stage (`make download`, `make inspect`, `make harmonize`, `make eval`, `make report`). |

Pin tool versions in `pyproject.toml` and `.pre-commit-config.yaml`. Look up current versions at setup time.

### 3.2 Git workflow and commits

Moved to docs/CONTRIBUTING.md.

### 3.3 Code style

Moved to `.claude/rules/python.md`.

### 3.4 Testing

- **Unit tests use tiny synthetic fixtures** (a handful of fake images with hand-written COCO annotations under `tests/fixtures/`), never downloaded data.
- Required coverage of behavior:
  - Class mapping: every label maps correctly; an unmapped label raises `UnmappedLabelError`; `DROP` removes annotations.
  - Field-test leak check: a leaked hash raises `DataLeakError`.
  - Splits: deterministic given a seed; no image in two splits.
  - Metrics wrapper: known-answer cases (perfect predictions → mAP 1.0; no predictions → recall 0; a known FP/FN mix → expected precision/recall).
  - Class coverage: a dataset without `robot` labels is not scored on `robot`.
- **Integration tests** that need the network or real data are marked `@pytest.mark.integration` and skipped in CI by default.

### 3.5 Reproducibility

Every number in the README must be regenerable from a clean clone with the API key and a documented command.

- **Pinned inputs:** exact Universe dataset versions in `configs/datasets.yaml`. After download, write `data/raw/<key>/MANIFEST.json` (file list + SHA256) and commit a small `reports/data_manifests/<key>.sha256` so drift is detectable.
- **Seeds:** one global seed in config, applied to Python, NumPy, and any framework in use. Splits are derived deterministically (seeded, or hash-based on filename).
- **Run metadata:** every eval or training record writes `reports/runs/<run_id>/meta.json` containing: git commit SHA and dirty flag, UTC timestamp, config file contents (or hash), dataset keys + versions + manifest hashes, model ID/architecture/input size, package versions (`uv pip freeze`), and hardware (CPU/GPU, or Pi model later). Refuse to write results from a dirty working tree unless `--allow-dirty` is passed, and flag it in meta.
- **Platform-trained models:** Roboflow platform training is not fully controllable, so record everything available (model ID, version, architecture, preprocessing, augmentation, training settings shown in the UI) in `reports/models.yaml`, and say plainly in docs what isn't reproducible bit-for-bit.
- **Generated results table:** `make report` builds the README results table (between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` markers) and `docs/EVALUATION.md` tables from `reports/`. Never edit those sections by hand.
- **Figures** are produced by scripts into `docs/assets/`, small PNGs only, with the generating command noted in the doc that uses them.

### 3.6 Definition of done (every PR)

Checked by the `/finish-task` skill (`.claude/skills/finish-task/SKILL.md`).

---

## 4. Class schema

Two classes: **`fuel`** and **`robot`**.

Source datasets use inconsistent names (`fuel`, `Fuel`, `FUEL`, `2026-FRC-Fuel-V2`, `game_piece`, etc.). Every source label maps to `fuel`, `robot`, or `DROP` via `configs/class_map.yaml`. An unmapped label is a hard error, never a silent drop. Labels like `blue_active` / `inactive` need inspection before mapping, since they may encode state rather than object type. Document the choice in `docs/DATASETS.md`.

Datasets that label only one class are usable, but **evaluation is per-class**, and a dataset with no `robot` labels is not scored on `robot` (unlabeled robots would count as false positives). Class coverage per dataset is recorded in `reports/class_coverage.json`.

---

## 5. Candidate datasets

All on Roboflow Universe, CC BY 4.0. Sizes and classes are approximate and must be verified.

| Key | Universe slug | Approx. size | Classes (unverified) |
|---|---|---|---|
| `marswars` | `marswars-robotics-program/2026-rebuilt` | ~1.3k | blue_active, game_piece, inactive |
| `lava` | `lava/frc-2026-aagvs` | ? | fuel variants |
| `testingfrfr` | `testing-frfr/frc-2026-mldc0` | ~13k | robot, fuel |
| `scorekeeper` | `blind-assistant-model/frc-scorekeeper-2026` | ~5k | robot, FUEL |
| `robotzftp2` | workspace `robot-zftp2` (several projects) | ? | REBUILT/Fuel |

Pick 2–3 for the main experiment based on Task 2's inspection, favoring maximum domain shift in camera angle, venue, lighting, and labeling style. Record the chosen datasets and exact versions in `configs/datasets.yaml`.

---

## 6. Repo layout

```
.
├── CLAUDE.md
├── README.md                    # pitch, CI badge, generated results table, attribution
├── CHANGELOG.md                 # Keep a Changelog
├── LICENSE                      # MIT for code
├── pyproject.toml               # deps, tool config (ruff, mypy, pytest), console scripts
├── uv.lock
├── requirements.txt             # exported from uv.lock, for Colab
├── Makefile
├── .pre-commit-config.yaml
├── .gitignore                   # data/, runs/, weights, .env, caches, checkpoints
├── .env.example                 # ROBOFLOW_API_KEY=
├── .github/
│   ├── workflows/ci.yml
│   └── pull_request_template.md
├── configs/
│   ├── project.yaml             # seed, paths, default thresholds
│   ├── datasets.yaml            # workspace, project, version, format per key
│   └── class_map.yaml           # per-dataset source label -> fuel | robot | DROP
├── src/frc_xdata/
│   ├── __init__.py
│   ├── config.py                # typed, validated config loading
│   ├── errors.py
│   ├── logging_utils.py
│   ├── provenance.py            # run metadata, manifests, git SHA
│   ├── download.py
│   ├── inspect_datasets.py
│   ├── harmonize.py
│   ├── splits.py
│   ├── merge.py                 # includes field-test leak check
│   ├── evaluate.py
│   └── report.py                # builds README/EVALUATION tables from reports/
├── tests/
│   ├── fixtures/                # tiny synthetic COCO datasets
│   ├── test_harmonize.py
│   ├── test_merge.py
│   ├── test_splits.py
│   └── test_metrics.py
├── notebooks/
│   └── train_and_eval.ipynb     # thin, Colab-ready, calls the package
├── docs/
│   ├── DATASETS.md
│   ├── EVALUATION.md
│   ├── DEPLOYMENT.md            # later
│   ├── RETRAINING.md
│   ├── MODEL_CARD.md            # intended use, data, metrics, limits
│   └── assets/                  # small generated PNGs only
├── reports/                     # committed small JSON/CSV/YAML outputs + run metadata
└── data/                        # gitignored
    ├── raw/
    ├── harmonized/
    └── field_test/              # our own REBUILT footage, eval-only
```

---

## 7. Task plan

One branch and one PR per task unless noted. Each PR meets the Definition of Done.

### Task 1 — Foundation (Sat Sept 26)
- Initial commit on `main`: `.gitignore`, `LICENSE`, minimal README, `CLAUDE.md`.
- Then on `chore/tooling`: `pyproject.toml` (uv, ruff, mypy, pytest config, console scripts), `uv.lock`, exported `requirements.txt`, `.pre-commit-config.yaml`, `Makefile`, `.github/workflows/ci.yml`, PR template, `CHANGELOG.md`, `.env.example`, package skeleton with `config.py`, `errors.py`, `logging_utils.py`, `provenance.py`, and one trivial passing test so CI goes green.
- README skeleton: pitch, CI badge, narrative arc, results-table markers with placeholder, empty attribution section.
- **Done when:** fresh clone → `make setup && make check` passes, and CI is green on the PR.

### Task 2 — Download + inspect (Sat Sept 26)
- `configs/datasets.yaml` with every candidate; look up the latest version on Universe for each.
- `download.py`: COCO format to `data/raw/<key>/`, idempotent, writes `MANIFEST.json`, commits a small hash file to `reports/data_manifests/`.
- `inspect_datasets.py`: per dataset/split: image count, resolution distribution, class names + instance counts, boxes per image, box-area buckets (small/medium/large as a fraction of image), zero-label images, and exact/near-duplicate images (perceptual hash) within and across datasets. A 4×4 sample grid per dataset with `supervision` annotators → `docs/assets/samples_<key>.png`. Summary → `reports/dataset_stats.csv`.
- **Done when:** stats and grids exist for every dataset that downloaded; failures are listed with reasons.

### Task 3 — Document the datasets (Sat Sept 26) — **stop for Nick's review**
- `docs/DATASETS.md`: per-dataset camera viewpoint, venue/lighting, label style (tight vs loose boxes, whether occluded fuel and fuel inside robots/hoppers are labeled), naming, and problems (duplicates, empty images, label errors, cross-dataset duplicates that could leak between train and test).
- Recommend 2–3 datasets and a baseline Dataset A with reasons. Draft attribution entries in the README.
- **Pause here** and summarize the recommendation for Nick before Task 4.

### Task 4 — Harmonize + splits (Sat Sept 26)
- `configs/class_map.yaml` covering every label found. `harmonize.py` rewrites to the two-class schema, logs before/after counts, raises on unmapped labels, writes `reports/class_coverage.json`.
- `splits.py`: deterministic per-source train/valid/test splits that respect the source's own splits where sensible; cross-dataset duplicates are removed or pinned to one split.
- Tests for mapping, splits, and determinism. Mapping log appended to `docs/DATASETS.md`.
- **Done when:** harmonized COCO loads with `sv.DetectionDataset.from_coco` and tests pass.

### Task 5 — Baseline training (start Sat night; Nick runs on platform)
- Nick uploads harmonized Dataset A and trains on Roboflow (RF-DETR nano/small or a small YOLO; check training credits first). Baseline uses no augmentation.
- Claude Code drafts the exact upload/training steps in `docs/RETRAINING.md` and records settings in `reports/models.yaml`.

### Task 6 — Evaluation harness (Sat Sept 26, testable before training finishes)
- `evaluate.py`: model (hosted Roboflow model ID or local weights via `inference`) × harmonized split → per-class mAP50, mAP50-95, precision, recall at a given confidence, PR curve, and confusion matrix with background. Uses `supervision` metrics (check the current API). Scores only classes the dataset labels.
- Caches raw predictions to `reports/runs/<run_id>/predictions.json` so thresholds can be re-swept without re-running inference. Writes `meta.json` per §3.5.
- `report.py` + `make report` regenerate the README table and `docs/EVALUATION.md` tables.
- Known-answer metric tests. Smoke-tested end to end on a small slice with any pretrained model.

### Task 7 — Cross-dataset results (Sun Sept 27) → `eval/baseline`
- Evaluate the baseline on A-test, B, C, and the field test set if available. `make report`. Write up the drop in `docs/EVALUATION.md`.

### Task 8 — Diagnosis (Sun–Mon Sept 27–28)
- Error slices by box size, brightness, blur (variance of Laplacian), crowding, and source. A failure gallery of 12–20 FN/FP examples with faces checked. Separate label-style mismatches (e.g., a "false positive" that's really an unlabeled fuel) from genuine model failures. Dataset problems are findings, not noise.

### Task 9 — Draft write-up + v0.1.0 release (Tue–Wed Sept 29–30)
- Blog-style draft following the narrative arc, `docs/MODEL_CARD.md` first draft, thin Colab notebook reproducing download → harmonize → eval. Update `CHANGELOG.md`, tag `v0.1.0`, publish a GitHub Release.

### Task 10 — Fix with merged data (v0.2.0)
- `merge.py` with preserved per-source held-out test splits and the leak check (tested). Retrain on merged data plus augmentation targeted at the diagnosed failure modes. Re-evaluate everywhere, including the field set.

### Task 11 — Threshold choice (v0.2.0)
- Sweep thresholds from cached predictions. Choose and justify one for the robot use case (a missed fuel vs a wasted pickup attempt have different costs).

### Task 12 — Edge deployment (v0.3.0, before interviews)
- Pi deployment via `inference`: FPS, p50/p95 latency, cloud vs edge accuracy, model size comparison, with hardware recorded in run metadata. `docs/DEPLOYMENT.md` runbook with known failure modes. Finalize `docs/RETRAINING.md`. Record the 2–3 minute demo video.

---

## 8. Field test set (our own REBUILT footage)

- Pending confirmation from Nick. If confirmed, raw files go in `data/field_test/raw/`.
- Extract frames at a low rate (e.g., 1 fps) and deduplicate with perceptual hashing.
- Label with the same two-class schema and the same label-style rules as in `docs/DATASETS.md`. Keep it small and careful (roughly 50–150 images).
- Record per-image source (event, camera position, lighting) in `data/field_test/manifest.csv` for slicing; commit its hash file like any other dataset.
- Eval-only, forever (rule 2). Faces: rule 3.

---

## 9. Writing style for docs

Moved to `.claude/rules/writing-style.md`.

---

## 10. Status

Moved to PROGRESS.md.
