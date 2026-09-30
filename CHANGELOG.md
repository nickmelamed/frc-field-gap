# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-09-30

### Added

- `frc-merge` (`make merge`) builds the merged training set from A, B, C,
  and `testingfrfr`. It keeps each source's test split whole and drops any
  training image that matches a valid, test, or lockbox image by source
  name, by recording, or by perceptual hash under flips and 90 degree
  rotations. A field-test match stops the run.
- `pankratz` as a lockbox dataset, scored only, with an `eval_only` split
  method and attribution in the README.
- Scoring leaves fuel labels at the frame edge out, with the predictions
  that mostly overlap them, and each run records the counts.
- `frc-verify-upload --augmented` checks versions with augmented train
  copies.
- Sample grids can leave out source names matching a pattern.
- Runs of `merged-noaug` and `merged-aug` on A, B, C, and `pankratz`
  test, with their diagnoses, and baseline-a rescored under the edge rule.
- `frc-diagnose --no-gallery`, and a review file per model.
- `frc-threshold` (`make threshold`) chooses each model's fuel threshold
  from its cached test runs on A, B, and C, pricing a false positive at
  three misses. The lockbox is scored only at the chosen threshold, with
  balls held by a person reported as out of scope. `make report` adds the
  tables and a cost figure to `docs/EVALUATION.md`.
- `deploy_confidence` in `reports/models.yaml`. `merged-noaug` is the model
  chosen for the robot, at 0.49.

### Changed

- The prediction limit applies to each scored class after unscored classes
  are dropped. Published baseline scores are unchanged.
- The version check compares boxes per class, not only their total.
- `platform.project` is now `platform.projects`, one project per dataset
  key. Upload reports carry the version in their name.
- The prediction cache cap is 500000 bytes, still under the commit hook.
- Diagnosis figures use six series colors, and the size legend names every
  dataset.
- `diagnose.error_counts` and `evaluate.precision_recall` are shared, so
  the threshold sweep counts errors the way the runs do.
- The package version is 0.2.0.

## [0.1.0] - 2026-09-28

### Added

- Python 3.11 project managed with uv, with a committed lockfile and a
  `requirements.txt` export for Colab.
- ruff, mypy (strict), and pytest with an 80% coverage floor.
- pre-commit hooks, including gitleaks, nbstripout, and a style check.
- GitHub Actions CI that runs lint, typecheck, and tests from the lockfile.
- Package skeleton `frc_xdata` with config loading, errors, logging, and
  provenance helpers.
- README skeleton, pull request template, and `.env.example`.
- `frc-download` (`make download`) downloads pinned Universe datasets as
  COCO, verifies them against a SHA256 manifest, and records failures.
  `--resolve` looks candidates up so versions can be pinned.
- `frc-inspect` (`make inspect`) writes per-split stats, class counts, a
  duplicate report, and face-checked sample grids.
  `--contact-sheet KEY` writes annotated sheets (most boxes, no boxes, one
  per label) to the gitignored `data/contact_sheets/` for checking label
  style by eye.
- Five candidate datasets pinned in `configs/datasets.yaml`, with
  attribution in the README.
- `docs/DATASETS.md` describing each dataset's viewpoint, labels, and
  problems, the chosen datasets (D-011), and a proposed class mapping.
- `frc-harmonize` (`make harmonize`) maps every source label to fuel,
  robot, or DROP through `configs/class_map.yaml`, re-splits the three
  chosen datasets so no test image has a near duplicate in train or valid,
  and writes COCO files to `data/harmonized/` with split, coverage, and
  label-count reports (D-012 to D-014).
- `frc-upload` (`make upload`) checks a harmonized dataset against
  `reports/splits.json` and the field test set, then uploads it to Roboflow
  one split at a time with the split named. `frc-verify-upload`
  (`make verify-upload`) downloads the generated version and checks every
  image's split and box count (D-015).
- `frc-evaluate` (`make eval`) scores a hosted Roboflow model on a
  harmonized split and writes cached predictions, per-class scores, a
  confusion matrix, a precision-recall sweep, per-recording box counts,
  bootstrap intervals over recordings, and run metadata to
  `reports/runs/<run_id>/`. `--from-cache` rescores a run without calling
  the model. `make setup-infer` installs the optional `inference` extra it
  needs (D-017, D-018).
- `frc-report` (`make report`) rebuilds the README results table and the
  results in `docs/EVALUATION.md` from `reports/runs/`.
- `docs/RETRAINING.md` gives the platform training steps, and
  `reports/models.yaml` records each run.
- baseline-a scored on the test splits of `marswars`, `robotzftp2`, and
  `scorekeeper`, with the results in the README and `docs/EVALUATION.md`
  and a write-up of how the model does on each dataset.
- `frc-diagnose` (`make diagnose MODEL=...`) slices a model's errors by
  source, relative box size, crowding, brightness, and sharpness, from its
  published runs' cached predictions, and draws a face-checked failure
  gallery. `--review-sheet` writes numbered crop sheets and a blank verdict
  file under `data/contact_sheets/`. The verdicts go in
  `reports/diagnosis/review.csv` (D-021).
- baseline-a's diagnosis, with a write-up in `docs/EVALUATION.md` of why
  it fails where it does.
- `docs/WRITEUP.md`, a draft write-up of the baseline, the cross-dataset
  results, and the diagnosis.
- `docs/MODEL_CARD.md`, a first model card for baseline-a.
- `notebooks/reproduce_baseline.ipynb`, a Colab notebook that downloads
  and harmonizes the datasets and rescores the committed predictions
  against the published scores (D-023).

### Changed

- `frc-evaluate` scores only the classes that both the dataset and the
  model's training data label, and lists the rest in `unscored_classes`
  (D-019).
- `frc-evaluate` caches and scores at most the 25 most confident predictions
  per image (`max_predictions_per_image`), including when it rescores an
  older cache (D-020).
- Each run records in `metrics.json` how many images reached that limit, and
  how many reached it with every kept box above the threshold. The report
  prints both under the run's table.
- `frc-report` also renders the diagnosis tables into `docs/EVALUATION.md`
  and draws the diagnosis figures in `docs/assets/`.
- matplotlib is a declared dependency.
- The package version is 0.1.0.
- `frc-download` skips a dataset with no pinned project or version with a
  warning, instead of recording a failure and exiting 1 (D-024).

### Fixed

- Three `scorekeeper` sample-grid tiles with partly visible faces are
  left out of `docs/assets/samples_scorekeeper.png`.

[Unreleased]: https://github.com/nickmelamed/frc-field-gap/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/nickmelamed/frc-field-gap/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nickmelamed/frc-field-gap/releases/tag/v0.1.0
