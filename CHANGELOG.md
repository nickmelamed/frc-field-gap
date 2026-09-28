# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
