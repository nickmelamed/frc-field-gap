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
- `docs/RETRAINING.md`, the procedure for platform training, and
  `reports/models.yaml` for recording each run.
