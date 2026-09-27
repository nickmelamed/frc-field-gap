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
- Five candidate datasets pinned in `configs/datasets.yaml`, with
  attribution in the README.
- `docs/DATASETS.md` describing each dataset's viewpoint, labels, and
  problems, the chosen datasets (D-011), and a proposed class mapping.
