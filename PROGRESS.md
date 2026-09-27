# Progress

Update after each task: what's done, what's blocked, and anything in
docs/SPEC.md that turned out to be wrong. Decisions that affect results go in
docs/DECISIONS.md. Task scope and "done when" criteria are in SPEC section 7.

## Tasks

- [x] Task 1: Foundation (Sat Sept 26)
  - [x] Extend `make agent-check` to the full version once ruff, mypy, and
        pytest are installed:

        agent-check: check  ## Fast checks the Claude Code Stop hook runs
        	python3 scripts/agent/check_style.py .
        	python3 scripts/agent/check_numbers.py README.md docs/DATASETS.md docs/EVALUATION.md docs/MODEL_CARD.md docs/DEPLOYMENT.md --sources reports

  - [x] Add the local check-style hook to `.pre-commit-config.yaml`:

        - repo: local
          hooks:
            - id: check-style
              name: comment and doc style
              entry: python3 scripts/agent/check_style.py
              language: system
              types_or: [python, markdown]

  - Differences from SPEC section 3.1 and 7. No console scripts yet, since
    they land with their modules (D-002). mypy runs as a local pre-commit hook
    through uv instead of mirrors-mypy. `uv.lock` is excluded from the 500 KB
    large-file hook (D-006). Python is capped below 3.14 by `inference`
    (D-005). `configs/project.yaml` waits for its first consumer in Task 2.
- [x] Task 2: Download + inspect (Sat Sept 26)
  - Five datasets downloaded and inspected (D-008). `lava` failed because
    it has no published version, and the reason is in
    `reports/download_failures.json`.
  - Differences from SPEC sections 5 and 7. `marswars` also has a
    `red_active` class. `robot-zftp2` holds several projects, so
    `rebuilt-dataset-hcmwl` is `robotzftp2` and `frc-2026-fuel-ndrbj` is an
    extra key, `robotzftp2_fuel`. The committed hash files hold one hash per
    image directory instead of one per image (D-009). Grids are laid out
    with PIL because supervision's `create_tiles` is deprecated. Boxes and
    labels still use supervision annotators. Test fixtures are built
    in `tests/conftest.py` at test time instead of committed images.
  - For Task 3. `testingfrfr` and `scorekeeper` versions hold augmented
    copies, and the same source photo appears in more than one split.
    `testingfrfr` shares many images with `scorekeeper`, `robotzftp2`, and
    `robotzftp2_fuel`. See `reports/dataset_stats.csv` (images against
    source_images) and `reports/duplicates.json`.
- [ ] Task 3: Document datasets, review checkpoint (Sat Sept 26)
- [ ] Task 4: Harmonize + splits (Sat Sept 26)
- [ ] Task 5: Baseline training (start Sat night, Nick runs on platform)
- [ ] Task 6: Eval harness + report generation (Sat Sept 26)
- [ ] Task 7: Cross-dataset results (Sun Sept 27)
- [ ] Task 8: Diagnosis (Sun to Mon Sept 27 to 28)
- [ ] Task 9: Draft write-up + v0.1.0 (Tue to Wed Sept 29 to 30)
- [ ] Task 10: Merged-data fix (v0.2.0)
- [ ] Task 11: Threshold choice (v0.2.0)
- [ ] Task 12: Edge deployment (v0.3.0, before interviews)

Field test set: pending confirmation

## Later, after v0.1.0 (raise with Nick)

- Add `hypothesis` for property-based tests of the metrics wrapper and the
  split logic, and run `mutmut` on `harmonize.py`, `splits.py`, and
  `merge.py`.
- Add a Claude review of every PR in CI.
- `requirements.txt` holds core deps only. The `infer` extra pulls CPU-only
  torch through `inference-models`, which would replace Colab's CUDA build.
  Decide how the notebook installs `inference` when it lands (Task 5 or 6).

## Open questions for Nick
