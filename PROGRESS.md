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
  - Five datasets downloaded and inspected (D-008). `lava` has no
    published version, so it is skipped with a warning (D-024).
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
- [x] Task 3: Document datasets, review checkpoint (Sat Sept 26)
  - `docs/DATASETS.md` describes all five datasets. `marswars` is Dataset A,
    `robotzftp2` B, and deduplicated `scorekeeper` C, approved by Nick on
    2026-09-26 (D-011).
  - Differences from SPEC sections 4 and 5. Every robot label comes from
    earlier seasons (2019 to 2024 broadcasts and pits), so no dataset has
    REBUILT robots. The `marswars` state classes mark whether the hub is lit
    and are proposed as DROP. `testingfrfr` is an aggregate of the others and
    is left out of evaluation. The duplicate counts miss flipped and rotated
    copies, so they are lower bounds.
- [x] Task 4: Harmonize + splits (Sat Sept 26)
  - `make harmonize` maps labels to fuel and robot (D-012), re-splits A, B,
    and C, and writes `data/harmonized/<key>/<split>/`, which loads with
    `sv.DetectionDataset.from_coco`. `marswars` and `robotzftp2` are cut by
    frame order within each recording, and `scorekeeper` keeps one copy per
    photo and is split by related-image groups (D-013, D-014). Reports are
    `reports/splits.json`, `reports/class_coverage.json`, and
    `reports/harmonize_counts.csv`.
  - Differences from SPEC section 7. None of the Roboflow splits was worth
    keeping, so all three chosen datasets are re-split. `splits.py` is a
    library that `frc-harmonize` calls, so there is one command. A
    test/train near duplicate, or one shared by two re-split datasets,
    raises the new `SplitLeakError`, since `DataLeakError` means a
    field-test image. FIRST's official videos in `marswars` go whole to
    train. The planned `--only` flag was left out, because a partial run
    would overwrite the reports with partial ones. `testingfrfr` and
    `robotzftp2_fuel` are harmonized but keep their own splits, and must not
    be trained on until `merge.py` removes their matches to every test
    split. Flip- and rotation-aware hashing waits for `merge.py`. Within A,
    B, and C, rotated `scorekeeper` copies are removed by D-014 instead.
  - For Tasks 5 and 6. Nick uploads `data/harmonized/marswars/` with its
    splits as they are. Evaluation reads `labeled` in
    `reports/class_coverage.json` to skip robot on fuel-only datasets, and
    should report the recordings or groups behind each test split (`units`
    in `reports/splits.json`) and fuel boxes per recording, since one
    Basler run holds most of A-test's fuel boxes (D-013).
- [x] Task 5: Baseline training (start Sat night, Nick runs on platform)
  - RF-DETR Nano trained on version 2 of `frc-rebuilt-fuel-a`, recorded in
    `reports/models.yaml`. `frc-upload` sent each split by name, and
    `frc-verify-upload` matched every image of the version to its split and
    box count (`reports/platform_upload_marswars.json`, D-015, D-016).
  - Differences from SPEC section 7. Harmonized COCO files gained empty
    `info` and `licenses`, since the SDK uploader failed without them. The
    module is `upload.py`, since `platform` would shadow the standard
    library module. Exported images are matched by name without the
    `.rf.<hash>` suffix and `_jpg` tag, which Roboflow drops. The baseline
    trained on version 2, which adds the 384x384 stretch the training page
    asked for, so small fuel shrinks to a few pixels (D-016). The model ID
    leaves out the workspace, which is made from an email address.
  - For Tasks 6 and later. Evaluation calls the model by
    `model_id` in `reports/models.yaml`, with the workspace taken from the
    API key. The numbers on Roboflow's model page used a threshold picked
    on the test split and are reference only. The upload check compares box
    counts, not classes, so it needs a class check before the two-class
    merged data in Task 10.
- [x] Task 6: Eval harness + report generation (Sat Sept 26)
  - `frc-evaluate` (`make eval`) scores a model from `reports/models.yaml` on
    a harmonized split and writes `reports/runs/<run_id>/` with cached
    predictions, scores, and SPEC 3.5 metadata. `--from-cache` rescores a
    run without calling the model. `frc-report` (`make report`) writes the
    README table and the results in `docs/EVALUATION.md` (D-017, D-018).
  - Differences from SPEC section 7. Only hosted models are supported,
    through `inference-sdk`. Local weights wait for Task 12. The hosted
    model is named by project and version, so the workspace is never looked
    up. The PR curve is a table, since a plot needs matplotlib as a declared
    dependency. The smoke run used the trained baseline on 10 A-test images,
    not a generic pretrained model. `make setup-infer` installs the extra,
    and CI still runs without it. The `runs/` ignore rule was anchored to
    the repo root, since it also hid `reports/runs/`.
  - For Task 7 and later. A full A-test run at the platform's threshold
    matched the model page within a few boxes (D-018). Each run refuses a
    dirty tree and a new run leaves untracked files, so commit each run
    before the next or the next one needs `--allow-dirty`, which the report
    leaves out. A full split takes a few minutes, mostly the bootstrap.
- [x] Task 7: Cross-dataset results (Sun Sept 27)
  - baseline-a scored on A-test, B-test, and C-test at confidence 0.5,
    published by `make report`, and written up at the top of
    `docs/EVALUATION.md`. B scores as well as A. C drops on false positives
    alone, and recall stays at 1.0.
  - Differences from SPEC section 7. Runs score only classes the model was
    trained on, so C is scored on fuel only (D-019). C's prediction cache
    was far over the size limit at the 0.01 floor, so every image keeps its
    25 most confident predictions (D-020). A and B were rescored from their
    first caches under that limit, and the first runs stay committed. On C
    the limit dropped confident false positives, so C's mAP50 is an upper
    bound and its false positive count a lower bound. Each run now records
    how often it reached the limit.
    The field test set was not scored, since it is not confirmed.
  - For Task 8. Most of C's false positives appear to land on robot and
    broadcast photos from earlier games, which have no fuel labels, and one
    of those games used balls. This came from a quick look at the cached
    predictions, so Task 8 should measure it. Why B is no harder than A is
    also open.
- [x] Task 8: Diagnosis (Sun to Mon Sept 27 to 28)
  - `frc-diagnose` (`make diagnose MODEL=baseline-a`) slices the published
    runs' errors from their caches, writes `reports/diagnosis/`, and draws
    the failure gallery. `make report` renders the tables and figures into
    `docs/EVALUATION.md`, which has the write-up (D-021).
  - Findings. C's false positives fall on photos without fuel, and the
    review finds mostly people and balls from earlier games. B is easy
    because its balls are large and few. Most of A's misses, including its
    small-box misses, are balls cut off by the frame edge, and none is a
    clearly visible ball. C's clustered fuel has unlabeled balls.
  - Differences from SPEC section 7. Boxes are matched by IoU order, as
    supervision's confusion matrix does, so slices reproduce the runs'
    counts. matplotlib is a declared dependency. Run selection moved to
    `runs.py`. `frc-report` draws the figures. The gallery has 15 tiles
    (D-021).
  - For Nick. Check the verdicts in `reports/diagnosis/review.csv` and look
    at `docs/assets/failures.png` before the PR merges, especially tiles 8,
    11, and 15, which show distant people. The lockbox waits until after
    v0.1.0 (D-022).
  - For Task 10. Add negatives with people and other games' balls. Set a
    label rule for balls cut off by the frame edge, where A's labels and
    the model disagree on how much to box. Fix or drop C's unlabeled
    clustered fuel. The D-020 limit still counts every predicted class
    before unscored ones are dropped.
- [ ] Task 9: Draft write-up + v0.1.0 (Tue to Wed Sept 29 to 30)
  - `docs/WRITEUP.md` covers the baseline, the cross-dataset scores, and
    the diagnosis, and check_numbers checks it. `docs/MODEL_CARD.md` is a
    first draft. `notebooks/reproduce_baseline.ipynb` downloads,
    harmonizes, and rescores the committed predictions on Colab (D-023).
    The package version is 0.1.0. The tag and the GitHub Release wait for
    `/release v0.1.0` after the PR merges.
  - Differences from SPEC sections 6 and 7. The notebook is
    `reproduce_baseline.ipynb`, not `train_and_eval.ipynb`, since training
    runs on the platform. It rescores from cache instead of calling the
    model, which only the training workspace can call. It passed from a
    fresh clone in a new Python 3.11 environment, but its Colab-only cells
    have not been run on Colab.
  - For Nick. Run the notebook on Colab once after tagging, since it clones
    `v0.1.0`.
  - `frc-download` now skips an unpinned dataset such as `lava` with a
    warning instead of failing, so `make download harmonize` runs through
    (D-024).
- [ ] Task 10: Merged-data fix (v0.2.0)
- [ ] Task 11: Threshold choice (v0.2.0)
- [ ] Task 12: Edge deployment (v0.3.0, before interviews)

Field test set: pending confirmation

## Later, after v0.1.0 (raise with Nick)

- Add `hypothesis` for property-based tests of the metrics wrapper and the
  split logic, and run `mutmut` on `harmonize.py`, `splits.py`, and
  `merge.py`.
- Add a Claude review of every PR in CI.
- Declare matplotlib to draw the PR curve and confusion matrix as figures.
  It is only a transitive dependency of supervision today.
- Local weights through `inference.get_model` behind the `Predictor`
  interface in `evaluate.py` (Task 12).

## Open questions for Nick

None. The robot scoring question is D-019.
