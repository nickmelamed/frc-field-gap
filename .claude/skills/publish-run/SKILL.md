---
name: publish-run
description: Score or rescore a model on a harmonized split, check the run, and commit it so the next run starts from a clean tree. Use for every run that goes into the README or docs/EVALUATION.md.
---

Publish one run for the model, dataset, and split Nick named. The split is
`test` unless he says otherwise.

1. Check that `git status --porcelain` is empty. A run refuses a dirty tree,
   and `--allow-dirty` runs are left out of the report.
2. Decide how to get predictions.
   - Rescore with `uv run frc-evaluate <model> <dataset> --from-cache <run_id>`
     when an earlier run of the same model, dataset, and split already has a
     `predictions.json`. It makes no hosted calls.
   - Otherwise run `make eval MODEL=<model> DATASET=<dataset> SPLIT=<split>`,
     which calls the hosted model and counts toward the account's usage.
   - Run either in the background. A split takes about 5 minutes, mostly the
     bootstrap. Don't edit files while it runs.
   - On `PredictionCacheTooLargeError`, stop and ask Nick. Don't raise the
     size limit or the per-image limit on your own.
3. Check the new run directory under `reports/runs/`.
   - In `meta.json`, `git.dirty` and `source_dirty` are false.
   - In `metrics.json`, `metrics.images` and `bootstrap.units` match the
     split's `images` and `units` in `reports/splits.json`.
   - `scored_classes` and `unscored_classes` are what D-019 predicts for this
     model and dataset.
   - Read `per_image_limit`. If `images_above_threshold` is above 0, the
     false positive count is a lower bound (D-020), and the write-up must
     say so.
4. Run `grep -ril "nickmelamed\|berkeley" <run_dir>`. It must exit 1. The
   hosted server's reply names a workspace made from Nick's email, and it
   must never reach `reports/`.
5. Commit only that run directory, as
   `chore(reports): add <model> run on <dataset> <split>`, or
   `rescore <model> on ...` for a rescore.
6. After the last run, run `make report` and `make agent-check`, then commit
   the README and docs/EVALUATION.md as `docs(report): ...`. Any number in
   prose must be copied from the generated tables.
