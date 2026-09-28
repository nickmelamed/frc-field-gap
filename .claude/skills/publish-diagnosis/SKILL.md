---
name: publish-diagnosis
description: Review a model's errors by eye and publish its diagnosis from a clean tree. Use after new runs of the model are published or review verdicts change.
---

Publish `reports/diagnosis/<model>/`, the failure gallery, and the
diagnosis tables for the model Nick named (D-021).

1. Run `uv run frc-diagnose <model> --allow-dirty --review-sheet`. It writes
   numbered crop sheets and `review_template.csv` under
   `data/contact_sheets/`, which can show faces and are never published.
   - Judge every crop, and zoom in where a crop is too small to tell.
   - Save the template as `reports/diagnosis/review.csv` with a verdict from
     `diagnose.review.verdicts` on every row.
   - Add a note to any row the write-up will lean on, such as what the
     object is. Claims in the docs must be backed by a verdict or a note.
   - If an earlier `review.csv` exists, keep its verdicts for rows still in
     the sample. The run fails if the file and the sample disagree.
2. Commit `review.csv`. Delete trial outputs (`reports/diagnosis/<model>/`
   and `docs/assets/failures.png`) and stash doc edits, until
   `git status --porcelain` is empty. A run refuses untracked files too.
3. Run `make diagnose MODEL=<model>`, then check:
   - `meta.json` has `git.dirty` false and hashes the current `review.csv`.
   - The gallery lines in the log match the last face-checked list. If any
     tile is new, run the `face-check` skill before committing the PNG.
   - A warning that the gallery is short is expected when exclusions and
     the frame gap leave too few errors. Say so in DECISIONS.
   - `grep -ril "nickmelamed\|berkeley" reports/diagnosis docs/assets`
     exits 1.
4. Commit `reports/diagnosis/<model>/` and the gallery PNG as
   `chore(reports): ...`.
5. Pop the stash, run `make report` and `make agent-check`, and commit the
   docs and figures. Every number in prose must appear in `diagnosis.json`,
   a run's `metrics.json`, or `review.csv`. Write counts such as "30 of the
   40", never a share that is not in `reports/`.
6. Ask Nick to check the verdicts and the gallery before the PR merges.
