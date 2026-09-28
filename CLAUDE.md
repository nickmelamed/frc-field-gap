# FRC REBUILT cross-dataset generalization

Game-piece and robot detection for the 2026 FRC game (REBUILT), built around
cross-dataset generalization. We train a detector on one team's public
Roboflow Universe dataset, measure how it degrades on other teams' footage,
diagnose why, fix it by harmonizing and merging datasets, and then deploy to
a Raspberry Pi with measured edge performance. The arc is baseline,
generalization failure, diagnosis, fix, edge performance.

The repo is public and will be read by Roboflow hiring teams and by FRC
students who want to reuse it, so the code, commits, and history are part of
the deliverable. The v0.1.0 deadline is Wed Sept 30, 2026, 11:59pm PT.

The full design is in docs/SPEC.md, which keeps the original section numbers.
Read the relevant section before changing anything it covers. Status and next
steps are in PROGRESS.md, decisions in docs/DECISIONS.md, and git conventions
in docs/CONTRIBUTING.md.

## Non-negotiable rules

1. Never commit data, weights, or secrets. The Roboflow API key comes only
   from `ROBOFLOW_API_KEY` (loaded from `.env` with `python-dotenv`). Never
   read `.env`, print the key, log it, or hardcode it.
2. The field test set is eval-only. Nothing under `data/field_test/` may
   appear in any training set, merged dataset, or Roboflow project used for
   training. `merge.py` fails loudly on a leaked image hash, and a test
   covers it.
3. No identifiable faces in anything published. Check every field-test image
   before it goes into the README, docs, gallery, or video. Blur or exclude,
   and when unsure, exclude.
4. Attribute every Universe dataset (CC BY 4.0) in the README with name,
   workspace, URL, license, and exact version.
5. Verify before assuming. Dataset slugs, versions, class names, and the
   `roboflow`, `supervision`, `inference`, and `rfdetr` APIs change. Check the
   installed version and its docs or source first. If the spec conflicts with
   reality, trust reality and record it in docs/DECISIONS.md.
6. Numbers in docs come from generated reports, never typed by hand.
   `make report` writes the README results table and docs/EVALUATION.md.
   `scripts/agent/check_numbers.py` enforces this.
7. Log every decision that affects results (class mappings, dropped images,
   split choices, thresholds) with a one-line reason.
8. Never rewrite or backdate published history, and never weaken a test,
   lint rule, or check to make it pass.
9. The repo stands on its own. Don't mention other client work.

## Commands

```bash
make setup          # uv sync, pre-commit install
make check          # lint, typecheck, test
make agent-check    # the fast checks the Stop hook runs
make download inspect harmonize eval report   # pipeline stages
```

Dependencies change only through `uv` and the lockfile.

## Where things live

- `src/frc_xdata/`: the package. Configs in `configs/*.yaml`.
- `reports/`: committed small outputs and run metadata. `data/`: gitignored.
- `tests/`: synthetic COCO fixtures only. Network tests are marked
  `integration` and skipped in CI.
- `.claude/rules/`: Python, test, and writing standards for matching files.
- `.claude/skills/`: `/finish-task`, `/release`, `/platform-training`,
  `/publish-run`.

## How to work here

- One branch and one PR per task in PROGRESS.md. Plan first when a task
  touches several modules, and wait for approval.
- Task 3 is a review checkpoint. Stop and summarize for Nick before Task 4.
- A task is done when the Stop hook's checks pass and you have shown the
  output. Show evidence, not claims.
- Small atomic Conventional Commits. Code and its tests in one commit. Never
  `--no-verify`, never force-push.
- You may branch and commit locally. Ask before pushing, opening or merging
  PRs, tagging, or changing dependencies, hooks, or CI.
- Nick runs platform training himself. Draft the steps, don't try to run them.
- If you repeat a multi-step procedure, or I give you the same instructions
  twice, propose a skill for it in .claude/skills/ and wait for approval.
- When compacting, keep the modified files, the current plan and task, and
  any failing checks.
