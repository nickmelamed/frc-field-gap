# Progress

Update after each task: what's done, what's blocked, and anything in
docs/SPEC.md that turned out to be wrong. Decisions that affect results go in
docs/DECISIONS.md. Task scope and "done when" criteria are in SPEC section 7.

## Tasks

- [ ] Task 1: Foundation (Sat Sept 26)
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

- [ ] Task 2: Download + inspect (Sat Sept 26)
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
