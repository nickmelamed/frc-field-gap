---
name: finish-task
description: Close out a task branch against the Definition of Done before asking to open the PR.
disable-model-invocation: true
---

Close out the current task (see PROGRESS.md).

1. Run `make check` and show the output. Fix failures before going on.
2. Run the spec-reviewer agent on `git diff main...HEAD`, pointing it at the
   SPEC sections for this task. Fix every problem it reports as affecting
   correctness, requirements, or the non-negotiable rules. List its optional
   items for Nick without acting on them.
3. Run the style-reviewer agent on the same diff and apply its rewrites.
4. Walk the Definition of Done and report each item with evidence.
   - `make check` passes locally.
   - New logic has tests, and coverage did not drop.
   - Public functions are typed and documented.
   - Configs were updated instead of hardcoding values.
   - README, the relevant docs, and CHANGELOG.md (under Unreleased) are
     updated.
   - Every new result has run metadata and was generated, not typed.
     `check_numbers.py` passes on README.md and docs/.
   - No data, weights, secrets, or faces were added. Check
     `git diff --stat main...HEAD` for anything under `data/` or any large
     binary.
   - Decisions that affect results are in docs/DECISIONS.md.
5. Tick the task in PROGRESS.md and note anything that turned out different
   from the spec.
6. List any multi-step procedure repeated during the task, or instructions
   Nick gave more than once. For each, draft a skill for
   `.claude/skills/<name>/SKILL.md` and show it to Nick. Write it only after
   he approves.
7. Commit fixes as atomic Conventional Commits.
8. Draft the PR description from `.github/pull_request_template.md` and show
   it to Nick with a short summary. Ask before pushing or opening the PR.
   PRs merge with rebase-and-merge once CI is green.
