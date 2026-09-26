# Git workflow and commits

- Branch per task. Use `feat/<short-name>`, `docs/<...>`, `fix/<...>`, `chore/<...>`, `eval/<...>`. Never commit directly to `main` after Task 1's initial commit.
- Pull request per task, even solo. The PR description uses `.github/pull_request_template.md`: summary, why, how it was tested, results/screenshots if relevant, checklist. Merge only with CI green. Use rebase-and-merge so atomic commits land linearly on `main`.
- Conventional Commits (`https://www.conventionalcommits.org`):
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`, `data` (dataset/config changes), `eval` (new results/reports).
  - Optional scope: `feat(harmonize): ...`, `eval(baseline): ...`.
  - Subject: imperative mood, lowercase, no trailing period, ≤ 72 characters. Example: `feat(harmonize): error on unmapped source labels`.
  - Body (when the change isn't trivial): what and *why*, wrapped at 72. Reference decisions in docs when relevant.
- Atomic commits. One logical change per commit. Tests go in the same commit as the code they test. Formatting-only changes go in their own `style:`/`chore:` commit. Each commit on `main` should pass `make check`.
- No junk in history. No "wip", "fix typo again", or commented-out code. Clean up locally before pushing a branch (interactive rebase on unpushed commits is fine).
- Releases. Tag milestones with SemVer and keep `CHANGELOG.md` in Keep a Changelog format.
  - `v0.1.0`: baseline + cross-dataset results + draft write-up (by Sept 30).
  - `v0.2.0`: merged-data fix + threshold analysis.
  - `v0.3.0`: Pi deployment + runbook.
  - Create GitHub Releases from tags with a short summary.
