---
name: release
description: Cut a tagged release (v0.1.0, v0.2.0, v0.3.0). Only Nick starts this.
disable-model-invocation: true
---

Prepare the release version Nick named when invoking this skill. If none was
given, ask for it.

1. Confirm the tree is clean, on `main`, with CI green.
2. Run `make check` and `make report`, and show that the generated README
   table and docs/EVALUATION.md match what is committed.
3. Run `scripts/agent/check_numbers.py` on README.md and docs/, and check
   every published image for faces (rule 3).
4. Confirm README attribution lists every Universe dataset with its exact
   version (rule 4).
5. Move CHANGELOG.md's Unreleased entries under the version and date.
6. Commit as `chore(release): <version>`.
7. Draft the GitHub Release summary. Show Nick the changelog entry, the
   summary, and the exact `git tag -a`, `git push`, and `gh release create`
   commands, then wait for approval.
