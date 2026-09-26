---
paths:
  - "**/*.py"
  - "**/*.md"
  - "**/*.qmd"
  - "**/*.R"
---

# Writing style for comments, docstrings, and docs

The owner writes this code. Everything should read like notes from a careful
engineer, not like generated text. `scripts/agent/check_style.py` checks the
mechanical parts after every edit.

## Comments

- Explain why, never what. If the code is clear, write no comment.
- Never mention CLAUDE.md, AGENTS.md, the agent, or instructions to the agent.
  If a rule matters to a reader, give the reason in plain terms instead.
- Never describe edit history ("updated", "now handles", "fixed"). That goes
  in the commit message.
- No section-banner comments.
- Match the comment density of the file. Do not add comments to lines you did
  not change.

## Docstrings

- Google style (ruff's `D` rules with the Google convention), on public
  modules, classes, and functions.
- One imperative summary line ("Return...", "Raise...").
- Add `Args`, `Returns`, and `Raises` sections only when they tell the reader
  something the name and type hints don't, such as units, shapes, a side
  effect, or which exception means what. If you add an `Args` section, list
  every argument, because ruff checks that.
- Never pad a docstring to look complete.

Good:

    """Map every source label to the two-class schema.

    Raises:
        UnmappedLabelError: If a label is missing from the class map. Labels
            are never dropped silently, so a new source label stops the run.
    """

Good comment:

    # From the dataset's official page (retrieved 2026-09-26), see D-009.
    # Keep the published wording, since these describe images, not diagnoses.

Bad comment (cites agent instructions and narrates its own diligence):

    # Copied verbatim (checked against the raw page). Never edit the wording:
    # CLAUDE.md section 4 forbids paraphrasing these.

## Prose (README, docs, write-ups, commit messages, PR descriptions)

- Plain sentences. No em dashes, and no en dashes except in numeric ranges.
- Semicolons rarely. Colons only to introduce code or a list.
- No bold lead-ins followed by a colon.
- Prefer paragraphs. Bullets only for items that are truly parallel.
- Avoid groups of three adjectives and "not X, but Y" framing.
- Avoid robust, seamless, comprehensive, leverage, delve, crucial, utilize,
  meticulous. No emoji.
- Say what the thing does and how to use it. Do not sell it.
- Results first, then method. Every claim is backed by a generated number or
  figure.
- Write so a high-school robotics student can follow. Define mAP,
  precision, and recall once.
- Be honest about limits, such as small test sets, label noise, what isn't
  reproducible, and what wasn't tested.

Add `style: ok` to a line only when its text is quoted verbatim from an
outside source.
