"""Find the runs under ``reports/runs/`` that the docs publish.

A run is published when it scored a whole split from a clean working tree.
For each model, dataset, and split, the newest such run is used.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from frc_xdata.evaluate import META_NAME, METRICS_NAME, RunResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishedRun:
    """A run's scores and the metadata the report shows next to them."""

    result: RunResult
    created: str
    dirty: bool


def load_runs(runs_dir: Path) -> list[PublishedRun]:
    """Read every run directory that has both scores and metadata.

    A directory missing either file is skipped with a warning, since the
    evaluator writes both or neither.
    """
    runs = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []:
        metrics, meta = run_dir / METRICS_NAME, run_dir / META_NAME
        if not (metrics.is_file() and meta.is_file()):
            logger.warning("skipping %s, which lacks %s or %s", run_dir, METRICS_NAME, META_NAME)
            continue
        info = json.loads(meta.read_text(encoding="utf-8"))
        runs.append(
            PublishedRun(
                result=RunResult.model_validate_json(metrics.read_text(encoding="utf-8")),
                created=info["created"],
                dirty=info["git"]["dirty"] or info.get("source_dirty", False),
            )
        )
    return runs


def select_runs(runs: Sequence[PublishedRun]) -> list[PublishedRun]:
    """Return the newest clean whole-split run for each model, dataset, and split."""
    newest: dict[tuple[str, str, str], PublishedRun] = {}
    for run in runs:
        r = run.result
        if run.dirty or r.limit is not None:
            continue
        key = (r.model, r.dataset, r.split)
        if key not in newest or run.created > newest[key].created:
            newest[key] = run
    return [newest[k] for k in sorted(newest)]
