"""Build the README results table and the tables in docs/EVALUATION.md.

``frc-report`` reads every run under ``reports/runs/`` and rewrites only the
text between each document's start and end markers. A run is published when
it scored a whole split from a clean working tree. For each model, dataset,
and split, the newest such run is used. Values are printed exactly as the run
stored them, so every number in the docs can be found under ``reports/``.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from frc_xdata.config import ProjectConfig, load_yaml
from frc_xdata.download import PROJECT_CONFIG
from frc_xdata.errors import ConfigError
from frc_xdata.evaluate import META_NAME, METRICS_NAME, ClassMetrics, RunResult
from frc_xdata.logging_utils import add_log_level_argument, setup_logging

README = Path("README.md")
EVALUATION = Path("docs/EVALUATION.md")
RESULTS_MARKERS = ("<!-- RESULTS:START -->", "<!-- RESULTS:END -->")
EVALUATION_MARKERS = ("<!-- EVALUATION:START -->", "<!-- EVALUATION:END -->")
MISSING = "n/a"
# scorekeeper's test split has over a hundred groups, so only the largest are listed.
MAX_UNITS = 10

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishedRun:
    """A run's scores and the metadata the report shows next to them."""

    result: RunResult
    created: str
    commit: str
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
                commit=info["git"]["commit"],
                dirty=info["git"]["dirty"],
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


def _value(value: float | None) -> str:
    return MISSING if value is None else str(value)


def _interval(run: PublishedRun, name: str, field: str) -> str:
    for c in run.result.bootstrap.classes:
        if c.name == name:
            interval = getattr(c, field)
            return f"{interval.low} to {interval.high}"
    return MISSING


def results_table(runs: Sequence[PublishedRun]) -> str:
    """Return the README table, one row per run and scored class."""
    if not runs:
        return "TBD. Results appear here once a model has been scored on a whole split."
    lines = [
        "| Model | Dataset | Split | Images | Class | mAP50 | mAP50-95 | Confidence | "
        "Precision | Recall | Recall interval |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        r, m = run.result, run.result.metrics
        for c in m.classes:
            lines.append(
                f"| {r.model} | {r.dataset} | {r.split} | {m.images} | {c.name} | {c.map50} | "
                f"{c.map50_95} | {m.confidence} | {_value(c.precision)} | {c.recall} | "
                f"{_interval(run, c.name, 'recall')} |"
            )
    return "\n".join(lines)


def _class_table(classes: Sequence[ClassMetrics], run: PublishedRun) -> list[str]:
    lines = [
        "| Class | Labeled boxes | mAP50 | mAP50 interval | mAP50-95 | Precision | Recall | "
        "Recall interval | Hits | False positives | Misses |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in classes:
        lines.append(
            f"| {c.name} | {c.instances} | {c.map50} | {_interval(run, c.name, 'map50')} | "
            f"{c.map50_95} | {_value(c.precision)} | {c.recall} | "
            f"{_interval(run, c.name, 'recall')} | {c.true_positives} | {c.false_positives} | "
            f"{c.false_negatives} |"
        )
    return lines


def run_section(run: PublishedRun) -> str:
    """Return the EVALUATION.md section for one run."""
    r, m = run.result, run.result.metrics
    boot = r.bootstrap
    lines = [
        f"### {r.model} on {r.dataset} {r.split}",
        "",
        f"Run `{r.run_id}` at commit `{run.commit}`, scored on {m.images} images. Precision, "
        f"recall, and the confusion matrix count predictions with confidence of at least "
        f"{m.confidence}. Intervals come from {boot.resamples} resamples of the split's "
        f"{boot.units} recordings or groups, and hold a {boot.level} share of the resampled "
        "scores.",
        "",
        *_class_table(m.classes, run),
        "",
        "mAP50-95 by labeled box size, in COCO's pixel areas on the original image:",
        "",
        "| Small | Medium | Large |",
        "|---|---|---|",
        f"| {_value(m.small_map50_95)} | {_value(m.medium_map50_95)} | "
        f"{_value(m.large_map50_95)} |",
        "",
        "Confusion matrix. Rows are labeled boxes, columns are predictions.",
        "",
        "| | " + " | ".join(m.confusion.labels) + " |",
        "|---" * (len(m.confusion.labels) + 1) + "|",
    ]
    for label, row in zip(m.confusion.labels, m.confusion.matrix, strict=True):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")

    for name, points in m.pr_curve.items():
        lines += [
            "",
            f"Precision and recall for {name} at each confidence threshold:",
            "",
            "| Confidence | Precision | Recall |",
            "|---|---|---|",
            *(f"| {p.threshold} | {_value(p.precision)} | {p.recall} |" for p in points),
        ]

    scored = r.scored_classes
    shown = r.units[:MAX_UNITS]
    lines += [
        "",
        "Recordings or groups with the most labeled boxes:",
        "",
        "| Recording or group | Images | " + " | ".join(f"{c} boxes" for c in scored) + " |",
        "|---|---|" + "---|" * len(scored),
        *(
            f"| `{u.unit}` | {u.images} | " + " | ".join(str(u.boxes[c]) for c in scored) + " |"
            for u in shown
        ),
    ]
    if len(r.units) > len(shown):
        rest = r.units[len(shown) :]
        counts = ", ".join(f"{sum(u.boxes[c] for u in rest)} {c} boxes" for c in scored)
        lines += [
            "",
            f"The other {len(rest)} hold {sum(u.images for u in rest)} images and {counts}.",
        ]
    return "\n".join(lines)


def evaluation_sections(runs: Sequence[PublishedRun]) -> str:
    """Return the generated part of EVALUATION.md."""
    if not runs:
        return "No model has been scored on a whole split yet."
    return "\n\n".join(run_section(run) for run in runs)


def replace_between(text: str, markers: tuple[str, str], body: str) -> str:
    """Return ``text`` with everything between the two markers replaced by ``body``.

    Raises:
        ConfigError: If a marker is missing, repeated, or the end comes first,
            since text outside the markers is written by hand and must not be
            overwritten.
    """
    start, end = markers
    if text.count(start) != 1 or text.count(end) != 1:
        raise ConfigError(f"expected {start} and {end} exactly once each")
    head, rest = text.split(start)
    if end not in rest:
        raise ConfigError(f"{end} comes before {start}")
    _, tail = rest.split(end)
    return f"{head}{start}\n{body}\n{end}{tail}"


def write_report(runs_dir: Path, readme: Path, evaluation: Path) -> list[Path]:
    """Rewrite the generated parts of both documents and return the ones that changed."""
    runs = select_runs(load_runs(runs_dir))
    changed = []
    for path, markers, body in (
        (readme, RESULTS_MARKERS, results_table(runs)),
        (evaluation, EVALUATION_MARKERS, evaluation_sections(runs)),
    ):
        old = path.read_text(encoding="utf-8")
        new = replace_between(old, markers, body)
        if new != old:
            path.write_text(new, encoding="utf-8")
            changed.append(path)
    logger.info("%d runs published, changed: %s", len(runs), [str(p) for p in changed] or "none")
    return changed


def main(argv: list[str] | None = None) -> int:
    """Run the report command line and return the exit code."""
    parser = argparse.ArgumentParser(description="Rebuild the results tables from reports/runs/.")
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    project = load_yaml(args.project_config, ProjectConfig)
    write_report(project.evaluate.runs_dir, README, EVALUATION)
    return 0
