"""Build the README results table and the tables in docs/EVALUATION.md.

``frc-report`` reads every run under ``reports/runs/`` and every diagnosis
under ``reports/diagnosis/``, and rewrites only the text between each
document's start and end markers. Runs are chosen by
:func:`frc_xdata.runs.select_runs`. Values are printed exactly as the run or diagnosis
stored them, so every number in the docs can be found under ``reports/``.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from frc_xdata.config import ProjectConfig, load_yaml
from frc_xdata.diagnose import DIAGNOSIS_NAME, FALSE_POSITIVE_KINDS, Diagnosis, load_diagnosis
from frc_xdata.download import PROJECT_CONFIG
from frc_xdata.errors import ConfigError
from frc_xdata.evaluate import ClassMetrics, RunResult
from frc_xdata.figures import write_figures
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.runs import PublishedRun, load_runs, select_runs

README = Path("README.md")
EVALUATION = Path("docs/EVALUATION.md")
RESULTS_MARKERS = ("<!-- RESULTS:START -->", "<!-- RESULTS:END -->")
EVALUATION_MARKERS = ("<!-- EVALUATION:START -->", "<!-- EVALUATION:END -->")
DIAGNOSIS_MARKERS = ("<!-- DIAGNOSIS:START -->", "<!-- DIAGNOSIS:END -->")
ERROR_PLURALS = {"false positive": "false positives", "miss": "misses"}
FALSE_POSITIVE_KIND_NAMES = {
    "duplicate": "Duplicate",
    "localization": "Localization",
    "inside_unscored": "Inside an unscored label",
    "background": "Background",
}
MISSING = "n/a"
# The interval level is a setting, not a result, so its line is exempt from
# the check that every number in the docs appears under reports/.
NOT_A_RESULT = " <!-- numbers: ok -->"
# scorekeeper's test split has over a hundred groups, so only the largest are listed.
MAX_UNITS = 10

logger = logging.getLogger(__name__)


def _value(value: float | None) -> str:
    return MISSING if value is None else str(value)


def _interval(run: PublishedRun, name: str, field: str) -> str:
    for c in run.result.bootstrap.classes:
        if c.name == name:
            interval = getattr(c, field)
            return MISSING if interval is None else f"{interval.low} to {interval.high}"
    return MISSING


def results_table(runs: Sequence[PublishedRun]) -> str:
    """Return the README table, one row per run and scored class."""
    if not runs:
        return "TBD. Results appear here once a model has been scored on a whole split."
    lines = [
        "| Model | Dataset | Split | Images | Class | mAP50 | mAP50 interval | mAP50-95 | "
        "Threshold | Precision | Recall | Recall interval |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        r, m = run.result, run.result.metrics
        for c in m.classes:
            lines.append(
                f"| {r.model} | {r.dataset} | {r.split} | {m.images} | {c.name} | "
                f"{_value(c.map50)} | {_interval(run, c.name, 'map50')} | "
                f"{_value(c.map50_95)} | {m.confidence} | {_value(c.precision)} | "
                f"{_value(c.recall)} | {_interval(run, c.name, 'recall')} |"
            )
    levels = sorted({_percent(run.result.bootstrap.level) for run in runs})
    lines += [
        "",
        f"Intervals hold the middle {' or '.join(levels)} of scores from resampling whole "
        f"recordings or photo groups, as `docs/EVALUATION.md` explains.{NOT_A_RESULT}",
    ]
    return "\n".join(lines)


def _percent(share: float) -> str:
    return f"{share:.0%}"


def _class_table(classes: Sequence[ClassMetrics], run: PublishedRun) -> list[str]:
    lines = [
        "| Class | Labeled boxes | mAP50 | mAP50 interval | mAP50-95 | Precision | Recall | "
        "Recall interval | Hits | False positives | Misses |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in classes:
        lines.append(
            f"| {c.name} | {c.instances} | {_value(c.map50)} | {_interval(run, c.name, 'map50')} | "
            f"{_value(c.map50_95)} | {_value(c.precision)} | {_value(c.recall)} | "
            f"{_interval(run, c.name, 'recall')} | {c.true_positives} | {c.false_positives} | "
            f"{c.false_negatives} |"
        )
    return lines


def _unscored_note(r: RunResult) -> list[str]:
    if not r.unscored_classes:
        return []
    names = " and ".join(r.unscored_classes)
    return [
        "",
        f"The {names} boxes in {r.dataset} are not scored, since {r.model} was never "
        "trained to find them.",
    ]


def _limit_note(r: RunResult) -> list[str]:
    limit = r.per_image_limit
    if limit is None or not limit.images_at_limit:
        return []
    lines = [
        "",
        f"{limit.images_at_limit} of {r.metrics.images} images reached the limit of "
        f"{limit.per_image} predictions per image (D-020), so fainter boxes may have been "
        "dropped there.",
    ]
    if limit.images_above_threshold:
        lines[-1] += (
            f" In {limit.images_above_threshold} of them every kept box is at or above the "
            "threshold, so boxes that would have counted at it were dropped, and the "
            "false positive count at the threshold is a lower bound."
        )
    return lines


def run_section(run: PublishedRun) -> str:
    """Return the EVALUATION.md section for one run."""
    r, m = run.result, run.result.metrics
    boot = r.bootstrap
    lines = [
        f"### {r.model} on {r.dataset} {r.split}",
        "",
        f"Run `{r.run_id}`, scored on {m.images} images. Precision, "
        f"recall, and the confusion matrix count predictions with confidence of at least "
        f"{m.confidence}.",
        f"Intervals come from {boot.resamples} resamples of the split's {boot.units} "
        f"recordings or groups, and hold the middle {_percent(boot.level)} of the resampled "
        f"scores.{NOT_A_RESULT}",
        "",
        *_class_table(m.classes, run),
        *_unscored_note(r),
        *_limit_note(r),
        "",
        "mAP50-95 by labeled box size. A box is small when its area is under 32x32 pixels "
        "and large when it is over 96x96, measured on the original image.",
        "",
        "| Small | Medium | Large |",
        "|---|---|---|",
        f"| {_value(m.small_map50_95)} | {_value(m.medium_map50_95)} | "
        f"{_value(m.large_map50_95)} |",
        "",
        "Confusion matrix. Rows are labeled boxes and columns are predictions. The "
        "background row holds predictions that matched no labeled box, and the background "
        "column holds labeled boxes the model missed.",
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


def load_diagnoses(diagnosis_dir: Path) -> list[Diagnosis]:
    """Read every model's diagnosis, in model order."""
    if not diagnosis_dir.is_dir():
        return []
    return [load_diagnosis(p) for p in sorted(diagnosis_dir.glob(f"*/{DIAGNOSIS_NAME}"))]


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _header(cells: Sequence[str]) -> list[str]:
    return [_row(cells), "|" + "---|" * len(cells)]


def _split_label(d: Diagnosis, dataset: str, class_name: str) -> str:
    """Name a split's rows by dataset, and by class when more than one is scored."""
    several = any(len(s.classes) > 1 for s in d.splits)
    return f"{dataset} {class_name}" if several else dataset


def _slice_tables(d: Diagnosis) -> list[str]:
    order: list[str] = []
    for s in d.splits:
        for c in s.classes:
            order += [x.by for x in c.slicings if x.by not in order]
    lines = []
    for by in order:
        title = by[0].upper() + by[1:]
        lines += [
            "",
            f"{title}:",
            "",
            *_header(
                [
                    "Dataset",
                    title,
                    "Images",
                    "Labeled boxes",
                    "Hits",
                    "False positives",
                    "Misses",
                    "Precision",
                    "Recall",
                    "mAP50",
                    "False positives per image",
                ]
            ),
        ]
        for s in d.splits:
            for c in s.classes:
                for sl in (x for x in c.slicings if x.by == by):
                    for x in sl.slices:
                        lines.append(
                            _row(
                                [
                                    _split_label(d, s.dataset, c.name),
                                    x.name,
                                    x.images,
                                    x.labeled,
                                    x.hits,
                                    x.false_positives,
                                    x.misses,
                                    _value(x.precision),
                                    _value(x.recall),
                                    _value(x.map50),
                                    x.false_positives_per_image,
                                ]
                            )
                        )
    return lines


def _size_table(d: Diagnosis) -> list[str]:
    b = d.area_buckets
    lines = [
        "",
        f"By relative box size. A box is small when it covers less than {b.small_max} of "
        f"the image area, medium below {b.medium_max}, and large otherwise. Labeled boxes "
        "are sized by their label and false positives by their own box.",
        "",
        *_header(["Dataset", "Size", "Labeled boxes", "Hits", "Recall", "False positives"]),
    ]
    for s in d.splits:
        for c in s.classes:
            for r in c.sizes:
                lines.append(
                    _row(
                        [
                            _split_label(d, s.dataset, c.name),
                            r.bucket,
                            r.labeled,
                            r.hits,
                            _value(r.recall),
                            r.false_positives,
                        ]
                    )
                )
    return lines


def _kind_table(d: Diagnosis) -> list[str]:
    lines = [
        "",
        "False positives by why they matched no label. A duplicate overlaps a label that "
        "another prediction already took. A localization error overlaps a label by more than "
        f"{d.localization_floor} IoU, but not enough to count. A box inside an unscored label "
        "has its center in a box of a class the model is not scored on, such as a robot. "
        "The rest are background.",
        "",
        *_header(
            [
                "Dataset",
                "Which images",
                *(FALSE_POSITIVE_KIND_NAMES[k] for k in FALSE_POSITIVE_KINDS),
            ]
        ),
    ]
    for s in d.splits:
        for c in s.classes:
            label = _split_label(d, s.dataset, c.name)
            rows = [("all", c.overall)]
            has = next((x for x in c.slicings if x.by == f"has labeled {c.name}"), None)
            if has is not None:
                names = {"yes": f"with labeled {c.name}", "no": f"without labeled {c.name}"}
                rows += [(names.get(x.name, x.name), x) for x in has.slices]
            for name, x in rows:
                kinds = x.false_positive_kinds
                lines.append(_row([label, name, *(kinds[k] for k in FALSE_POSITIVE_KINDS)]))
    return lines


def _quantiles(values: Sequence[float]) -> str:
    return " / ".join(str(v) for v in values) if values else MISSING


def _domain_table(d: Diagnosis) -> list[str]:
    qs = ", ".join(str(q) for q in d.quantiles)
    lines = [
        "",
        f"The model's training split next to each test split. Each cell gives the quantiles "
        f"{qs}. Box side is the side of a square with the box's share of the image area, as "
        "a fraction of the image side.",
        "",
        *_header(
            [
                "Split",
                "Role",
                "Images",
                "Labeled boxes",
                "Brightness",
                "Sharpness",
                "Box side",
                "Boxes per image",
            ]
        ),
    ]
    for f in d.domain:
        lines.append(
            _row(
                [
                    f"{f.dataset} {f.split}",
                    f.role,
                    f.images,
                    f.boxes,
                    _quantiles(f.brightness),
                    _quantiles(f.sharpness),
                    _quantiles(f.box_side),
                    _quantiles(f.boxes_per_image),
                ]
            )
        )
    return lines


def _review_table(d: Diagnosis) -> list[str]:
    if not d.review:
        return ["", "No errors have been reviewed by eye yet."]
    verdicts: list[str] = []
    for r in d.review:
        verdicts += [v for v in r.verdicts if v not in verdicts]
    groups = [f"{r.dataset} {ERROR_PLURALS.get(r.kind, r.kind)}" for r in d.review]
    lines = [
        "",
        "Errors judged by eye. Where a split has more errors than were reviewed, the reviewed "
        "ones are a seeded random sample.",
        "",
        *_header(["Verdict", *groups]),
        _row(["Errors", *(r.total for r in d.review)]),
        _row(["Reviewed", *(r.sampled for r in d.review)]),
    ]
    lines += [_row([v, *(r.verdicts.get(v, 0) for r in d.review)]) for v in verdicts]
    return lines


def _gallery_table(d: Diagnosis) -> list[str]:
    if not d.gallery:
        return []
    return [
        "",
        "The failure gallery, tile by tile from the top left:",
        "",
        *_header(["Tile", "Dataset", "Error", "Source"]),
        *(_row([n, t.dataset, t.kind, f"`{t.source}`"]) for n, t in enumerate(d.gallery, start=1)),
    ]


def diagnosis_section(d: Diagnosis) -> str:
    """Return the EVALUATION.md section for one model's diagnosis."""
    lines = [
        f"### Where {d.model}'s errors fall",
        "",
        f"Hits, false positives, and misses count predictions with confidence of at least "
        f"{d.confidence}, matched to labels the way supervision's confusion matrix matches "
        "them (see Limits). Brightness "
        "is the mean gray level from 0 to 255 and sharpness the variance of the Laplacian, "
        f"both measured on the image stretched to {d.feature_px} pixels square, as the model "
        "sees it. Their bins hold equal numbers of images, pooled over every test split, so a "
        "bin can hold few images of one dataset.",
        *_slice_tables(d),
        *_size_table(d),
        *_kind_table(d),
        *_domain_table(d),
        *_review_table(d),
        *_gallery_table(d),
    ]
    return "\n".join(lines)


def diagnosis_sections(diagnoses: Sequence[Diagnosis]) -> str:
    """Return the generated diagnosis part of EVALUATION.md."""
    if not diagnoses:
        return "No model has been diagnosed yet."
    return "\n\n".join(diagnosis_section(d) for d in diagnoses)


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


def write_report(
    runs_dir: Path, readme: Path, evaluation: Path, diagnosis_dir: Path | None = None
) -> list[Path]:
    """Rewrite the generated parts of both documents and return the ones that changed.

    The diagnosis part is written only once a diagnosis exists under
    ``diagnosis_dir``.
    """
    runs = select_runs(load_runs(runs_dir))
    diagnoses = [] if diagnosis_dir is None else load_diagnoses(diagnosis_dir)
    parts = [
        (readme, RESULTS_MARKERS, results_table(runs)),
        (evaluation, EVALUATION_MARKERS, evaluation_sections(runs)),
    ]
    if diagnoses:
        parts.append((evaluation, DIAGNOSIS_MARKERS, diagnosis_sections(diagnoses)))
    changed: list[Path] = []
    for path, markers, body in parts:
        old = path.read_text(encoding="utf-8")
        new = replace_between(old, markers, body)
        if new != old:
            path.write_text(new, encoding="utf-8")
            if path not in changed:
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
    write_report(project.evaluate.runs_dir, README, EVALUATION, project.diagnose.output_dir)
    for d in load_diagnoses(project.diagnose.output_dir):
        for path in write_figures(d, project.paths.assets_dir):
            logger.info("wrote %s", path)
    return 0
