"""Score a model's predictions on a harmonized split.

Metrics come from supervision. Mean average precision uses every cached
prediction, while precision, recall, and the confusion matrix count only
predictions at or above a confidence threshold. A dataset is scored only on
the classes it labels (``labeled`` in ``reports/class_coverage.json``), so a
robot prediction on a fuel-only dataset is neither a hit nor a false
positive.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import supervision as sv
from pydantic import BaseModel, ConfigDict
from supervision.metrics import (
    MeanAveragePrecision,
    MeanAveragePrecisionResult,
    Precision,
    Recall,
)

from frc_xdata.errors import ConfigError

# Metrics are stored rounded, and the report prints them unchanged, so every
# number in the docs appears verbatim under reports/.
DECIMALS = 3


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ClassMetrics(_Record):
    """Scores for one class. Counts are boxes at the confidence threshold."""

    name: str
    instances: int
    map50: float
    map50_95: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int


class PRPoint(_Record):
    """Precision and recall for one class at one confidence threshold."""

    threshold: float
    precision: float
    recall: float


class ConfusionTable(_Record):
    """Box counts by true class (rows) and predicted class (columns).

    The last row and column are background. A box in the background column was
    missed, and a box in the background row matched no labeled box.
    """

    labels: list[str]
    matrix: list[list[int]]


class EvalMetrics(_Record):
    """Every score for one model on one split."""

    images: int
    confidence: float
    iou: float
    map50: float
    map50_95: float
    small_map50_95: float
    medium_map50_95: float
    large_map50_95: float
    classes: list[ClassMetrics]
    confusion: ConfusionTable
    pr_curve: dict[str, list[PRPoint]]


def scored_classes(coverage_path: Path, key: str) -> list[str]:
    """Return the classes that dataset ``key`` labels.

    Raises:
        ConfigError: If the coverage report has no entry for ``key``, which
            means the dataset was never harmonized.
    """
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if key not in coverage:
        raise ConfigError(f"{key} is not in {coverage_path}. Run make harmonize first")
    labeled: list[str] = coverage[key]["labeled"]
    return labeled


def _select(detections: sv.Detections, mask: npt.NDArray[np.bool_]) -> sv.Detections:
    selected = detections[mask]
    if not isinstance(selected, sv.Detections):
        raise TypeError(f"expected Detections from a boolean mask, got {type(selected)}")
    return selected


def restrict(detections: sv.Detections, class_ids: Sequence[int]) -> sv.Detections:
    """Return only the detections whose class is in ``class_ids``."""
    if detections.class_id is None:
        return detections
    return _select(detections, np.isin(detections.class_id, np.asarray(class_ids)))


def above(detections: sv.Detections, confidence: float) -> sv.Detections:
    """Return the detections at or above ``confidence``."""
    if detections.confidence is None:
        return detections
    return _select(detections, np.asarray(detections.confidence) >= confidence)


def _round(value: float) -> float:
    return round(float(value), DECIMALS)


def _per_class(
    values: npt.NDArray[np.floating[Any]],
    found: npt.NDArray[np.integer[Any]],
    class_ids: Sequence[int],
) -> list[float]:
    """Pick one value per class id, with 0 for a class supervision did not report."""
    lookup = {int(c): float(v) for c, v in zip(found, values, strict=True)}
    return [lookup.get(c, 0.0) for c in class_ids]


def _precision_recall(
    predictions: list[sv.Detections], targets: list[sv.Detections], class_ids: Sequence[int]
) -> tuple[list[float], list[float]]:
    precision = Precision().update(predictions, targets).compute()
    recall = Recall().update(predictions, targets).compute()
    # Column 0 is IoU 0.5.
    return (
        _per_class(precision.precision_per_class[:, 0], precision.matched_classes, class_ids),
        _per_class(recall.recall_per_class[:, 0], recall.matched_classes, class_ids),
    )


def pr_curve(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    classes: Sequence[str],
    class_ids: Sequence[int],
    thresholds: Sequence[float],
) -> dict[str, list[PRPoint]]:
    """Return precision and recall at IoU 0.5 for each class at each threshold.

    Args:
        predictions: Per-image predictions, already limited to scored classes.
        targets: Per-image labels in the same order, limited the same way.
        classes: Every class name in the dataset, indexed by class id.
        class_ids: The scored class ids.
        thresholds: Confidence thresholds to sweep.
    """
    curve: dict[str, list[PRPoint]] = {classes[c]: [] for c in class_ids}
    for t in thresholds:
        kept = [above(p, t) for p in predictions]
        precision, recall = _precision_recall(kept, targets, class_ids)
        for c, p, r in zip(class_ids, precision, recall, strict=True):
            curve[classes[c]].append(
                PRPoint(threshold=_round(t), precision=_round(p), recall=_round(r))
            )
    return curve


def compute_metrics(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    classes: Sequence[str],
    scored: Sequence[str],
    confidence: float,
    iou: float,
    thresholds: Sequence[float],
) -> EvalMetrics:
    """Score per-image predictions against labels, on the scored classes only.

    Args:
        predictions: Per-image predictions, with confidences.
        targets: Per-image labels, in the same image order.
        classes: Every class name in the dataset, indexed by class id.
        scored: The class names this dataset labels.
        confidence: Threshold for precision, recall, and the confusion matrix.
            Mean average precision always uses every prediction.
        iou: Overlap needed for a prediction to count as a hit in the confusion
            matrix. Precision and recall are read at IoU 0.5.
        thresholds: Confidence thresholds for the precision-recall curve.

    Raises:
        ConfigError: If a scored class is not one of ``classes``, or the two
            lists of images differ in length.
    """
    unknown = sorted(set(scored) - set(classes))
    if unknown:
        raise ConfigError(f"scored classes {unknown} are not in the dataset classes {classes}")
    if len(predictions) != len(targets):
        raise ConfigError(f"{len(predictions)} prediction sets for {len(targets)} images")
    class_ids = sorted(classes.index(name) for name in scored)
    preds = [restrict(p, class_ids) for p in predictions]
    labels = [restrict(t, class_ids) for t in targets]

    mean_ap = MeanAveragePrecision().update(preds, labels).compute()
    ap50 = _per_class(mean_ap.ap_per_class[:, 0], mean_ap.matched_classes, class_ids)
    ap50_95 = _per_class(mean_ap.ap_per_class.mean(axis=1), mean_ap.matched_classes, class_ids)
    kept = [above(p, confidence) for p in preds]
    precision, recall = _precision_recall(kept, labels, class_ids)

    # The confusion matrix matches boxes by IoU regardless of class and needs
    # IoU strictly above the threshold, so with several classes its counts can
    # differ slightly from the class-aware precision and recall.
    scored_names = [classes[c] for c in class_ids]
    remap = {c: i for i, c in enumerate(class_ids)}
    matrix = sv.ConfusionMatrix.from_detections(
        predictions=[_reindex(p, remap) for p in kept],
        targets=[_reindex(t, remap) for t in labels],
        classes=scored_names,
        conf_threshold=confidence,
        iou_threshold=iou,
    ).matrix.astype(int)
    background = len(class_ids)

    per_class = [
        ClassMetrics(
            name=scored_names[i],
            instances=int(matrix[i].sum()),
            map50=_round(ap50[i]),
            map50_95=_round(ap50_95[i]),
            precision=_round(precision[i]),
            recall=_round(recall[i]),
            true_positives=int(matrix[i, i]),
            false_positives=int(matrix[:, i].sum() - matrix[i, i]),
            false_negatives=int(matrix[i].sum() - matrix[i, i]),
        )
        for i in range(background)
    ]
    return EvalMetrics(
        images=len(targets),
        confidence=confidence,
        iou=iou,
        map50=_round(mean_ap.map50),
        map50_95=_round(mean_ap.map50_95),
        small_map50_95=_round(_size_map(mean_ap.small_objects)),
        medium_map50_95=_round(_size_map(mean_ap.medium_objects)),
        large_map50_95=_round(_size_map(mean_ap.large_objects)),
        classes=per_class,
        confusion=ConfusionTable(labels=[*scored_names, "background"], matrix=matrix.tolist()),
        pr_curve=pr_curve(preds, labels, classes, class_ids, thresholds),
    )


def _size_map(result: MeanAveragePrecisionResult | None) -> float:
    return 0.0 if result is None else result.map50_95


def _reindex(detections: sv.Detections, remap: dict[int, int]) -> sv.Detections:
    """Renumber classes 0..n-1, since the confusion matrix indexes rows by class id."""
    if detections.class_id is None or len(detections) == 0:
        return detections
    return sv.Detections(
        xyxy=detections.xyxy,
        confidence=detections.confidence,
        class_id=np.array([remap[int(c)] for c in detections.class_id], dtype=int),
    )
