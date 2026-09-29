"""Score a model's predictions on a harmonized split.

``frc-evaluate`` runs a model from ``reports/models.yaml`` on every image of
one harmonized split and writes ``reports/runs/<run_id>/`` with the raw
predictions, the scores, and the run's metadata. ``--from-cache`` scores an
earlier run's predictions again without calling the model.

Metrics come from supervision. Mean average precision uses every cached
prediction, while precision, recall, and the confusion matrix count only
predictions at or above a confidence threshold. A dataset is scored only on
the classes it labels (``labeled`` in ``reports/class_coverage.json``) that
the model's training dataset labels too. A robot prediction on a fuel-only
dataset is neither a hit nor a false positive, and a fuel-only model is not
scored on robot boxes.
"""

import argparse
import json
import logging
import platform
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import supervision as sv
import yaml
from pydantic import BaseModel, ConfigDict
from supervision.metrics import (
    MeanAveragePrecision,
    MeanAveragePrecisionResult,
    Precision,
    Recall,
)

from frc_xdata.config import (
    DatasetsConfig,
    ModelEntry,
    ModelsFile,
    ProjectConfig,
    SplitMethod,
    load_yaml,
)
from frc_xdata.download import DATASETS_CONFIG, PROJECT_CONFIG, redact, redacted_api_key
from frc_xdata.errors import (
    ConfigError,
    DirtyTreeError,
    PredictionCacheTooLargeError,
    UnmappedLabelError,
)
from frc_xdata.harmonize import SPLITS_NAME
from frc_xdata.inspect_datasets import ANNOTATIONS_NAME, ImageRecord, load_split, phash
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.provenance import (
    git_commit,
    git_is_dirty,
    git_tree,
    package_versions,
    sha256_file,
    utc_timestamp,
)
from frc_xdata.splits import SPLITS, leak_groups, recording_key

# Metrics are stored rounded, and the report prints them unchanged, so every
# number in the docs appears verbatim under reports/.
DECIMALS = 3
# Cached predictions are rounded so a whole split is small enough to commit.
# Runs are scored from the rounded values, so rescoring a cache matches.
CONFIDENCE_DECIMALS = 4
COORD_DECIMALS = 1
PREDICTIONS_NAME = "predictions.json"
METRICS_NAME = "metrics.json"
META_NAME = "meta.json"
# ``predictions_from`` for a run that called the model itself.
THIS_RUN = "this run"
LOG_EVERY = 25

logger = logging.getLogger(__name__)


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ClassMetrics(_Record):
    """Scores for one class. Counts are boxes at the confidence threshold."""

    name: str
    instances: int
    # None when the split has no labeled box of this class, and precision is
    # None when the model made no prediction of it. Neither is then defined.
    map50: float | None
    map50_95: float | None
    precision: float | None
    recall: float | None
    true_positives: int
    false_positives: int
    false_negatives: int


class PRPoint(_Record):
    """Precision and recall for one class at one confidence threshold."""

    threshold: float
    precision: float | None
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
    # None when the split has no labeled box of that size.
    small_map50_95: float | None
    medium_map50_95: float | None
    large_map50_95: float | None
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


def shared_classes(labeled: Sequence[str], trained: Sequence[str]) -> list[str]:
    """Return the classes in ``labeled`` that the model was trained on, in dataset order.

    Raises:
        ConfigError: If the two share no class, since the run would score nothing.
    """
    shared = [name for name in labeled if name in trained]
    if not shared:
        raise ConfigError(f"the dataset labels {list(labeled)} but the model knows {list(trained)}")
    return shared


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


def at_edge(
    xyxy: npt.NDArray[np.floating[Any]], width: float, height: float, tolerance: float
) -> npt.NDArray[np.bool_]:
    """Return which boxes reach within ``tolerance`` pixels of the image edge."""
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return (
        (boxes[:, 0] <= tolerance)
        | (boxes[:, 1] <= tolerance)
        | (boxes[:, 2] >= width - tolerance)
        | (boxes[:, 3] >= height - tolerance)
    )


def overlap_of_smaller(
    a: npt.NDArray[np.floating[Any]], b: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.float64]:
    """Return the intersection of each pair of boxes over the smaller box's area.

    It is 1 when one box lies inside the other, however different their
    sizes, which IoU would score low.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    w = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    h = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    smaller = np.minimum(area_a[:, None], area_b[None, :])
    shares: npt.NDArray[np.float64] = np.divide(
        w * h, smaller, out=np.zeros_like(smaller), where=smaller > 0
    )
    return shares


class IgnoredAtEdge(_Record):
    """Labels at the frame edge a run left out of scoring, and the predictions dropped with them.

    The settings are kept with the counts so a later rescore or diagnosis
    applies exactly the same rule.
    """

    classes: list[str]
    tolerance_px: float
    min_overlap: float
    labels: int
    predictions: int


def ignore_edge_labels(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    sizes: Sequence[tuple[int, int]],
    class_ids: Sequence[int],
    tolerance: float,
    min_overlap: float,
    iou: float,
) -> tuple[list[sv.Detections], list[sv.Detections], int, int]:
    """Leave labels at the frame edge out of scoring, with the predictions that match them.

    A label of one of ``class_ids`` within ``tolerance`` pixels of the edge
    is removed. A prediction of the same class is removed when
    :func:`overlap_of_smaller` with such a label is above ``min_overlap``,
    unless its IoU with a remaining label of its class is at least ``iou``,
    so a hit on a scored ball is never dropped.

    Args:
        predictions: Per-image predictions.
        targets: Per-image labels, in the same order.
        sizes: Each image's width and height in pixels.
        class_ids: Classes whose edge labels are ignored.
        tolerance: Pixels from the edge that still count as touching it.
        min_overlap: Overlap with an ignored label above which a
            prediction is dropped.
        iou: IoU with a remaining label at which a prediction is kept.

    Returns:
        The predictions and labels to score, and how many labels and
        predictions were removed.
    """
    kept_preds: list[sv.Detections] = []
    kept_targets: list[sv.Detections] = []
    labels_removed = 0
    preds_removed = 0
    for preds, labels, (width, height) in zip(predictions, targets, sizes, strict=True):
        label_xyxy = np.asarray(labels.xyxy, dtype=np.float64).reshape(-1, 4)
        pred_xyxy = np.asarray(preds.xyxy, dtype=np.float64).reshape(-1, 4)
        label_cls = np.asarray(labels.class_id if labels.class_id is not None else [], dtype=int)
        edge = np.isin(label_cls, class_ids) & at_edge(label_xyxy, width, height, tolerance)
        pred_cls = np.asarray(preds.class_id if preds.class_id is not None else [], dtype=int)
        drop = np.zeros(len(preds), dtype=bool)
        for cid in class_ids:
            ignored = label_xyxy[edge & (label_cls == cid)]
            mine = pred_cls == cid
            if not len(ignored) or not mine.any():
                continue
            near = (overlap_of_smaller(pred_xyxy, ignored) > min_overlap).any(axis=1)
            scored = label_xyxy[~edge & (label_cls == cid)]
            hit = (
                (np.asarray(sv.box_iou_batch(pred_xyxy, scored)) >= iou).any(axis=1)
                if len(scored) and len(preds)
                else np.zeros(len(preds), dtype=bool)
            )
            drop |= mine & near & ~hit
        labels_removed += int(edge.sum())
        preds_removed += int(drop.sum())
        kept_targets.append(_select(labels, ~edge) if edge.any() else labels)
        kept_preds.append(_select(preds, ~drop) if drop.any() else preds)
    return kept_preds, kept_targets, labels_removed, preds_removed


def apply_edge_ignore(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    records: Sequence[ImageRecord],
    classes: Sequence[str],
    ignore: Sequence[str],
    tolerance: float,
    min_overlap: float,
    iou: float,
) -> tuple[list[sv.Detections], list[sv.Detections], list[ImageRecord], IgnoredAtEdge]:
    """Run :func:`ignore_edge_labels` on a split and drop the same labels from its records.

    ``records`` are the split's images in the order of ``targets``, and
    ``classes`` the dataset's class names by id. The returned records keep
    only the scored boxes, so counts per recording agree with the scores.
    """
    class_ids = [classes.index(c) for c in ignore]
    preds, kept, n_labels, n_preds = ignore_edge_labels(
        predictions,
        targets,
        [(r.width, r.height) for r in records],
        class_ids,
        tolerance,
        min_overlap,
        iou,
    )
    kept_records = []
    for r in records:
        xyxy = np.array([(b.x, b.y, b.x + b.w, b.y + b.h) for b in r.boxes]).reshape(-1, 4)
        edge = at_edge(xyxy, r.width, r.height, tolerance)
        boxes = zip(r.boxes, edge, strict=True)
        kept_records.append(
            replace(r, boxes=tuple(b for b, e in boxes if not (e and b.label in ignore)))
        )
    summary = IgnoredAtEdge(
        classes=list(ignore),
        tolerance_px=tolerance,
        min_overlap=min_overlap,
        labels=n_labels,
        predictions=n_preds,
    )
    return preds, kept, kept_records, summary


def _round(value: float) -> float:
    return round(float(value), DECIMALS)


def _maybe_round(value: float | None) -> float | None:
    return None if value is None else _round(value)


def _per_class(
    values: npt.NDArray[np.floating[Any]],
    found: npt.NDArray[np.integer[Any]],
    class_ids: Sequence[int],
) -> list[float]:
    """Pick one value per class id, with 0 for a class supervision did not report."""
    lookup = {int(c): float(v) for c, v in zip(found, values, strict=True)}
    return [lookup.get(c, 0.0) for c in class_ids]


def _count_class(detections: list[sv.Detections], class_id: int) -> int:
    return sum(int((d.class_id == class_id).sum()) for d in detections if d.class_id is not None)


def _precision_recall(
    predictions: list[sv.Detections], targets: list[sv.Detections], class_ids: Sequence[int]
) -> tuple[list[float | None], list[float]]:
    precision = Precision().update(predictions, targets).compute()
    recall = Recall().update(predictions, targets).compute()
    # Column 0 is IoU 0.5. supervision reports 0 precision for a class with
    # no predictions, where it is really undefined.
    per_class = _per_class(
        precision.precision_per_class[:, 0], precision.matched_classes, class_ids
    )
    return (
        [
            p if _count_class(predictions, c) else None
            for c, p in zip(class_ids, per_class, strict=True)
        ],
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
                PRPoint(threshold=_round(t), precision=_maybe_round(p), recall=_round(r))
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

    labeled = [_count_class(labels, c) for c in class_ids]
    per_class = [
        ClassMetrics(
            name=scored_names[i],
            instances=int(matrix[i].sum()),
            map50=_round(ap50[i]) if labeled[i] else None,
            map50_95=_round(ap50_95[i]) if labeled[i] else None,
            precision=_maybe_round(precision[i]),
            recall=_round(recall[i]) if labeled[i] else None,
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
        small_map50_95=_maybe_round(_size_map(mean_ap.small_objects)),
        medium_map50_95=_maybe_round(_size_map(mean_ap.medium_objects)),
        large_map50_95=_maybe_round(_size_map(mean_ap.large_objects)),
        classes=per_class,
        confusion=ConfusionTable(labels=[*scored_names, "background"], matrix=matrix.tolist()),
        pr_curve=pr_curve(preds, labels, classes, class_ids, thresholds),
    )


def _size_map(result: MeanAveragePrecisionResult | None) -> float | None:
    # COCO scores a size with no labeled boxes as -1.
    if result is None or result.map50_95 < 0:
        return None
    return result.map50_95


def _reindex(detections: sv.Detections, remap: dict[int, int]) -> sv.Detections:
    """Renumber classes 0..n-1, since the confusion matrix indexes rows by class id."""
    if detections.class_id is None or len(detections) == 0:
        return detections
    return sv.Detections(
        xyxy=detections.xyxy,
        confidence=detections.confidence,
        class_id=np.array([remap[int(c)] for c in detections.class_id], dtype=int),
    )


class Prediction(_Record):
    """One predicted box, in the original image's pixels."""

    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def _row(p: Prediction) -> list[str | float]:
    return [
        p.class_name,
        round(p.confidence, CONFIDENCE_DECIMALS),
        *(round(v, COORD_DECIMALS) for v in p.xyxy),
    ]


def top_k(predictions: Sequence[Prediction], k: int) -> list[Prediction]:
    """Return the ``k`` most confident predictions, keeping the given order among equals."""
    return sorted(predictions, key=lambda p: -p.confidence)[:k]


def limit_predictions(
    predictions: Sequence[Prediction], k: int, classes: Sequence[str] | None = None
) -> list[Prediction]:
    """Return the ``k`` most confident predictions of each class, most confident first.

    Classes outside ``classes`` are dropped before the limit, so boxes of a
    class the run does not score never push out boxes of one it does. With
    ``classes`` left out, every class is kept. Equal confidences keep the
    given order.
    """
    kept: list[Prediction] = []
    per_class: Counter[str] = Counter()
    for p in top_k(predictions, len(predictions)):
        if (classes is None or p.class_name in classes) and per_class[p.class_name] < k:
            per_class[p.class_name] += 1
            kept.append(p)
    return kept


def predictions_text(
    predictions: Mapping[str, Sequence[Prediction]],
    floor: float,
    per_image: int,
    max_bytes: int,
    classes: Sequence[str] | None = None,
) -> str:
    """Return the cache file text for each image's ``per_image`` most confident predictions.

    The limit applies to each class in ``classes`` (every class when left
    out), as in :func:`limit_predictions`. Predictions below ``floor`` are
    left out. Each image is one line keyed by
    file name, so a diff shows which images changed. An image with no
    predictions is kept with an empty list, which tells it apart from an image
    that was never run.

    Raises:
        PredictionCacheTooLargeError: If the text would exceed ``max_bytes``.
    """
    lines = []
    for name, preds in predictions.items():
        kept = limit_predictions([p for p in preds if p.confidence >= floor], per_image, classes)
        lines.append(f"{json.dumps(name)}: {json.dumps([_row(p) for p in kept])}")
    text = (
        "{\n"
        f'"confidence_floor": {json.dumps(floor)},\n'
        f'"max_predictions_per_image": {json.dumps(per_image)},\n'
        '"images": {\n' + ",\n".join(lines) + "\n}\n}\n"
    )
    size = len(text.encode("utf-8"))
    if size > max_bytes:
        raise PredictionCacheTooLargeError(
            f"cached predictions would be {size} bytes, over the {max_bytes} byte limit. "
            "Raise the confidence floor, lower max_predictions_per_image, "
            "or evaluate a smaller split"
        )
    return text


def save_predictions(
    path: Path,
    predictions: Mapping[str, Sequence[Prediction]],
    floor: float,
    per_image: int,
    max_bytes: int,
    classes: Sequence[str] | None = None,
) -> int:
    """Write the cache for :func:`predictions_text` to ``path`` and return its size in bytes.

    Raises:
        PredictionCacheTooLargeError: If the file would exceed ``max_bytes``.
            Nothing is written in that case.
    """
    text = predictions_text(predictions, floor, per_image, max_bytes, classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def load_predictions(path: Path) -> dict[str, list[Prediction]]:
    """Read predictions written by :func:`save_predictions`, keyed by image file name."""
    return parse_predictions(path.read_text(encoding="utf-8"))


def parse_predictions(text: str) -> dict[str, list[Prediction]]:
    """Parse cache text from :func:`predictions_text`, keyed by image file name."""
    data = json.loads(text)
    return {
        name: [
            Prediction(class_name=row[0], confidence=row[1], xyxy=(row[2], row[3], row[4], row[5]))
            for row in rows
        ]
        for name, rows in data["images"].items()
    }


def to_detections(predictions: Sequence[Prediction], classes: Sequence[str]) -> sv.Detections:
    """Convert predictions to supervision detections with the dataset's class ids.

    Classes are matched by name, since a model numbers its classes its own way.

    Raises:
        UnmappedLabelError: If the model predicts a class the dataset does not
            have.
    """
    unknown = sorted({p.class_name for p in predictions} - set(classes))
    if unknown:
        raise UnmappedLabelError(f"model predicts {unknown}, which are not in {list(classes)}")
    if not predictions:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([p.xyxy for p in predictions], dtype=float),
        confidence=np.array([p.confidence for p in predictions], dtype=float),
        class_id=np.array([classes.index(p.class_name) for p in predictions], dtype=int),
    )


def align(
    predictions: Mapping[str, Sequence[Prediction]],
    image_names: Sequence[str],
    classes: Sequence[str],
) -> list[sv.Detections]:
    """Return one set of detections per image, in the order of ``image_names``.

    Raises:
        ConfigError: If an image has no entry in ``predictions``, so a cache
            from another split or a partial run is never scored.
    """
    missing = [name for name in image_names if name not in predictions]
    if missing:
        raise ConfigError(f"{len(missing)} images have no cached predictions, such as {missing[0]}")
    return [to_detections(predictions[name], classes) for name in image_names]


class UnitCount(_Record):
    """How many images and labeled boxes one recording or group adds to a split."""

    unit: str
    images: int
    boxes: dict[str, int]


class Interval(_Record):
    """A bootstrap percentile interval."""

    low: float
    high: float


class ClassInterval(_Record):
    """Bootstrap intervals for one class."""

    name: str
    # None when no resample holds a labeled box of this class.
    map50: Interval | None
    recall: Interval | None


class BootstrapResult(_Record):
    """Intervals from resampling whole recordings or groups with replacement."""

    units: int
    resamples: int
    level: float
    seed: int
    classes: list[ClassInterval]


def image_units(
    records: Sequence[ImageRecord],
    method: SplitMethod | None,
    hashes: Sequence[int] | None,
    max_distance: int,
) -> list[str]:
    """Return the recording or group of each image, the same way the split made them.

    A temporal dataset's unit is the recording in the source name. A grouped
    dataset's unit is its related-image group, rebuilt from the images alone,
    and so is an eval-only dataset's.
    Whole groups went to one split, so rebuilding them from one split gives
    the same groups. A dataset that was not re-split has one unit per image.

    Args:
        records: The images of one split.
        method: How the dataset was re-split, or None if it keeps its own splits.
        hashes: Perceptual hash of each image, in the order of ``records``.
            Needed only for the grouped method.
        max_distance: Largest hash distance that joins two images.

    Raises:
        ConfigError: If a temporal source name names no recording, or hashes
            are missing for a grouped dataset.
    """
    if method is None:
        return [r.ref.file_name for r in records]
    if method.method == "temporal":
        units = []
        for r in records:
            recording = recording_key(r.source_name, method.recording_pattern)
            if recording is None:
                raise ConfigError(
                    f"{r.ref}: {r.source_name!r} does not match the recording pattern"
                )
            units.append(recording)
        return units
    if hashes is None:
        raise ConfigError(f"{method.method} datasets need image hashes to rebuild their groups")
    labels = [""] * len(records)
    for group in leak_groups(records, hashes, method.recording_pattern, max_distance):
        label = min(records[i].ref.file_name for i in group)
        for i in group:
            labels[i] = label
    return labels


def unit_counts(
    records: Sequence[ImageRecord], units: Sequence[str], scored: Sequence[str]
) -> list[UnitCount]:
    """Count images and scored boxes per unit, most boxes first."""
    images: dict[str, int] = {}
    boxes: dict[str, dict[str, int]] = {}
    for r, unit in zip(records, units, strict=True):
        images[unit] = images.get(unit, 0) + 1
        counts = boxes.setdefault(unit, dict.fromkeys(scored, 0))
        for b in r.boxes:
            if b.label in counts:
                counts[b.label] += 1
    return sorted(
        (UnitCount(unit=u, images=images[u], boxes=boxes[u]) for u in images),
        key=lambda c: (-sum(c.boxes.values()), c.unit),
    )


def bootstrap(
    predictions: list[sv.Detections],
    targets: list[sv.Detections],
    units: Sequence[str],
    classes: Sequence[str],
    scored: Sequence[str],
    confidence: float,
    resamples: int,
    level: float,
    seed: int,
) -> BootstrapResult:
    """Return percentile intervals for mAP50 and recall, resampling whole units.

    Frames from one recording are alike, so resampling single images would
    treat them as independent and give intervals that are too narrow.

    Args:
        predictions: Per-image predictions, with confidences.
        targets: Per-image labels, in the same order.
        units: The recording or group of each image, in the same order.
        classes: Every class name in the dataset, indexed by class id.
        scored: The class names this dataset labels.
        confidence: Threshold for recall.
        resamples: How many resamples to draw.
        level: Share of resampled values inside the interval, such as 0.95.
        seed: Seed for the random generator, so intervals are repeatable.
    """
    class_ids = sorted(classes.index(name) for name in scored)
    preds = [restrict(p, class_ids) for p in predictions]
    labels = [restrict(t, class_ids) for t in targets]
    members: dict[str, list[int]] = {}
    for i, unit in enumerate(units):
        members.setdefault(unit, []).append(i)
    groups = [members[u] for u in sorted(members)]

    rng = np.random.default_rng(seed)
    # A resample with no labeled box of a class says nothing about it, so its
    # score is left out (NaN) rather than counted as zero.
    ap50 = np.full((resamples, len(class_ids)), np.nan)
    recall = np.full((resamples, len(class_ids)), np.nan)
    for n in range(resamples):
        picked = [i for g in rng.integers(0, len(groups), len(groups)) for i in groups[g]]
        sample_preds = [preds[i] for i in picked]
        sample_labels = [labels[i] for i in picked]
        present = np.array([_count_class(sample_labels, c) > 0 for c in class_ids])
        mean_ap = MeanAveragePrecision().update(sample_preds, sample_labels).compute()
        ap = _per_class(mean_ap.ap_per_class[:, 0], mean_ap.matched_classes, class_ids)
        _, rec = _precision_recall(
            [above(p, confidence) for p in sample_preds], sample_labels, class_ids
        )
        ap50[n, present] = np.array(ap)[present]
        recall[n, present] = np.array(rec, dtype=float)[present]

    return BootstrapResult(
        units=len(groups),
        resamples=resamples,
        level=level,
        seed=seed,
        classes=[
            ClassInterval(
                name=classes[c],
                map50=_percentile_interval(ap50[:, i], level),
                recall=_percentile_interval(recall[:, i], level),
            )
            for i, c in enumerate(class_ids)
        ],
    )


def _percentile_interval(values: npt.NDArray[np.float64], level: float) -> Interval | None:
    kept = values[~np.isnan(values)]
    if kept.size == 0:
        return None
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(kept, [tail, 100 - tail])
    return Interval(low=_round(low), high=_round(high))


class Predictor(Protocol):
    """Anything that turns an image file into predicted boxes."""

    @property
    def server(self) -> dict[str, str]:
        """Return what the model's host reported about how it ran the model."""
        ...

    def predict(self, image: Path) -> list[Prediction]:
        """Return the predictions for one image, in its own pixels."""
        ...


def parse_response(
    response: Mapping[str, Any], model_name: str, server: dict[str, str]
) -> list[Prediction]:
    """Turn one hosted inference response into predictions.

    The response names the model the server ran, including the workspace,
    which is made from an email address. Only the part after the slash is
    compared and nothing else from it is kept.

    Args:
        response: The JSON the server returned for one image.
        model_name: The model name the response must report.
        server: Filled in with the backend and quantization the server used.

    Raises:
        ConfigError: If the server ran a different model.
    """
    resolved = response.get("resolved_model")
    if resolved is not None:
        name = str(resolved.get("model_id", "")).rsplit("/", 1)[-1]
        if name != model_name:
            raise ConfigError(f"the server ran {name!r}, not {model_name!r}")
        for key in ("backend", "quantization"):
            if key in resolved:
                server[key] = str(resolved[key])
    predictions = []
    for p in response["predictions"]:
        x, y, w, h = (float(p[k]) for k in ("x", "y", "width", "height"))
        predictions.append(
            Prediction(
                class_name=p["class"],
                confidence=float(p["confidence"]),
                xyxy=(x - w / 2, y - h / 2, x + w / 2, y + h / 2),
            )
        )
    return predictions


def inference_errors() -> tuple[type[Exception], ...]:
    """Return the exceptions a hosted inference call raises for a network or API failure."""
    import requests

    try:
        from inference_sdk.http.errors import HTTPClientError
    except ImportError:
        return (requests.RequestException,)
    return (HTTPClientError, requests.RequestException)


class HostedPredictor:
    """Call a model hosted on Roboflow through inference-sdk."""

    def __init__(self, api_url: str, api_key: str, entry: ModelEntry, floor: float) -> None:
        """Connect to the hosted model named by ``entry``.

        Raises:
            ConfigError: If inference-sdk is not installed.
        """
        try:
            from inference_sdk import InferenceConfiguration, InferenceHTTPClient
        except ImportError as e:
            raise ConfigError("frc-evaluate needs the inference extra. Run make setup-infer") from e
        self._client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self._client.configure(
            InferenceConfiguration(confidence_threshold=floor, api_key_transport="both")
        )
        self._model_ref = f"{entry.project}/{entry.version}"
        self._model_name = entry.model_id
        self._server: dict[str, str] = {}

    @property
    def server(self) -> dict[str, str]:
        """Return the backend and quantization the server last reported."""
        return self._server

    def predict(self, image: Path) -> list[Prediction]:
        """Return the hosted model's predictions for one image."""
        response = self._client.infer(str(image), model_id=self._model_ref)
        return parse_response(response, self._model_name, self._server)


class PerImageLimit(_Record):
    """How often the prediction limit per image and class was reached in one run."""

    # The limit applies to each class separately. Runs of one-class models
    # are unaffected, so the name is kept for the runs already published.
    per_image: int
    # Images where some class holds exactly ``per_image`` predictions. The
    # cache cannot tell whether more were dropped, so these are the images
    # that may have lost some.
    images_at_limit: int
    # Images where some class is at the limit and its least confident kept
    # box is at or above the threshold, so boxes that would have counted at
    # the threshold were dropped.
    images_above_threshold: int


def per_image_limit(
    predictions: Sequence[Sequence[Prediction]], per_image: int, confidence: float
) -> PerImageLimit:
    """Count the images where some class's predictions reach ``per_image`` after the limit."""
    at_limit = 0
    above = 0
    for preds in predictions:
        by_class: dict[str, list[float]] = defaultdict(list)
        for p in preds:
            by_class[p.class_name].append(p.confidence)
        full = [c for c in by_class.values() if len(c) >= per_image]
        at_limit += bool(full)
        above += any(min(c) >= confidence for c in full)
    return PerImageLimit(
        per_image=per_image, images_at_limit=at_limit, images_above_threshold=above
    )


class RunResult(_Record):
    """Everything ``metrics.json`` holds for one run."""

    run_id: str
    model: str
    dataset: str
    split: str
    limit: int | None
    predictions_from: str
    scored_classes: list[str]
    # Labeled in the dataset but missing from the model's training data.
    unscored_classes: list[str] = []
    # None in runs scored before the limit was recorded.
    per_image_limit: PerImageLimit | None = None
    # None in runs scored without ignoring labels at the frame edge (D-027).
    edge_ignore: IgnoredAtEdge | None = None
    metrics: EvalMetrics
    units: list[UnitCount]
    bootstrap: BootstrapResult


def _run_id(model: str, dataset: str, split: str, limit: int | None) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slice_tag = "" if limit is None else f"__first{limit}"
    return f"{model}__{dataset}-{split}{slice_tag}__{stamp}"


def _iso_dates(value: object) -> str:
    # YAML reads a bare date such as 2026-09-27 as a date object.
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _json_text(data: Any) -> str:
    return json.dumps(data, indent=2, default=_iso_dates) + "\n"


def _predict_all(
    predictor: Predictor, split_dir: Path, names: Sequence[str]
) -> dict[str, list[Prediction]]:
    predictions = {}
    for n, name in enumerate(names, start=1):
        predictions[name] = predictor.predict(split_dir / name)
        if n % LOG_EVERY == 0 or n == len(names):
            logger.info("predicted %d of %d images", n, len(names))
    return predictions


def _file_hashes(paths: Sequence[Path]) -> dict[str, str | None]:
    return {str(p): sha256_file(p) if p.is_file() else None for p in paths}


def run_meta(
    *,
    repo: Path,
    project: ProjectConfig,
    datasets_config: Path,
    config_files: Sequence[Path],
    run_id: str,
    model: str,
    dataset: str,
    split: str,
    split_dir: Path,
    images: int,
    limit: int | None,
    predictions_from: str,
    server: Mapping[str, str],
    argv: Sequence[str],
    dirty: bool,
    source_dirty: bool,
) -> dict[str, Any]:
    """Return what SPEC section 3.5 asks every eval record to keep.

    ``source_dirty`` is True when the predictions were rescored from a run
    made from a dirty tree, which taints this run as much as its own tree.
    """
    datasets = load_yaml(datasets_config, DatasetsConfig).datasets
    manifest = _manifest(project, dataset)
    raw_models = yaml.safe_load(project.evaluate.models_file.read_text(encoding="utf-8"))
    spec = datasets.get(dataset)
    return {
        "run_id": run_id,
        "created": utc_timestamp(),
        "command": ["frc-evaluate", *argv],
        "git": {"commit": git_commit(repo), "tree": git_tree(repo), "dirty": dirty},
        "seed": project.seed,
        "configs": _file_hashes(config_files),
        "dataset": {
            "key": dataset,
            "universe": None if spec is None else spec.model_dump(mode="json"),
            "manifest_sha256": sha256_file(manifest),
            "split": split,
            "annotations_sha256": sha256_file(split_dir / ANNOTATIONS_NAME),
            "images": images,
            "limit": limit,
        },
        "model": {"name": model, **raw_models["models"][model], "server": dict(server)},
        "predictions_from": predictions_from,
        "source_dirty": source_dirty,
        "evaluate": project.evaluate.model_dump(mode="json"),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "packages": package_versions(),
    }


def _manifest(project: ProjectConfig, dataset: str) -> Path:
    """Return the committed manifest digest that ties a run to its raw data.

    Raises:
        ConfigError: If it is missing, so the data behind a run could not be
            traced.
    """
    manifest = project.paths.manifests_dir / f"{dataset}.sha256"
    if not manifest.is_file():
        raise ConfigError(f"{manifest} is missing. Run make download for {dataset}")
    return manifest


def run_evaluation(
    *,
    repo: Path,
    project: ProjectConfig,
    datasets_config: Path,
    config_files: Sequence[Path],
    model: str,
    dataset: str,
    split: str,
    limit: int | None,
    from_cache: str | None,
    make_predictor: Callable[[ModelEntry], Predictor],
    argv: Sequence[str],
    dirty: bool,
) -> Path:
    """Score ``model`` on one harmonized split and write its run directory.

    Args:
        repo: The repository root, for git state.
        project: Settings from ``configs/project.yaml``.
        datasets_config: The Universe datasets config, for the dataset's version.
        config_files: Every file whose hash the run should record.
        model: Run name in ``reports/models.yaml``.
        dataset: Harmonized dataset key.
        split: train, valid, or test.
        limit: Score only the first ``limit`` images, sorted by file name.
        from_cache: Run id whose predictions to score again, instead of
            calling the model.
        make_predictor: Builds the predictor for a model entry. Called only
            when predictions are not cached.
        argv: The command-line arguments, recorded in ``meta.json``.
        dirty: Whether the working tree had uncommitted changes.

    Returns:
        The new run directory.

    Raises:
        ConfigError: If the model, dataset, split, or cached run is unknown.
    """
    cfg = project.evaluate
    entries = load_yaml(cfg.models_file, ModelsFile).models
    if model not in entries:
        raise ConfigError(f"{model} is not in {cfg.models_file}")
    split_dir = project.paths.harmonized_dir / dataset / split
    if not (split_dir / ANNOTATIONS_NAME).is_file():
        raise ConfigError(f"{split_dir} has no annotations. Run make harmonize first")
    _manifest(project, dataset)

    ds = sv.DetectionDataset.from_coco(
        images_directory_path=str(split_dir), annotations_path=str(split_dir / ANNOTATIONS_NAME)
    )
    records = sorted(load_split(split_dir, dataset), key=lambda r: r.ref.file_name)[:limit]
    names = [r.ref.file_name for r in records]
    if not names:
        raise ConfigError(f"{split_dir} has no images")
    targets = [ds.annotations[str(split_dir / name)] for name in names]
    labeled = scored_classes(cfg.coverage_file, dataset)
    scored = shared_classes(labeled, scored_classes(cfg.coverage_file, entries[model].dataset))
    run_id = _run_id(model, dataset, split, limit)
    run_dir = cfg.runs_dir / run_id
    if run_dir.exists():
        raise ConfigError(f"{run_dir} already exists")

    method = project.splits.datasets.get(dataset)
    hashes = (
        [phash(split_dir / n) for n in names] if method and method.method != "temporal" else None
    )
    units = image_units(records, method, hashes, project.inspect.near_duplicate_max_distance)
    _check_units(units, project, dataset, split, limit)

    server: dict[str, str] = {}
    cache_text = None
    source_dirty = False
    if from_cache is None:
        predictor = make_predictor(entries[model])
        cache_text = predictions_text(
            _predict_all(predictor, split_dir, names),
            cfg.confidence_floor,
            cfg.max_predictions_per_image,
            cfg.max_prediction_bytes,
            scored,
        )
        # Score the rounded predictions the cache keeps, so rescoring the
        # cache reproduces this run exactly.
        predictions = parse_predictions(cache_text)
        server = predictor.server
        source = THIS_RUN
    else:
        predictions, source_dirty = _cached_run(cfg.runs_dir / from_cache, model, dataset, split)
        source = from_cache
    # Older caches predate the limit, so rescores apply it too.
    predictions = {
        n: limit_predictions(p, cfg.max_predictions_per_image, scored)
        for n, p in predictions.items()
    }
    detections = align(predictions, names, ds.classes)
    ignored = None
    if cfg.edge_ignore is not None:
        detections, targets, records, ignored = apply_edge_ignore(
            detections,
            targets,
            records,
            ds.classes,
            [c for c in cfg.edge_ignore.classes if c in scored],
            cfg.edge_ignore.tolerance_px,
            cfg.edge_ignore.min_overlap,
            cfg.iou,
        )

    result = RunResult(
        run_id=run_id,
        model=model,
        dataset=dataset,
        split=split,
        limit=limit,
        predictions_from=source,
        scored_classes=scored,
        unscored_classes=[name for name in labeled if name not in scored],
        metrics=compute_metrics(
            detections,
            targets,
            ds.classes,
            scored,
            confidence=cfg.confidence,
            iou=cfg.iou,
            thresholds=cfg.pr_thresholds.values,
        ),
        per_image_limit=per_image_limit(
            [predictions[n] for n in names], cfg.max_predictions_per_image, cfg.confidence
        ),
        edge_ignore=ignored,
        units=unit_counts(records, units, scored),
        bootstrap=bootstrap(
            detections,
            targets,
            units,
            ds.classes,
            scored,
            confidence=cfg.confidence,
            resamples=cfg.bootstrap.resamples,
            level=cfg.bootstrap.level,
            seed=project.seed,
        ),
    )
    meta = run_meta(
        repo=repo,
        project=project,
        datasets_config=datasets_config,
        config_files=config_files,
        run_id=run_id,
        model=model,
        dataset=dataset,
        split=split,
        split_dir=split_dir,
        images=len(names),
        limit=limit,
        predictions_from=source,
        server=server,
        argv=argv,
        dirty=dirty,
        source_dirty=source_dirty,
    )
    # Every file is rendered first, so a failure leaves no partial run behind.
    files = {METRICS_NAME: _json_text(result.model_dump(mode="json")), META_NAME: _json_text(meta)}
    if cache_text is not None:
        files[PREDICTIONS_NAME] = cache_text
    run_dir.mkdir(parents=True)
    for name, text in files.items():
        (run_dir / name).write_text(text, encoding="utf-8")
    logger.info(
        "wrote %s (%s)", run_dir, ", ".join(f"{n} {len(t)} bytes" for n, t in files.items())
    )
    return run_dir


def _cached_run(
    run_dir: Path, model: str, dataset: str, split: str
) -> tuple[dict[str, list[Prediction]], bool]:
    """Return an earlier run's predictions, and whether that run came from a dirty tree.

    Raises:
        ConfigError: If the run is missing or scored another model, dataset,
            or split, since its predictions would be published under the
            wrong name.
    """
    cache = run_dir / PREDICTIONS_NAME
    if not cache.is_file():
        raise ConfigError(f"{cache} does not exist")
    earlier = RunResult.model_validate_json((run_dir / METRICS_NAME).read_text(encoding="utf-8"))
    if (earlier.model, earlier.dataset, earlier.split) != (model, dataset, split):
        raise ConfigError(
            f"{run_dir.name} scored {earlier.model} on {earlier.dataset} {earlier.split}, "
            f"not {model} on {dataset} {split}"
        )
    meta = json.loads((run_dir / META_NAME).read_text(encoding="utf-8"))
    return load_predictions(cache), bool(meta["git"]["dirty"] or meta.get("source_dirty", False))


def _check_units(
    units: Sequence[str], project: ProjectConfig, dataset: str, split: str, limit: int | None
) -> None:
    """Fail if the rebuilt units disagree with the split report, unless only a slice was scored.

    Raises:
        ConfigError: If the unit count differs from ``reports/splits.json``.
    """
    report = project.paths.reports_dir / SPLITS_NAME
    if limit is not None or not report.is_file():
        return
    expected = json.loads(report.read_text(encoding="utf-8"))["datasets"].get(dataset)
    if expected is None:
        return
    want = expected["splits"][split]["units"]
    if len(set(units)) != want:
        raise ConfigError(
            f"{dataset} {split} rebuilt {len(set(units))} units, but {report} records {want}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation command line and return the exit code."""
    parser = argparse.ArgumentParser(
        description="Score a model on a harmonized split and write reports/runs/<run_id>/."
    )
    parser.add_argument("model", help="run name in reports/models.yaml, such as baseline-a")
    parser.add_argument("dataset", help="harmonized dataset key, such as marswars")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--limit", type=int, help="score only the first N images, by file name")
    parser.add_argument("--from-cache", metavar="RUN_ID", help="score this run's predictions again")
    parser.add_argument("--allow-dirty", action="store_true", help="run from a dirty tree")
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    parser.add_argument("--datasets-config", type=Path, default=DATASETS_CONFIG)
    add_log_level_argument(parser)
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)
    setup_logging(args.log_level)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    repo = Path.cwd()
    dirty = git_is_dirty(repo)
    if dirty and not args.allow_dirty:
        raise DirtyTreeError("commit or stash changes first, or pass --allow-dirty")
    project = load_yaml(args.project_config, ProjectConfig)
    cfg = project.evaluate
    config_files = [
        args.project_config,
        args.datasets_config,
        cfg.models_file,
        cfg.coverage_file,
        project.paths.reports_dir / SPLITS_NAME,
    ]

    api_key = "" if args.from_cache else redacted_api_key()

    def hosted(entry: ModelEntry) -> Predictor:
        return HostedPredictor(cfg.api_url, api_key, entry, cfg.confidence_floor)

    try:
        run_evaluation(
            repo=repo,
            project=project,
            datasets_config=args.datasets_config,
            config_files=config_files,
            model=args.model,
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            from_cache=args.from_cache,
            make_predictor=hosted,
            argv=raw_args,
            dirty=dirty,
        )
    except inference_errors() as e:
        # These can carry the key in a request URL.
        logger.error("inference failed: %s", redact(f"{type(e).__name__}: {e}", api_key))
        return 1
    return 0
