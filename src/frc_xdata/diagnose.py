"""Find where a model's errors fall, from the predictions its runs cached.

``frc-diagnose`` reads the published runs of one model, matches every
prediction to the labels the same way the run's confusion matrix did, and
slices the hits, false positives, and misses by box size, brightness, blur,
crowding, and source. It never calls the model.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import supervision as sv

FalsePositiveKind = Literal["duplicate", "localization", "inside_unscored", "background"]
FALSE_POSITIVE_KINDS: tuple[FalsePositiveKind, ...] = (
    "duplicate",
    "localization",
    "inside_unscored",
    "background",
)


@dataclass(frozen=True)
class Match:
    """How one image's predictions pair up with its labels, by index.

    Pairs are ``(prediction, label)``. A confusion is a pair whose classes
    differ, which counts as a false positive for the predicted class and a
    miss for the labeled one.
    """

    hits: tuple[tuple[int, int], ...]
    confusions: tuple[tuple[int, int], ...]
    false_positives: tuple[int, ...]
    misses: tuple[int, ...]


def _class_ids(detections: sv.Detections) -> npt.NDArray[np.int_]:
    if detections.class_id is None:
        raise ValueError("detections need class ids to be matched")
    return np.asarray(detections.class_id, dtype=int)


def _iou(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return np.asarray(sv.box_iou_batch(a, b), dtype=np.float64)


def _xyxy(detections: sv.Detections) -> npt.NDArray[np.float64]:
    return np.asarray(detections.xyxy, dtype=np.float64)


def iou_matrix(labels: sv.Detections, predictions: sv.Detections) -> npt.NDArray[np.float64]:
    """Return IoU with labels as rows and predictions as columns."""
    return _iou(_xyxy(labels), _xyxy(predictions))


def match_boxes(predictions: sv.Detections, labels: sv.Detections, iou: float) -> Match:
    """Pair predictions with labels the way supervision's confusion matrix does.

    A pair needs IoU strictly above ``iou``. Pairs whose classes agree are
    taken first, then by IoU from highest to lowest, and each box joins at
    most one pair. Following the same rule means the counts here add up to
    the hits, false positives, and misses a run reports.

    Args:
        predictions: One image's predictions, already cut at the confidence
            threshold.
        labels: The image's labeled boxes.
        iou: Overlap a pair must exceed.
    """
    overlaps = iou_matrix(labels, predictions)
    label_ids, pred_ids = np.nonzero(overlaps > iou)
    same = _class_ids(labels)[label_ids] == _class_ids(predictions)[pred_ids]
    order = np.lexsort((-overlaps[label_ids, pred_ids], ~same))
    hits: list[tuple[int, int]] = []
    confusions: list[tuple[int, int]] = []
    used_labels: set[int] = set()
    used_preds: set[int] = set()
    for k in order:
        label, pred = int(label_ids[k]), int(pred_ids[k])
        if label in used_labels or pred in used_preds:
            continue
        (hits if same[k] else confusions).append((pred, label))
        used_labels.add(label)
        used_preds.add(pred)
    return Match(
        hits=tuple(hits),
        confusions=tuple(confusions),
        false_positives=tuple(i for i in range(len(predictions)) if i not in used_preds),
        misses=tuple(i for i in range(len(labels)) if i not in used_labels),
    )


def _inside(point: tuple[float, float], boxes: npt.NDArray[np.float64]) -> bool:
    x, y = point
    return bool(
        np.any((boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3]))
    )


def false_positive_kind(
    box: npt.NDArray[np.float64],
    labels: sv.Detections,
    matched: Sequence[int],
    unscored: npt.NDArray[np.float64],
    iou: float,
    localization_floor: float,
) -> FalsePositiveKind:
    """Say why a false positive matched no label.

    A duplicate overlaps a label that another prediction already took. A
    localization error overlaps a label, but not enough to count, which
    often means the label or the prediction is loose. A box whose center
    lies inside a labeled box of a class that is not scored, such as a
    robot on a fuel-only model, is kept apart, since the model may be
    firing on part of that object. Anything else is background.

    Args:
        box: The false positive as ``[x1, y1, x2, y2]``.
        labels: The image's labeled boxes of the scored classes.
        matched: Indices into ``labels`` that some prediction took.
        unscored: Labeled boxes of unscored classes, shape ``(n, 4)``.
        iou: Overlap a hit must exceed.
        localization_floor: Overlap a box must exceed to count as a
            localization error instead of background.
    """
    if len(labels):
        overlaps = _iou(_xyxy(labels), box[None, :])[:, 0]
        taken = np.zeros(len(labels), dtype=bool)
        taken[list(matched)] = True
        if np.any(taken & (overlaps > iou)):
            return "duplicate"
        if np.any(overlaps > localization_floor):
            return "localization"
    center = (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)
    if len(unscored) and _inside(center, unscored):
        return "inside_unscored"
    return "background"
