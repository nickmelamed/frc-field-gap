"""Find where a model's errors fall, from the predictions its runs cached.

``frc-diagnose`` reads the published runs of one model, matches every
prediction to the labels the same way the run's confusion matrix did, and
slices the hits, false positives, and misses by box size, brightness, blur,
crowding, and source. It never calls the model.
"""

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import supervision as sv
from PIL import Image
from pydantic import BaseModel, ConfigDict

from frc_xdata.config import (
    AreaBuckets,
    DiagnoseConfig,
    GalleryConfig,
    ModelsFile,
    ProjectConfig,
    ReviewConfig,
    SourcePattern,
    load_yaml,
)
from frc_xdata.download import PROJECT_CONFIG
from frc_xdata.errors import ConfigError, CountMismatchError, DirtyTreeError
from frc_xdata.evaluate import (
    DECIMALS,
    RunResult,
    above,
    align,
    compute_metrics,
    limit_predictions,
    load_predictions,
    restrict,
)
from frc_xdata.harmonize import SPLITS_NAME
from frc_xdata.inspect_datasets import (
    ANNOTATIONS_NAME,
    Box,
    ImageRecord,
    area_bucket,
    encode_png,
    load_split,
    tile,
)
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.provenance import (
    git_commit,
    git_is_dirty,
    git_tree,
    package_versions,
    sha256_file,
    utc_timestamp,
)
from frc_xdata.runs import load_runs, predictions_path, select_runs
from frc_xdata.splits import recording_key

DIAGNOSIS_NAME = "diagnosis.json"
META_NAME = "meta.json"
GALLERY_NAME = "failures.png"
REVIEW_SHEET_PREFIX = "review_"
# Brightness and blur are on scales of tens to thousands, so one decimal
# is enough to name a bin.
FEATURE_DECIMALS = 1
SIZE_BUCKETS = ("small", "medium", "large")
OTHER_SOURCE = "other"
ErrorKind = Literal["false positive", "miss"]
ERROR_KINDS: tuple[ErrorKind, ...] = ("false positive", "miss")
# Labeled boxes, false positives, and misses, in colors that stay apart for
# the common kinds of color blindness.
LABEL_COLOR = sv.Color(0, 158, 115)
FALSE_POSITIVE_COLOR = sv.Color(213, 94, 0)
MISS_COLOR = sv.Color(86, 180, 233)
REVIEW_FIELDS = ("dataset", "file_name", "kind", "x1", "y1", "x2", "y2", "verdict", "note")

logger = logging.getLogger(__name__)

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
    the hits, false positives, and misses a run reports. ``predictions``
    must already be cut at the confidence threshold.
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


def image_features(image: Image.Image, side: int) -> tuple[float, float]:
    """Return the mean gray level and the variance of the Laplacian of ``image``.

    The image is first stretched to ``side`` pixels square, the way the model
    sees it. Gray levels run from 0 to 255. A low Laplacian variance means few
    sharp edges, which usually means blur.
    """
    gray = np.asarray(
        image.convert("L").resize((side, side), Image.Resampling.BILINEAR), dtype=np.float64
    )
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    return float(gray.mean()), float(laplacian.var())


def quantile_edges(values: Sequence[float], bins: int) -> list[float]:
    """Return the ``bins - 1`` edges that cut ``values`` into equal-count bins."""
    if not values:
        raise ValueError("no values to cut into bins")
    cuts = np.quantile(np.asarray(values, dtype=np.float64), np.arange(1, bins) / bins)
    return [round(float(c), FEATURE_DECIMALS) for c in cuts]


def bin_index(value: float, edges: Sequence[float]) -> int:
    """Return the bin a value falls in. A value on an edge goes to the higher bin."""
    return int(np.searchsorted(np.asarray(edges), value, side="right"))


def bin_names(edges: Sequence[float]) -> list[str]:
    """Name the bins that ``edges`` make, lowest first."""
    if not edges:
        return ["all"]
    inner = [f"{a} to {b}" for a, b in pairwise(edges)]
    return [f"below {edges[0]}", *inner, f"{edges[-1]} and above"]


def count_names(edges: Sequence[int]) -> list[str]:
    """Name bins of whole counts whose lower edges are ``edges``."""
    names = [str(a) if b - a == 1 else f"{a} to {b - 1}" for a, b in pairwise(edges)]
    return [*names, f"{edges[-1]} or more"]


def source_of(source_name: str, patterns: Sequence[SourcePattern]) -> str:
    """Return the first pattern name that matches, or ``other``."""
    for p in patterns:
        if re.search(p.pattern, source_name):
            return p.name
    return OTHER_SOURCE


def relative_side(box: Box, width: int, height: int) -> float:
    """Return the side of a square with the box's share of the image area, as a fraction."""
    return float(np.sqrt(box.w * box.h / (width * height)))


def _box(xyxy: npt.NDArray[np.float64]) -> Box:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return Box("", x1, y1, x2 - x1, y2 - y1)


@dataclass(frozen=True)
class ImageErrors:
    """One image's hits, false positives, and misses for one class."""

    matched_labels: tuple[int, ...]
    missed_labels: tuple[int, ...]
    false_positive_boxes: npt.NDArray[np.float64]
    false_positive_scores: tuple[float, ...]
    false_positive_kinds: tuple[FalsePositiveKind, ...]


def image_errors(
    predictions: sv.Detections,
    labels: sv.Detections,
    unscored: npt.NDArray[np.float64],
    confidence: float,
    iou: float,
    localization_floor: float,
) -> ImageErrors:
    """Match one image's predictions of one class at ``confidence`` and type each false positive.

    Args:
        predictions: The image's predictions of the class, at every confidence.
        labels: The image's labeled boxes of the class.
        unscored: Labeled boxes of classes that are not scored, shape ``(n, 4)``.
        confidence: Threshold a prediction must reach to count.
        iou: Overlap a hit must exceed.
        localization_floor: See :func:`false_positive_kind`.
    """
    kept = above(predictions, confidence)
    match = match_boxes(kept, labels, iou)
    matched = tuple(label for _, label in match.hits)
    boxes = _xyxy(kept)[list(match.false_positives)].reshape(-1, 4)
    scores = np.asarray(kept.confidence, dtype=np.float64)[list(match.false_positives)]
    return ImageErrors(
        matched_labels=matched,
        missed_labels=match.misses,
        false_positive_boxes=boxes,
        false_positive_scores=tuple(float(c) for c in scores),
        false_positive_kinds=tuple(
            false_positive_kind(b, labels, matched, unscored, iou, localization_floor)
            for b in boxes
        ),
    )


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SliceScore(_Record):
    """Scores for the images in one slice, at the run's threshold."""

    name: str
    images: int
    labeled: int
    hits: int
    false_positives: int
    misses: int
    # None where undefined, as in a run's metrics (D-018).
    map50: float | None
    precision: float | None
    recall: float | None
    false_positives_per_image: float
    false_positive_kinds: dict[str, int]


class Slicing(_Record):
    """One way of cutting a split into groups of images."""

    by: str
    slices: list[SliceScore]


class SizeRow(_Record):
    """Hits and misses of labeled boxes, and false positives, of one relative size."""

    bucket: str
    labeled: int
    hits: int
    recall: float | None
    false_positives: int


class ClassDiagnosis(_Record):
    """Where one scored class's errors fall in one split."""

    name: str
    overall: SliceScore
    slicings: list[Slicing]
    sizes: list[SizeRow]


class SplitDiagnosis(_Record):
    """The diagnosis of one published run."""

    run_id: str
    predictions_from: str
    dataset: str
    split: str
    max_predictions_per_image: int
    classes: list[ClassDiagnosis]


class FeatureSummary(_Record):
    """Quantiles of image and box features for one split, to compare domains."""

    dataset: str
    split: str
    role: Literal["training", "test"]
    images: int
    boxes: int
    brightness: list[float]
    sharpness: list[float]
    box_side: list[float]
    boxes_per_image: list[float]


class GalleryTile(_Record):
    """One tile of the failure gallery, row by row from the top left."""

    dataset: str
    file_name: str
    kind: str
    source: str


class ReviewCount(_Record):
    """Verdicts on the sampled errors of one kind in one dataset."""

    dataset: str
    kind: str
    total: int
    sampled: int
    verdicts: dict[str, int]
    # Verdicts by the automatic false positive kind, for false positives.
    by_automatic_kind: dict[str, dict[str, int]]


class Diagnosis(_Record):
    """Everything ``diagnosis.json`` holds for one model."""

    model: str
    confidence: float
    iou: float
    feature_px: int
    localization_floor: float
    area_buckets: AreaBuckets
    quantiles: list[float]
    brightness_edges: list[float]
    sharpness_edges: list[float]
    splits: list[SplitDiagnosis]
    domain: list[FeatureSummary]
    gallery: list[GalleryTile]
    review: list[ReviewCount]


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, DECIMALS) if whole else None


def score_slice(
    name: str,
    indices: Sequence[int],
    predictions: Sequence[sv.Detections],
    labels: Sequence[sv.Detections],
    errors: Sequence[ImageErrors],
    classes: Sequence[str],
    class_name: str,
    confidence: float,
    iou: float,
) -> SliceScore:
    """Score one class on the images at ``indices``.

    Scores come from the same function a run uses, so a slice of every image
    reproduces the run's numbers.

    Raises:
        CountMismatchError: If the per-box matching disagrees with the
            confusion matrix on this slice.
    """
    preds = [predictions[i] for i in indices]
    targets = [labels[i] for i in indices]
    (c,) = compute_metrics(
        preds, targets, classes, [class_name], confidence, iou, [confidence]
    ).classes
    errs = [errors[i] for i in indices]
    hits = sum(len(e.matched_labels) for e in errs)
    misses = sum(len(e.missed_labels) for e in errs)
    kinds = Counter(k for e in errs for k in e.false_positive_kinds)
    fps = sum(kinds.values())
    if (hits, fps, misses) != (c.true_positives, c.false_positives, c.false_negatives):
        raise CountMismatchError(
            f"slice {name!r}: matched {hits} hits, {fps} false positives, {misses} misses, "
            f"but the confusion matrix has {c.true_positives}, {c.false_positives}, "
            f"{c.false_negatives}"
        )
    return SliceScore(
        name=name,
        images=len(indices),
        labeled=c.instances,
        hits=hits,
        false_positives=fps,
        misses=misses,
        map50=c.map50,
        precision=c.precision,
        recall=c.recall,
        false_positives_per_image=round(fps / len(indices), DECIMALS),
        false_positive_kinds={k: kinds.get(k, 0) for k in FALSE_POSITIVE_KINDS},
    )


def slicing(
    by: str,
    groups: Sequence[str],
    order: Sequence[str],
    score: "SliceScorer",
) -> Slicing:
    """Score each named group of images, in ``order``, leaving out empty groups.

    ``groups`` holds the group of each image, and ``score`` is called once
    per non-empty group with its image indices.
    """
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    unknown = sorted(set(members) - set(order))
    return Slicing(
        by=by,
        slices=[score(name, members[name]) for name in [*order, *unknown] if name in members],
    )


class SliceScorer:
    """Scores slices of one class in one split."""

    def __init__(
        self,
        predictions: Sequence[sv.Detections],
        labels: Sequence[sv.Detections],
        errors: Sequence[ImageErrors],
        classes: Sequence[str],
        class_name: str,
        confidence: float,
        iou: float,
    ) -> None:
        """Keep one class's per-image detections and errors."""
        self._args = (predictions, labels, errors, classes, class_name, confidence, iou)

    def __call__(self, name: str, indices: Sequence[int]) -> SliceScore:
        """Score the images at ``indices`` under ``name``."""
        return score_slice(name, indices, *self._args)


def size_rows(
    records: Sequence[ImageRecord],
    labels: Sequence[sv.Detections],
    errors: Sequence[ImageErrors],
    buckets: AreaBuckets,
) -> list[SizeRow]:
    """Count hits and misses by labeled box size, and false positives by their own size.

    Sizes are shares of the image area (``inspect.area_buckets``), so they
    mean the same thing at every resolution, unlike COCO's pixel sizes.
    """
    labeled: Counter[str] = Counter()
    hit: Counter[str] = Counter()
    false: Counter[str] = Counter()
    for r, lab, e in zip(records, labels, errors, strict=True):
        matched = set(e.matched_labels)
        for i, xyxy in enumerate(_xyxy(lab)):
            bucket = area_bucket(_box(xyxy), r.width, r.height, buckets)
            labeled[bucket] += 1
            hit[bucket] += i in matched
        for xyxy in e.false_positive_boxes:
            false[area_bucket(_box(xyxy), r.width, r.height, buckets)] += 1
    return [
        SizeRow(
            bucket=b,
            labeled=labeled[b],
            hits=hit[b],
            recall=_share(hit[b], labeled[b]),
            false_positives=false[b],
        )
        for b in SIZE_BUCKETS
    ]


def quantiles_of(values: Sequence[float], quantiles: Sequence[float]) -> list[float]:
    """Return the quantiles of ``values``, rounded, or an empty list when there are none."""
    if not values:
        return []
    found = np.quantile(np.asarray(values, dtype=np.float64), quantiles)
    return [round(float(v), DECIMALS) for v in found]


@dataclass(frozen=True)
class SplitCases:
    """One published run's images, labels, and cached predictions."""

    result: RunResult
    split_dir: Path
    records: list[ImageRecord]
    classes: list[str]
    predictions: list[sv.Detections]
    labels: list[sv.Detections]


def load_cases(project: ProjectConfig, result: RunResult) -> SplitCases:
    """Load a run's split and its cached predictions, under the run's per-image limit.

    Raises:
        ConfigError: If the split or the cache is missing, or the cache does
            not cover every image.
    """
    split_dir = project.paths.harmonized_dir / result.dataset / result.split
    if not (split_dir / ANNOTATIONS_NAME).is_file():
        raise ConfigError(f"{split_dir} has no annotations. Run make harmonize first")
    cache = predictions_path(project.evaluate.runs_dir, result)
    if not cache.is_file():
        raise ConfigError(f"{result.run_id} reads predictions from {cache}, which is missing")
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=str(split_dir), annotations_path=str(split_dir / ANNOTATIONS_NAME)
    )
    records = sorted(load_split(split_dir, result.dataset), key=lambda r: r.ref.file_name)
    names = [r.ref.file_name for r in records]
    limit = (
        project.evaluate.max_predictions_per_image
        if result.per_image_limit is None
        else result.per_image_limit.per_image
    )
    cached = {
        n: limit_predictions(p, limit, result.scored_classes)
        for n, p in load_predictions(cache).items()
    }
    return SplitCases(
        result=result,
        split_dir=split_dir,
        records=records,
        classes=list(ds.classes),
        predictions=align(cached, names, ds.classes),
        labels=[ds.annotations[str(split_dir / n)] for n in names],
    )


def split_features(
    split_dir: Path, records: Sequence[ImageRecord], side: int
) -> list[tuple[float, float]]:
    """Return brightness and sharpness for every image, in the order of ``records``."""

    def one(r: ImageRecord) -> tuple[float, float]:
        with Image.open(split_dir / r.ref.file_name) as image:
            return image_features(image, side)

    with ThreadPoolExecutor() as pool:
        return list(pool.map(one, records))


def image_sources(
    project: ProjectConfig, dataset: str, records: Sequence[ImageRecord]
) -> tuple[list[str], list[str]] | None:
    """Return each image's source and the order to report sources in.

    Recordings name the source for datasets cut by frame order, reported in
    name order. Other datasets need patterns under ``diagnose.sources``,
    reported in the patterns' order. Returns None when neither applies.
    """
    patterns = project.diagnose.sources.get(dataset)
    if patterns is not None:
        sources = [source_of(r.source_name, patterns) for r in records]
        return sources, [*(p.name for p in patterns), OTHER_SOURCE]
    method = project.splits.datasets.get(dataset)
    if method is None or method.method != "temporal":
        return None
    keys = [recording_key(r.source_name, method.recording_pattern) or OTHER_SOURCE for r in records]
    return keys, sorted(set(keys))


def check_run_totals(cases: SplitCases) -> None:
    """Fail if matching every scored class together does not reproduce the run's counts.

    The slices match one class at a time, which for a single scored class is
    the run's own matching. With several classes the run can also pair a box
    with a label of another class, so the whole split is matched again here
    with every scored class at once, the way the run did.

    Raises:
        CountMismatchError: If any class's hits, false positives, or misses
            differ from the run's published counts.
    """
    r = cases.result
    scored_ids = [cases.classes.index(c) for c in r.scored_classes]
    counts = {c: [0, 0, 0] for c in scored_ids}
    for preds, labels in zip(cases.predictions, cases.labels, strict=True):
        kept = above(restrict(preds, scored_ids), r.metrics.confidence)
        wanted = restrict(labels, scored_ids)
        match = match_boxes(kept, wanted, r.metrics.iou)
        pred_ids, label_ids = _class_ids(kept), _class_ids(wanted)
        for p, _ in match.hits:
            counts[int(pred_ids[p])][0] += 1
        for p, lab in match.confusions:
            counts[int(pred_ids[p])][1] += 1
            counts[int(label_ids[lab])][2] += 1
        for p in match.false_positives:
            counts[int(pred_ids[p])][1] += 1
        for lab in match.misses:
            counts[int(label_ids[lab])][2] += 1
    published = {c.name: c for c in r.metrics.classes}
    for cid in scored_ids:
        c = published[cases.classes[cid]]
        got = tuple(counts[cid])
        want = (c.true_positives, c.false_positives, c.false_negatives)
        if got != want:
            raise CountMismatchError(
                f"{r.run_id} {c.name}: rebuilt {got}, but the run published {want}"
            )


@dataclass(frozen=True)
class ClassErrors:
    """One scored class's detections and per-image errors in one split."""

    name: str
    predictions: list[sv.Detections]
    labels: list[sv.Detections]
    errors: list[ImageErrors]


def class_errors(cases: SplitCases, localization_floor: float) -> list[ClassErrors]:
    """Match each scored class on its own, at the run's threshold and IoU."""
    r = cases.result
    scored_ids = [cases.classes.index(c) for c in r.scored_classes]
    other_ids = [i for i in range(len(cases.classes)) if i not in scored_ids]
    unscored = [_xyxy(restrict(lab, other_ids)) for lab in cases.labels]
    found = []
    for name, cid in zip(r.scored_classes, scored_ids, strict=True):
        preds = [restrict(p, [cid]) for p in cases.predictions]
        labels = [restrict(lab, [cid]) for lab in cases.labels]
        errors = [
            image_errors(p, lab, u, r.metrics.confidence, r.metrics.iou, localization_floor)
            for p, lab, u in zip(preds, labels, unscored, strict=True)
        ]
        found.append(ClassErrors(name, preds, labels, errors))
    return found


def diagnose_split(
    cases: SplitCases,
    found: Sequence[ClassErrors],
    features: Sequence[tuple[float, float]],
    sources: tuple[list[str], list[str]] | None,
    brightness_edges: Sequence[float],
    sharpness_edges: Sequence[float],
    cfg: DiagnoseConfig,
    buckets: AreaBuckets,
) -> SplitDiagnosis:
    """Slice every scored class's errors in one run.

    Raises:
        CountMismatchError: If the matching does not reproduce the run's
            published counts.
    """
    check_run_totals(cases)
    r = cases.result
    confidence, iou = r.metrics.confidence, r.metrics.iou
    brightness_names, sharpness_names = bin_names(brightness_edges), bin_names(sharpness_edges)
    crowd_names = count_names(cfg.crowding_edges)
    classes = []
    for ce in found:
        name, preds, labels, errors = ce.name, ce.predictions, ce.labels, ce.errors
        score = SliceScorer(preds, labels, errors, cases.classes, name, confidence, iou)
        overall = score("all", range(len(preds)))
        counts = [len(lab) for lab in labels]
        slicings = [
            slicing(
                "brightness",
                [brightness_names[bin_index(b, brightness_edges)] for b, _ in features],
                brightness_names,
                score,
            ),
            slicing(
                "sharpness",
                [sharpness_names[bin_index(s, sharpness_edges)] for _, s in features],
                sharpness_names,
                score,
            ),
            slicing(
                f"labeled {name} per image",
                [crowd_names[bin_index(n, cfg.crowding_edges) - 1] for n in counts],
                crowd_names,
                score,
            ),
            slicing(
                f"has labeled {name}", ["yes" if n else "no" for n in counts], ["yes", "no"], score
            ),
        ]
        if sources is not None:
            slicings.append(slicing("source", *sources, score))
        classes.append(
            ClassDiagnosis(
                name=name,
                overall=overall,
                slicings=slicings,
                sizes=size_rows(cases.records, labels, errors, buckets),
            )
        )
    diagnosis = SplitDiagnosis(
        run_id=r.run_id,
        predictions_from=r.predictions_from,
        dataset=r.dataset,
        split=r.split,
        max_predictions_per_image=(
            r.per_image_limit.per_image if r.per_image_limit is not None else 0
        ),
        classes=classes,
    )
    return diagnosis


def feature_summary(
    dataset: str,
    split: str,
    role: Literal["training", "test"],
    records: Sequence[ImageRecord],
    features: Sequence[tuple[float, float]],
    classes: Sequence[str],
    quantiles: Sequence[float],
) -> FeatureSummary:
    """Summarize image and labeled box features of the ``classes`` boxes in one split."""
    boxes = [[b for b in r.boxes if b.label in classes] for r in records]
    sides = [
        relative_side(b, r.width, r.height)
        for r, bs in zip(records, boxes, strict=True)
        for b in bs
    ]
    return FeatureSummary(
        dataset=dataset,
        split=split,
        role=role,
        images=len(records),
        boxes=len(sides),
        brightness=quantiles_of([f[0] for f in features], quantiles),
        sharpness=quantiles_of([f[1] for f in features], quantiles),
        box_side=quantiles_of(sides, quantiles),
        boxes_per_image=quantiles_of([float(len(bs)) for bs in boxes], quantiles),
    )


@dataclass(frozen=True)
class ErrorItem:
    """One false positive or missed label, located well enough to find it again."""

    dataset: str
    split: str
    file_name: str
    source: str
    kind: ErrorKind
    # The automatic kind of a false positive, empty for a miss.
    automatic_kind: str
    # Confidence of a false positive, 0 for a miss.
    score: float
    xyxy: tuple[float, float, float, float]
    # Frame number within the recording, for datasets cut by frame order.
    frame: int | None = None

    @property
    def key(self) -> tuple[str, str, str, tuple[float, float, float, float]]:
        """Identify the box in the review file."""
        return (self.dataset, self.file_name, self.kind, self.xyxy)


def _rounded(xyxy: npt.NDArray[np.float64]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (round(float(v), 1) for v in xyxy)
    return (x1, y1, x2, y2)


def error_items(
    cases: SplitCases,
    errors: Sequence[ClassErrors],
    sources: Sequence[str] | None,
    frames: Sequence[int | None],
) -> list[ErrorItem]:
    """List every false positive and missed label in one split, image by image."""
    items = []
    r = cases.result
    for ce in errors:
        for i, (record, lab, e) in enumerate(zip(cases.records, ce.labels, ce.errors, strict=True)):
            src = OTHER_SOURCE if sources is None else sources[i]
            base = (r.dataset, r.split, record.ref.file_name, src)
            for box, kind, score in zip(
                e.false_positive_boxes, e.false_positive_kinds, e.false_positive_scores, strict=True
            ):
                items.append(
                    ErrorItem(*base, "false positive", kind, score, _rounded(box), frames[i])
                )
            for m in e.missed_labels:
                items.append(ErrorItem(*base, "miss", "", 0.0, _rounded(_xyxy(lab)[m]), frames[i]))
    return items


def image_frames(
    project: ProjectConfig, dataset: str, records: Sequence[ImageRecord]
) -> list[int | None]:
    """Return each image's frame number, or None where the dataset has no recordings."""
    method = project.splits.datasets.get(dataset)
    if method is None or method.method != "temporal":
        return [None] * len(records)
    found = [re.match(method.recording_pattern, r.source_name) for r in records]
    return [None if m is None else int(m.group("frame")) for m in found]


def _near(item: ErrorItem, picks: Sequence[ErrorItem], gap: int) -> bool:
    return item.frame is not None and any(
        p.source == item.source and p.frame is not None and abs(p.frame - item.frame) < gap
        for p in picks
    )


def _spread(items: Sequence[ErrorItem]) -> list[ErrorItem]:
    """Order items so each source gives one before any gives a second."""
    by_source: dict[str, list[ErrorItem]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)
    queues = list(by_source.values())
    spread = []
    for depth in range(max((len(q) for q in queues), default=0)):
        spread += [q[depth] for q in queues if depth < len(q)]
    return spread


def gallery_picks(
    items: Sequence[ErrorItem], count: int, exclude: Sequence[str], min_frame_gap: int = 0
) -> list[ErrorItem]:
    """Pick ``count`` images for the gallery, one error each.

    Picks alternate between the most confident false positives and the
    images with the most misses, spread over sources, with one tile per
    image. Frames of one recording closer than ``min_frame_gap`` look alike,
    so the second is skipped. If one kind runs out, the other fills the
    rest. Excluded images are skipped, so the next one in order takes their
    place.
    """
    shown = [i for i in items if i.file_name not in set(exclude)]
    misses_per_image = Counter(i.file_name for i in shown if i.kind == "miss")
    firsts: dict[tuple[str, str], ErrorItem] = {}
    for i in shown:
        # One error stands for its image: the most confident false positive,
        # or the first miss.
        key = (i.file_name, i.kind)
        if key not in firsts or i.score > firsts[key].score:
            firsts[key] = i
    ranked = {
        "false positive": _spread(
            sorted(
                (i for i in firsts.values() if i.kind == "false positive"),
                key=lambda i: (-i.score, i.file_name),
            )
        ),
        "miss": _spread(
            sorted(
                (i for i in firsts.values() if i.kind == "miss"),
                key=lambda i: (-misses_per_image[i.file_name], i.file_name),
            )
        ),
    }
    picks: list[ErrorItem] = []
    used: set[str] = set()
    sources: Counter[str] = Counter()
    kinds = [k for k in ERROR_KINDS if ranked[k]]
    turn = 0
    while len(picks) < count and kinds:
        kind = kinds[turn % len(kinds)]
        turn += 1
        left = [
            i
            for i in ranked[kind]
            if i.file_name not in used and not _near(i, picks, min_frame_gap)
        ]
        if not left:
            kinds.remove(kind)
            continue
        # The least shown source goes first, across both kinds, so two
        # frames of one recording rarely sit side by side.
        item = min(left, key=lambda i: sources[i.source])
        picks.append(item)
        used.add(item.file_name)
        sources[item.source] += 1
    return picks


def _annotator(color: sv.Color, thickness: int) -> sv.BoxAnnotator:
    return sv.BoxAnnotator(color=color, thickness=thickness)


def _detections(boxes: npt.NDArray[np.float64]) -> sv.Detections:
    return sv.Detections(xyxy=boxes.reshape(-1, 4), class_id=np.zeros(len(boxes), dtype=int))


def draw_errors(
    image: Image.Image,
    labels: npt.NDArray[np.float64],
    false_positives: npt.NDArray[np.float64],
    misses: npt.NDArray[np.float64],
    side: int,
    thickness: int = 2,
) -> Image.Image:
    """Draw labeled boxes, false positives, and misses on ``image`` scaled to fit ``side``."""
    scale = side / max(image.size)
    shown = image.convert("RGB").resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    )
    for boxes, color, width in (
        (labels, LABEL_COLOR, 1),
        (misses, MISS_COLOR, thickness),
        (false_positives, FALSE_POSITIVE_COLOR, thickness),
    ):
        if len(boxes):
            shown = _annotator(color, width).annotate(shown, _detections(boxes * scale))
    return shown


def _image_boxes(
    item: ErrorItem, items: Sequence[ErrorItem], labels: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    same = [i for i in items if i.file_name == item.file_name]
    fps = np.array([i.xyxy for i in same if i.kind == "false positive"]).reshape(-1, 4)
    misses = np.array([i.xyxy for i in same if i.kind == "miss"]).reshape(-1, 4)
    return fps, misses


def gallery_image(
    picks: Sequence[tuple[ErrorItem, Path, npt.NDArray[np.float64]]],
    items: Sequence[ErrorItem],
    cfg: GalleryConfig,
) -> Image.Image:
    """Tile each pick's whole image with all of its errors drawn.

    Args:
        picks: Each pick with its image path and its labeled boxes.
        items: Every error in the diagnosed splits.
        cfg: Gallery layout.
    """
    tiles, captions = [], []
    for item, path, labels in picks:
        fps, misses = _image_boxes(item, items, labels)
        with Image.open(path) as image:
            tiles.append(draw_errors(image, labels, fps, misses, cfg.tile_px))
        captions.append(f"{item.dataset}: {item.kind}")
    return tile(tiles, captions, cfg)


def review_sample(
    items: Sequence[ErrorItem], per_kind: int, seed: str
) -> dict[tuple[str, str], list[ErrorItem]]:
    """Draw a seeded sample of up to ``per_kind`` errors of each kind in each dataset."""
    groups: dict[tuple[str, str], list[ErrorItem]] = {}
    for i in items:
        groups.setdefault((i.dataset, i.kind), []).append(i)
    sample = {}
    for (dataset, kind), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda i: (i.file_name, i.xyxy))
        rng = random.Random(f"{seed}:{dataset}:{kind}")
        chosen = ordered if len(ordered) <= per_kind else rng.sample(ordered, per_kind)
        sample[(dataset, kind)] = sorted(chosen, key=lambda i: (i.file_name, i.xyxy))
    return sample


def crop_box(
    image: Image.Image, xyxy: Sequence[float], context: float, min_px: int
) -> tuple[Image.Image, tuple[float, float]]:
    """Crop a square around a box with ``context`` box widths of margin.

    Returns:
        The crop and its top left corner in the image, to move boxes into it.
    """
    x1, y1, x2, y2 = xyxy
    side = max(context * max(x2 - x1, y2 - y1), min_px)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - side / 2, 0), max(image.width - side, 0))
    top = min(max(cy - side / 2, 0), max(image.height - side, 0))
    box = (round(left), round(top), round(left + side), round(top + side))
    return image.crop(box), (box[0], box[1])


def review_sheet(
    sample: Sequence[ErrorItem],
    paths: dict[str, Path],
    labels: dict[str, npt.NDArray[np.float64]],
    cfg: ReviewConfig,
) -> Image.Image:
    """Tile a numbered crop around each sampled error, numbered from 1 row by row."""
    tiles, captions = [], []
    for n, item in enumerate(sample, start=1):
        with Image.open(paths[item.file_name]) as image:
            crop, (left, top) = crop_box(
                image.convert("RGB"), item.xyxy, cfg.context, cfg.min_crop_px
            )
        offset = np.array([left, top, left, top], dtype=np.float64)
        box = np.array([item.xyxy], dtype=np.float64) - offset
        fps, misses = (
            (box, np.zeros((0, 4))) if item.kind == "false positive" else (np.zeros((0, 4)), box)
        )
        tiles.append(draw_errors(crop, labels[item.file_name] - offset, fps, misses, cfg.tile_px))
        captions.append(f"{n} {item.score:.2f}" if item.kind == "false positive" else str(n))
    return tile(tiles, captions, cfg)


def read_review(
    path: Path, verdicts: Sequence[str]
) -> dict[tuple[str, str, str, tuple[float, float, float, float]], str]:
    """Read the verdicts judged by eye, keyed like :attr:`ErrorItem.key`.

    Raises:
        ConfigError: If a row has a verdict outside ``verdicts`` or appears twice.
    """
    found = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["dataset"],
                row["file_name"],
                row["kind"],
                (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
            )
            if row["verdict"] not in verdicts:
                raise ConfigError(f"{path}: {row['verdict']!r} is not one of {list(verdicts)}")
            if key in found:
                raise ConfigError(f"{path}: {key} is reviewed twice")
            found[key] = row["verdict"]
    return found


def review_counts(
    sample: dict[tuple[str, str], list[ErrorItem]],
    totals: Counter[tuple[str, str]],
    verdicts: dict[tuple[str, str, str, tuple[float, float, float, float]], str],
    allowed: Sequence[str],
) -> list[ReviewCount]:
    """Count verdicts for each sampled group.

    Raises:
        ConfigError: If a sampled error has no verdict, or a verdict matches
            no sampled error, which means the sample or the data changed
            since the review.
    """
    wanted = {i.key for group in sample.values() for i in group}
    if stale := sorted(set(verdicts) - wanted):
        raise ConfigError(f"{len(stale)} reviewed errors are not in the sample, such as {stale[0]}")
    counts = []
    for (dataset, kind), group in sample.items():
        if missing := [i for i in group if i.key not in verdicts]:
            raise ConfigError(
                f"{len(missing)} sampled {kind}s in {dataset} have no verdict, "
                f"such as {missing[0].file_name} {missing[0].xyxy}"
            )
        tally = Counter(verdicts[i.key] for i in group)
        by_kind: dict[str, Counter[str]] = {}
        for i in group:
            if i.automatic_kind:
                by_kind.setdefault(i.automatic_kind, Counter())[verdicts[i.key]] += 1
        counts.append(
            ReviewCount(
                dataset=dataset,
                kind=kind,
                total=totals[(dataset, kind)],
                sampled=len(group),
                verdicts={v: tally[v] for v in allowed if tally[v]},
                by_automatic_kind={
                    k: {v: c[v] for v in allowed if c[v]}
                    for k in FALSE_POSITIVE_KINDS
                    if (c := by_kind.get(k))
                },
            )
        )
    return counts


def write_review_template(path: Path, sample: dict[tuple[str, str], list[ErrorItem]]) -> None:
    """Write a review file with an empty verdict for every sampled error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(REVIEW_FIELDS)
        for group in sample.values():
            for i in group:
                writer.writerow([i.dataset, i.file_name, i.kind, *i.xyxy, "", ""])


@dataclass(frozen=True)
class DiagnosisRun:
    """A diagnosis and what drawing its images needs."""

    diagnosis: Diagnosis
    cases: list[SplitCases]
    errors: list[list[ClassErrors]]
    items: list[ErrorItem]
    picks: list[ErrorItem]
    review_sample: dict[tuple[str, str], list[ErrorItem]]


def diagnose_model(project: ProjectConfig, model: str) -> DiagnosisRun:
    """Diagnose every published run of ``model`` and compare its training data with each split.

    Verdicts are counted when the review file exists. Without it, the review
    is left empty so the sample can be drawn and judged first.

    Raises:
        ConfigError: If the model is unknown or has no published run, or the
            review file does not match the sample.
    """
    cfg = project.diagnose
    ev = project.evaluate
    entries = load_yaml(ev.models_file, ModelsFile).models
    if model not in entries:
        raise ConfigError(f"{model} is not in {ev.models_file}")
    runs = [r.result for r in select_runs(load_runs(ev.runs_dir)) if r.result.model == model]
    if not runs:
        raise ConfigError(f"{model} has no published run under {ev.runs_dir}")
    thresholds = {(r.metrics.confidence, r.metrics.iou) for r in runs}
    if len(thresholds) != 1:
        raise ConfigError(f"{model}'s runs use different thresholds {sorted(thresholds)}")
    ((confidence, iou),) = thresholds

    all_cases = [load_cases(project, r) for r in runs]
    features = []
    for c in all_cases:
        logger.info(
            "measuring %d images of %s %s", len(c.records), c.result.dataset, c.result.split
        )
        features.append(split_features(c.split_dir, c.records, cfg.feature_px))
    pooled = [f for fs in features for f in fs]
    brightness_edges = quantile_edges([b for b, _ in pooled], cfg.feature_bins)
    sharpness_edges = quantile_edges([s for _, s in pooled], cfg.feature_bins)

    splits, all_errors, items = [], [], []
    for c, fs in zip(all_cases, features, strict=True):
        errors = class_errors(c, cfg.localization_floor)
        sources = image_sources(project, c.result.dataset, c.records)
        splits.append(
            diagnose_split(
                c,
                errors,
                fs,
                sources,
                brightness_edges,
                sharpness_edges,
                cfg,
                project.inspect.area_buckets,
            )
        )
        all_errors.append(errors)
        frames = image_frames(project, c.result.dataset, c.records)
        items += error_items(c, errors, None if sources is None else sources[0], frames)

    picks = [
        pick
        for dataset, count in cfg.gallery.per_dataset.items()
        for pick in gallery_picks(
            [
                i
                for i in items
                if i.dataset == dataset
                and i.source not in cfg.gallery.exclude_sources.get(dataset, [])
            ],
            count,
            cfg.gallery.exclude.get(dataset, []),
            cfg.gallery.min_frame_gap,
        )
    ]
    for dataset, count in cfg.gallery.per_dataset.items():
        shown = sum(p.dataset == dataset for p in picks)
        if shown < count:
            logger.warning(
                "the gallery shows %d of %d %s tiles, since exclusions and the frame gap "
                "leave no more errors to pick",
                shown,
                count,
                dataset,
            )
    sample = review_sample(items, cfg.review.per_kind, str(project.seed))
    review: list[ReviewCount] = []
    if cfg.review.file.is_file():
        totals: Counter[tuple[str, str]] = Counter((i.dataset, i.kind) for i in items)
        verdicts = read_review(cfg.review.file, cfg.review.verdicts)
        review = review_counts(sample, totals, verdicts, cfg.review.verdicts)
    else:
        logger.warning("%s does not exist, so no verdicts are counted", cfg.review.file)

    trained_on = entries[model].dataset
    train_dir = project.paths.harmonized_dir / trained_on / "train"
    train_records = sorted(load_split(train_dir, trained_on), key=lambda r: r.ref.file_name)
    logger.info("measuring %d images of %s train", len(train_records), trained_on)
    scored = sorted({c for r in runs for c in r.scored_classes})
    domain = [
        feature_summary(
            trained_on,
            "train",
            "training",
            train_records,
            split_features(train_dir, train_records, cfg.feature_px),
            scored,
            cfg.quantiles,
        ),
        *(
            feature_summary(
                c.result.dataset,
                c.result.split,
                "test",
                c.records,
                fs,
                c.result.scored_classes,
                cfg.quantiles,
            )
            for c, fs in zip(all_cases, features, strict=True)
        ),
    ]
    diagnosis = Diagnosis(
        model=model,
        confidence=confidence,
        iou=iou,
        feature_px=cfg.feature_px,
        localization_floor=cfg.localization_floor,
        area_buckets=project.inspect.area_buckets,
        quantiles=cfg.quantiles,
        brightness_edges=brightness_edges,
        sharpness_edges=sharpness_edges,
        splits=splits,
        domain=domain,
        gallery=[
            GalleryTile(dataset=i.dataset, file_name=i.file_name, kind=i.kind, source=i.source)
            for i in picks
        ],
        review=review,
    )
    return DiagnosisRun(diagnosis, all_cases, all_errors, items, picks, sample)


def _image_labels(run: DiagnosisRun) -> dict[tuple[str, str], tuple[Path, npt.NDArray[np.float64]]]:
    """Map each image to its path and its labeled boxes of the scored classes."""
    found = {}
    for cases, errors in zip(run.cases, run.errors, strict=True):
        for n, r in enumerate(cases.records):
            boxes = [_xyxy(ce.labels[n]) for ce in errors]
            found[(r.ref.dataset, r.ref.file_name)] = (
                cases.split_dir / r.ref.file_name,
                np.concatenate(boxes).reshape(-1, 4),
            )
    return found


def write_gallery(run: DiagnosisRun, cfg: GalleryConfig, path: Path) -> None:
    """Write the gallery PNG and log the file behind each tile for the face check.

    Positions read ``r<row>c<col>``, counted from 1 at the top left.
    """
    images = _image_labels(run)
    picks = [(i, *images[(i.dataset, i.file_name)]) for i in run.picks]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(gallery_image(picks, run.items, cfg), cfg.max_bytes))
    for n, i in enumerate(run.picks):
        row, col = divmod(n, cfg.cols)
        logger.info("gallery r%dc%d %s/%s %s", row + 1, col + 1, i.dataset, i.file_name, i.kind)
    logger.info("wrote %s, which must pass the face check before it is committed", path)


def write_review_sheets(run: DiagnosisRun, cfg: ReviewConfig, out_dir: Path) -> None:
    """Write one numbered crop sheet per sampled group, and a review file to fill in.

    The sheets are not face-checked, so they go under the gitignored
    contact sheet directory.
    """
    images = _image_labels(run)
    per_sheet = cfg.rows * cfg.cols
    out_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, kind), group in run.review_sample.items():
        paths = {i.file_name: images[(dataset, i.file_name)][0] for i in group}
        labels = {i.file_name: images[(dataset, i.file_name)][1] for i in group}
        for start in range(0, len(group), per_sheet):
            sheet = review_sheet(group[start : start + per_sheet], paths, labels, cfg)
            stem = f"{REVIEW_SHEET_PREFIX}{dataset}_{kind.replace(' ', '_')}"
            name = f"{stem}_{start // per_sheet + 1}.jpg"
            sheet.save(out_dir / name)
            logger.info(
                "wrote %s (%s, %s, numbered from %d)", out_dir / name, dataset, kind, start + 1
            )
    template = out_dir / f"{REVIEW_SHEET_PREFIX}template.csv"
    write_review_template(template, run.review_sample)
    logger.info("fill in %s by eye and save it as %s", template, cfg.file)


def diagnosis_meta(
    repo: Path,
    project: ProjectConfig,
    project_config: Path,
    diagnosis: Diagnosis,
    argv: Sequence[str],
    dirty: bool,
) -> dict[str, object]:
    """Return the inputs, hashes, and versions behind one diagnosis."""
    ev = project.evaluate
    annotations = {
        f"{s.dataset}/{s.split}": sha256_file(
            project.paths.harmonized_dir / s.dataset / s.split / ANNOTATIONS_NAME
        )
        for s in diagnosis.splits
    }
    for d in diagnosis.domain:
        path = project.paths.harmonized_dir / d.dataset / d.split / ANNOTATIONS_NAME
        annotations[f"{d.dataset}/{d.split}"] = sha256_file(path)
    configs = [
        project_config,
        ev.models_file,
        ev.coverage_file,
        project.paths.reports_dir / SPLITS_NAME,
        project.diagnose.review.file,
    ]
    return {
        "created": utc_timestamp(),
        "command": ["frc-diagnose", *argv],
        "git": {"commit": git_commit(repo), "tree": git_tree(repo), "dirty": dirty},
        "configs": {str(p): sha256_file(p) if p.is_file() else None for p in configs},
        "annotations_sha256": annotations,
        "runs": {s.run_id: s.predictions_from for s in diagnosis.splits},
        "packages": package_versions(),
    }


def write_diagnosis(out_dir: Path, diagnosis: Diagnosis, meta: dict[str, object]) -> None:
    """Write ``diagnosis.json`` and ``meta.json``, rendering both before writing either."""
    files = {
        DIAGNOSIS_NAME: json.dumps(diagnosis.model_dump(mode="json"), indent=2) + "\n",
        META_NAME: json.dumps(meta, indent=2) + "\n",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    logger.info("wrote %s", ", ".join(str(out_dir / n) for n in files))


def load_diagnosis(path: Path) -> Diagnosis:
    """Read a ``diagnosis.json`` written by :func:`write_diagnosis`."""
    return Diagnosis.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Run the diagnosis command line and return the exit code."""
    parser = argparse.ArgumentParser(
        description="Slice a model's errors from its published runs into reports/diagnosis/."
    )
    parser.add_argument("model", help="run name in reports/models.yaml, such as baseline-a")
    parser.add_argument("--allow-dirty", action="store_true", help="run from a dirty tree")
    parser.add_argument(
        "--review-sheet",
        action="store_true",
        help="write crop sheets of the review sample under data/contact_sheets/ and nothing else",
    )
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    add_log_level_argument(parser)
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)
    setup_logging(args.log_level)

    repo = Path.cwd()
    dirty = git_is_dirty(repo)
    if dirty and not args.allow_dirty:
        raise DirtyTreeError("commit or stash changes first, or pass --allow-dirty")
    project = load_yaml(args.project_config, ProjectConfig)
    run = diagnose_model(project, args.model)
    if args.review_sheet:
        write_review_sheets(run, project.diagnose.review, project.paths.contact_sheet_dir)
        return 0
    meta = diagnosis_meta(repo, project, args.project_config, run.diagnosis, raw_args, dirty)
    write_diagnosis(project.diagnose.output_dir / args.model, run.diagnosis, meta)
    write_gallery(run, project.diagnose.gallery, project.paths.assets_dir / GALLERY_NAME)
    return 0
