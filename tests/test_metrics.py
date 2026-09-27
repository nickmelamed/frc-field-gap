import json
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from frc_xdata.errors import ConfigError
from frc_xdata.evaluate import compute_metrics, scored_classes

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["fuel", "robot"]
FUEL, ROBOT = 0, 1
THRESHOLDS = [0.1, 0.5, 0.85, 0.95]
A = [0, 0, 10, 10]
B = [20, 20, 30, 30]
C = [40, 40, 50, 50]
FAR = [50, 0, 60, 10]


def labels(*boxes: list[int], cls: int = FUEL) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float), class_id=np.full(len(boxes), cls, dtype=int)
    )


def preds(*items: tuple[list[int], float, int]) -> sv.Detections:
    if not items:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([box for box, _, _ in items], dtype=float),
        confidence=np.array([conf for _, conf, _ in items]),
        class_id=np.array([cls for _, _, cls in items], dtype=int),
    )


def score(
    predictions: list[sv.Detections], targets: list[sv.Detections], scored: list[str] | None = None
):
    return compute_metrics(
        predictions,
        targets,
        CLASSES,
        scored or ["fuel"],
        confidence=0.5,
        iou=0.5,
        thresholds=THRESHOLDS,
    )


def test_perfect_predictions_score_one() -> None:
    targets = [labels(A, B), labels(C)]
    m = score([preds((A, 0.9, FUEL), (B, 0.8, FUEL)), preds((C, 0.95, FUEL))], targets)
    fuel = m.classes[0]
    assert (m.map50, m.map50_95) == (1.0, 1.0)
    assert (fuel.precision, fuel.recall) == (1.0, 1.0)
    assert (fuel.true_positives, fuel.false_positives, fuel.false_negatives) == (3, 0, 0)
    assert fuel.instances == 3
    # Every box is small, so there is no medium or large score to report.
    assert m.small_map50_95 == 1.0
    assert (m.medium_map50_95, m.large_map50_95) == (None, None)


def test_no_predictions_give_zero_recall() -> None:
    m = score([sv.Detections.empty(), sv.Detections.empty()], [labels(A, B), labels(C)])
    fuel = m.classes[0]
    assert fuel.recall == 0.0
    # Precision is undefined with no predictions, not zero.
    assert fuel.precision is None
    assert m.map50 == 0.0
    assert (fuel.true_positives, fuel.false_negatives) == (0, 3)
    assert m.confusion.matrix == [[0, 3], [0, 0]]


def mixed() -> tuple[list[sv.Detections], list[sv.Detections]]:
    """One hit (A), two misses (B, C), and two false positives (FAR, and one on a background image).

    C also has a low-confidence hit at 0.2, below the 0.5 threshold.
    """
    targets = [labels(A, B), labels(C), labels()]
    predictions = [
        preds((A, 0.9, FUEL), (FAR, 0.8, FUEL)),
        preds((C, 0.2, FUEL)),
        preds((B, 0.7, FUEL)),
    ]
    return predictions, targets


def test_known_mix_of_hits_misses_and_false_positives() -> None:
    m = score(*mixed())
    fuel = m.classes[0]
    assert (fuel.true_positives, fuel.false_positives, fuel.false_negatives) == (1, 2, 2)
    assert (fuel.precision, fuel.recall) == (0.333, 0.333)
    assert m.confusion.labels == ["fuel", "background"]
    assert m.confusion.matrix == [[1, 2], [2, 0]]


def test_pr_curve_sweeps_thresholds() -> None:
    curve = score(*mixed()).pr_curve["fuel"]
    assert [(p.threshold, p.precision, p.recall) for p in curve] == [
        (0.1, 0.5, 0.667),
        (0.5, 0.333, 0.333),
        (0.85, 1.0, 0.333),
        (0.95, None, 0.0),
    ]


def test_fuel_only_dataset_is_not_scored_on_robot() -> None:
    targets = [labels(A)]
    predictions = [preds((A, 0.9, FUEL), (C, 0.9, ROBOT))]
    m = score(predictions, targets, scored=["fuel"])
    assert [c.name for c in m.classes] == ["fuel"]
    assert m.classes[0].false_positives == 0
    assert m.classes[0].precision == 1.0
    assert "robot" not in m.pr_curve


def test_two_class_dataset_scores_both() -> None:
    targets = [labels(A), labels(C, cls=ROBOT)]
    predictions = [preds((A, 0.9, FUEL)), preds((C, 0.9, ROBOT))]
    m = score(predictions, targets, scored=["fuel", "robot"])
    assert [(c.name, c.recall) for c in m.classes] == [("fuel", 1.0), ("robot", 1.0)]
    assert m.confusion.matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_wrong_class_counts_as_a_confusion() -> None:
    m = score([preds((A, 0.9, ROBOT))], [labels(A)], scored=["fuel", "robot"])
    fuel, robot = m.classes
    assert m.confusion.matrix[FUEL][ROBOT] == 1
    assert (fuel.false_negatives, robot.false_positives) == (1, 1)


def test_unknown_scored_class_raises() -> None:
    with pytest.raises(ConfigError, match="goal"):
        score([sv.Detections.empty()], [labels(A)], scored=["goal"])


def test_mismatched_image_counts_raise() -> None:
    with pytest.raises(ConfigError, match="prediction sets"):
        score([sv.Detections.empty()], [labels(A), labels(B)])


def test_scored_classes_reads_coverage(tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"alpha": {"labeled": ["fuel"]}}), encoding="utf-8")
    assert scored_classes(path, "alpha") == ["fuel"]
    with pytest.raises(ConfigError, match="beta"):
        scored_classes(path, "beta")


def test_fuel_only_datasets_in_the_real_coverage_report() -> None:
    coverage = REPO_ROOT / "reports" / "class_coverage.json"
    assert scored_classes(coverage, "marswars") == ["fuel"]
    assert scored_classes(coverage, "scorekeeper") == ["fuel", "robot"]


def test_a_scored_class_with_no_labeled_boxes_has_no_score() -> None:
    m = score([preds((A, 0.9, FUEL), (C, 0.9, ROBOT))], [labels(A)], scored=["fuel", "robot"])
    fuel, robot = m.classes
    assert (fuel.map50, fuel.recall) == (1.0, 1.0)
    assert robot.instances == 0
    assert (robot.map50, robot.map50_95, robot.recall) == (None, None, None)
    # Its one prediction is still a false positive, so precision is defined.
    assert (robot.precision, robot.false_positives) == (0.0, 1)
