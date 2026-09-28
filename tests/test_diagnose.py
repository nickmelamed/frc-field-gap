import numpy as np
import supervision as sv

from frc_xdata.diagnose import false_positive_kind, match_boxes

FUEL, ROBOT = 0, 1
NO_BOXES = np.zeros((0, 4))


def dets(
    boxes: list[tuple[float, float, float, float]],
    classes: list[int] | None = None,
    conf: list[float] | None = None,
) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        class_id=np.array(classes or [FUEL] * len(boxes)),
        confidence=None if conf is None else np.array(conf, dtype=float),
    )


def test_perfect_predictions_are_all_hits() -> None:
    boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
    m = match_boxes(dets(boxes), dets(boxes), iou=0.5)
    assert sorted(m.hits) == [(0, 0), (1, 1)]
    assert m.false_positives == m.misses == m.confusions == ()


def test_no_predictions_miss_every_label() -> None:
    m = match_boxes(dets([]), dets([(0, 0, 10, 10)]), iou=0.5)
    assert m.misses == (0,)
    assert m.hits == m.false_positives == ()


def test_no_labels_make_every_prediction_false() -> None:
    m = match_boxes(dets([(0, 0, 10, 10), (5, 5, 9, 9)]), dets([]), iou=0.5)
    assert m.false_positives == (0, 1)


def test_iou_of_exactly_the_threshold_is_not_a_match() -> None:
    # The prediction covers half the label and nothing else, so IoU is 0.5.
    m = match_boxes(dets([(0, 0, 10, 5)]), dets([(0, 0, 10, 10)]), iou=0.5)
    assert m.hits == ()
    assert m.false_positives == (0,)
    assert m.misses == (0,)


def test_the_better_overlap_wins_and_the_other_is_a_false_positive() -> None:
    label = [(0, 0, 10, 10)]
    preds = [(0, 0, 10, 8), (0, 0, 10, 10)]
    m = match_boxes(dets(preds), dets(label), iou=0.5)
    assert m.hits == ((1, 0),)
    assert m.false_positives == (0,)


def test_a_class_mismatch_is_a_confusion() -> None:
    m = match_boxes(dets([(0, 0, 10, 10)], [ROBOT]), dets([(0, 0, 10, 10)], [FUEL]), iou=0.5)
    assert m.confusions == ((0, 0),)
    assert m.hits == m.false_positives == m.misses == ()


def test_a_same_class_pair_is_taken_before_a_better_mismatched_one() -> None:
    labels = dets([(0, 0, 10, 10)], [FUEL])
    preds = dets([(0, 0, 10, 10), (0, 0, 10, 8)], [ROBOT, FUEL])
    m = match_boxes(preds, labels, iou=0.5)
    assert m.hits == ((1, 0),)
    assert m.false_positives == (0,)


def test_totals_agree_with_the_supervision_confusion_matrix() -> None:
    labels = dets(
        [(0, 0, 10, 10), (12, 0, 22, 10), (40, 40, 60, 60), (100, 100, 110, 110)],
        [FUEL, FUEL, ROBOT, FUEL],
    )
    preds = dets(
        [
            (1, 0, 11, 10),
            (0, 0, 10, 9),
            (13, 1, 22, 10),
            (41, 40, 60, 61),
            (70, 70, 80, 80),
            (100, 100, 104, 104),
        ],
        [FUEL, FUEL, FUEL, FUEL, FUEL, FUEL],
        [0.9, 0.8, 0.7, 0.6, 0.55, 0.52],
    )
    m = match_boxes(preds, labels, iou=0.5)
    matrix = sv.ConfusionMatrix.from_detections(
        predictions=[preds],
        targets=[labels],
        classes=["fuel", "robot"],
        conf_threshold=0.5,
        iou_threshold=0.5,
    ).matrix.astype(int)
    assert len(m.hits) == int(np.trace(matrix[:2, :2]))
    assert len(m.confusions) == int(matrix[:2, :2].sum() - np.trace(matrix[:2, :2]))
    assert len(m.false_positives) == int(matrix[2, :2].sum())
    assert len(m.misses) == int(matrix[:2, 2].sum())


def test_a_second_box_on_a_taken_label_is_a_duplicate() -> None:
    labels = dets([(0, 0, 10, 10)])
    kind = false_positive_kind(np.array([0, 0, 10, 9.0]), labels, [0], NO_BOXES, 0.5, 0.1)
    assert kind == "duplicate"


def test_a_loose_box_on_a_label_is_a_localization_error() -> None:
    labels = dets([(0, 0, 10, 10)])
    kind = false_positive_kind(np.array([0, 0, 10, 30.0]), labels, [], NO_BOXES, 0.5, 0.1)
    assert kind == "localization"


def test_a_box_centered_on_a_robot_is_kept_apart() -> None:
    robots = np.array([[50, 50, 150, 150.0]])
    kind = false_positive_kind(np.array([90, 90, 100, 100.0]), dets([]), [], robots, 0.5, 0.1)
    assert kind == "inside_unscored"


def test_a_box_touching_nothing_is_background() -> None:
    labels = dets([(0, 0, 10, 10)])
    robots = np.array([[50, 50, 150, 150.0]])
    box = np.array([200, 200, 210, 210.0])
    assert false_positive_kind(box, labels, [0], robots, 0.5, 0.1) == "background"
