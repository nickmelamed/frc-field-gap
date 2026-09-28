import json
from itertools import count
from pathlib import Path

import numpy as np
import pytest
import supervision as sv
from conftest import smooth_image, write_split
from PIL import Image, ImageFilter
from pydantic import ValidationError
from test_evaluate_cli import ARGS, fake, workspace

from frc_xdata import diagnose, evaluate
from frc_xdata.config import AreaBuckets, DiagnoseConfig, SourcePattern
from frc_xdata.diagnose import (
    ImageErrors,
    SliceScorer,
    bin_index,
    bin_names,
    count_names,
    false_positive_kind,
    image_errors,
    image_features,
    load_diagnosis,
    match_boxes,
    quantile_edges,
    quantiles_of,
    relative_side,
    score_slice,
    size_rows,
    slicing,
    source_of,
)
from frc_xdata.errors import ConfigError, CountMismatchError, DirtyTreeError
from frc_xdata.evaluate import compute_metrics
from frc_xdata.harmonize import to_coco
from frc_xdata.inspect_datasets import Box, ImageRecord, ImageRef

# The evaluate CLI fixtures build a harmonized split and a model to run.
__all__ = ["fake", "workspace"]

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


def gray(value: int, side: int = 32) -> Image.Image:
    return Image.new("L", (side, side), value)


def test_a_flat_image_has_no_edges_and_its_own_brightness() -> None:
    assert image_features(gray(90), 16) == (90.0, 0.0)


def test_a_checkerboard_is_sharper_than_its_blurred_copy() -> None:
    board = Image.fromarray((np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8))
    blurred = board.filter(ImageFilter.GaussianBlur(2))
    assert image_features(board, 32)[1] > image_features(blurred, 32)[1]


def test_features_are_measured_at_the_model_size() -> None:
    # Stretching a wide image to a square does not change its mean gray level.
    wide = Image.new("L", (200, 50), 40)
    assert image_features(wide, 16) == (40.0, 0.0)


def test_quantile_edges_cut_equal_counts() -> None:
    assert quantile_edges([1, 2, 3, 4, 5, 6], 3) == [2.7, 4.3]
    assert quantile_edges([1, 2, 3, 4, 5, 6], 3) == quantile_edges([6, 5, 4, 3, 2, 1], 3)


def test_a_value_on_an_edge_goes_to_the_higher_bin() -> None:
    assert [bin_index(v, [10.0, 20.0]) for v in (9.9, 10.0, 19.9, 20.0, 99)] == [0, 1, 1, 2, 2]


def test_bin_names_follow_the_edges() -> None:
    assert bin_names([10.0, 20.0]) == ["below 10.0", "10.0 to 20.0", "20.0 and above"]


def test_count_names_join_ranges_of_whole_counts() -> None:
    assert count_names([0, 1, 2, 5, 10]) == ["0", "1", "2 to 4", "5 to 9", "10 or more"]


def test_the_first_matching_source_wins_and_the_rest_are_other() -> None:
    patterns = [
        SourcePattern(name="fuel", pattern=r"^frame_"),
        SourcePattern(name="broadcast", pattern=r"Match"),
    ]
    assert source_of("frame_Match_1.jpg", patterns) == "fuel"
    assert source_of("Match-2.jpg", patterns) == "broadcast"
    assert source_of("IMG_1.jpg", patterns) == "other"


def test_crowding_edges_must_start_at_zero_and_increase() -> None:
    base = {
        "output_dir": "reports/diagnosis",
        "feature_px": 384,
        "localization_floor": 0.1,
        "feature_bins": 3,
        "quantiles": [0.5],
    }
    DiagnoseConfig(**base, crowding_edges=[0, 1, 5])
    for bad in ([1, 2], [0, 2, 2], []):
        with pytest.raises(ValidationError):
            DiagnoseConfig(**base, crowding_edges=bad)


# Image 0 has two labels, one found and one missed, plus a loose false
# positive. Image 1 has one label, found. Image 2 has none, and a false
# positive inside a robot.
LABELS = [dets([(0, 0, 10, 10), (50, 50, 60, 60)]), dets([(0, 0, 10, 10)]), dets([])]
PREDS = [
    dets([(0, 0, 10, 10), (0, 0, 10, 30)], conf=[0.9, 0.8]),
    dets([(0, 0, 10, 10), (70, 70, 80, 80)], conf=[0.9, 0.3]),
    dets([(20, 20, 30, 30)], conf=[0.7]),
]
ROBOTS = [NO_BOXES, NO_BOXES, np.array([[0, 0, 60, 60.0]])]


def errors() -> list[ImageErrors]:
    return [
        image_errors(p, lab, r, 0.5, 0.5, 0.1)
        for p, lab, r in zip(PREDS, LABELS, ROBOTS, strict=True)
    ]


def test_image_errors_count_only_predictions_at_the_threshold() -> None:
    e0, e1, e2 = errors()
    assert (e0.matched_labels, e0.missed_labels, e0.false_positive_kinds) == (
        (0,),
        (1,),
        ("localization",),
    )
    assert (e1.matched_labels, e1.false_positive_kinds) == ((0,), ())
    assert e2.false_positive_kinds == ("inside_unscored",)


def test_a_slice_of_every_image_matches_compute_metrics() -> None:
    s = score_slice("all", [0, 1, 2], PREDS, LABELS, errors(), ["fuel", "robot"], "fuel", 0.5, 0.5)
    (c,) = compute_metrics(PREDS, LABELS, ["fuel", "robot"], ["fuel"], 0.5, 0.5, [0.5]).classes
    assert (s.hits, s.false_positives, s.misses) == (2, 2, 1)
    assert (s.precision, s.recall, s.map50) == (c.precision, c.recall, c.map50)
    assert s.false_positives_per_image == 0.667
    assert s.false_positive_kinds == {
        "duplicate": 0,
        "localization": 1,
        "inside_unscored": 1,
        "background": 0,
    }


def test_slices_split_the_counts_without_losing_any() -> None:
    score = SliceScorer(PREDS, LABELS, errors(), ["fuel", "robot"], "fuel", 0.5, 0.5)
    result = slicing("has labeled fuel", ["yes", "yes", "no"], ["yes", "no"], score)
    yes, no = result.slices
    assert (yes.name, yes.images, yes.hits, yes.false_positives, yes.misses) == ("yes", 2, 2, 1, 1)
    assert (no.name, no.images, no.labeled, no.false_positives) == ("no", 1, 0, 1)
    assert no.recall is None


def test_an_empty_group_is_left_out() -> None:
    score = SliceScorer(PREDS, LABELS, errors(), ["fuel", "robot"], "fuel", 0.5, 0.5)
    result = slicing("crowding", ["1", "1", "0"], ["0", "1", "10 or more"], score)
    assert [s.name for s in result.slices] == ["0", "1"]


def test_matching_that_disagrees_with_the_confusion_matrix_fails() -> None:
    wrong = [*errors()[:2], image_errors(dets([]), dets([]), NO_BOXES, 0.5, 0.5, 0.1)]
    with pytest.raises(CountMismatchError):
        score_slice("all", [0, 1, 2], PREDS, LABELS, wrong, ["fuel", "robot"], "fuel", 0.5, 0.5)


def test_size_rows_use_the_share_of_the_image() -> None:
    buckets = AreaBuckets(small_max=0.0025, medium_max=0.0225)
    records = [
        ImageRecord(ImageRef("d", "test", f"{i}.jpg"), f"{i}.jpg", 100, 100, ()) for i in range(3)
    ]
    # Every 10x10 label covers 1% of a 100x100 image, which is medium. The
    # 10x30 false positive covers 3%, which is large, and the robot one 1%.
    rows = {r.bucket: r for r in size_rows(records, LABELS, errors(), buckets)}
    assert (rows["medium"].labeled, rows["medium"].hits, rows["medium"].recall) == (3, 2, 0.667)
    assert rows["small"].recall is None
    assert (rows["large"].false_positives, rows["medium"].false_positives) == (1, 1)


def test_relative_side_is_the_square_root_of_the_area_share() -> None:
    assert relative_side(Box("fuel", 0, 0, 10, 40), 100, 100) == 0.2


def test_quantiles_of_nothing_are_empty() -> None:
    assert quantiles_of([], [0.5]) == []
    assert quantiles_of([1.0, 2.0, 3.0], [0.5]) == [2.0]


@pytest.fixture
def diagnosed(workspace: Path, fake: object, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The evaluate workspace after one live run, with a train split to compare against."""
    # Runs made within one second would tie, and the report takes the newest.
    stamps = count()
    monkeypatch.setattr(evaluate, "utc_timestamp", lambda: f"2026-09-27T10:00:{next(stamps):02d}Z")
    assert evaluate.main(ARGS) == 0
    train_dir = workspace / "data" / "harmonized" / "alpha" / "train"
    records = [
        ImageRecord(
            ImageRef("alpha", "train", "t.jpg"), "t.jpg", 64, 64, (Box("fuel", 0, 0, 8, 8),)
        )
    ]
    write_split(train_dir, to_coco(records, ["fuel", "robot"]))
    smooth_image(9).save(train_dir / "t.jpg")
    for name in ("git_is_dirty", "git_commit", "git_tree"):
        monkeypatch.setattr(diagnose, name, getattr(evaluate, name))
    return workspace


def test_main_reproduces_the_run_and_records_where_it_came_from(diagnosed: Path) -> None:
    assert diagnose.main(["m", "--project-config", "configs/project.yaml"]) == 0
    out = diagnosed / "reports" / "diagnosis" / "m"
    result = load_diagnosis(out / "diagnosis.json")
    (split,) = result.splits
    (fuel,) = split.classes
    assert (split.run_id, split.predictions_from) == ("run1", "this run")
    assert (fuel.overall.hits, fuel.overall.false_positives, fuel.overall.misses) == (3, 0, 0)
    assert [d.role for d in result.domain] == ["training", "test"]
    assert result.domain[0].boxes == 1
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["runs"] == {"run1": "this run"}
    assert meta["git"] == {"commit": "abc123", "tree": "def456", "dirty": False}


def test_main_follows_a_rescore_to_the_cache_it_read(diagnosed: Path) -> None:
    assert evaluate.main([*ARGS, "--from-cache", "run1"]) == 0
    assert diagnose.main(["m", "--project-config", "configs/project.yaml"]) == 0
    result = load_diagnosis(diagnosed / "reports" / "diagnosis" / "m" / "diagnosis.json")
    assert (result.splits[0].run_id, result.splits[0].predictions_from) == ("run2", "run1")


def test_main_refuses_a_dirty_tree(diagnosed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnose, "git_is_dirty", lambda repo: True)
    with pytest.raises(DirtyTreeError):
        diagnose.main(["m", "--project-config", "configs/project.yaml"])


def test_a_model_without_a_published_run_is_an_error(diagnosed: Path) -> None:
    with pytest.raises(ConfigError, match="no published run"):
        diagnose.main(["m2", "--project-config", "configs/project.yaml"])
