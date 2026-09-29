import json
from collections import Counter
from collections.abc import Callable
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import supervision as sv
import yaml
from conftest import smooth_image, write_split
from PIL import Image, ImageFilter
from pydantic import ValidationError
from test_evaluate_cli import ARGS, REPO_ROOT, fake, workspace

from frc_xdata import diagnose, evaluate
from frc_xdata.config import (
    AreaBuckets,
    DiagnoseConfig,
    GalleryConfig,
    ProjectConfig,
    SourcePattern,
    load_yaml,
)
from frc_xdata.diagnose import (
    ErrorItem,
    ErrorKind,
    ImageErrors,
    SliceScorer,
    SplitCases,
    bin_index,
    bin_names,
    check_run_totals,
    count_names,
    crop_box,
    draw_errors,
    false_positive_kind,
    gallery_picks,
    image_errors,
    image_features,
    load_diagnosis,
    match_boxes,
    quantile_edges,
    quantiles_of,
    read_review,
    relative_side,
    review_counts,
    review_sample,
    score_slice,
    size_rows,
    slicing,
    source_of,
)
from frc_xdata.errors import ConfigError, CountMismatchError, DirtyTreeError
from frc_xdata.evaluate import RunResult, compute_metrics
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
    project = load_yaml(REPO_ROOT / "configs" / "project.yaml", ProjectConfig)
    base = project.diagnose.model_dump(exclude={"crowding_edges"})
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


def item(
    name: str,
    kind: ErrorKind = "false positive",
    score: float = 0.9,
    source: str = "s",
    frame: int | None = None,
    auto: str = "background",
    dataset: str = "d",
) -> ErrorItem:
    return ErrorItem(
        dataset=dataset,
        split="test",
        file_name=name,
        source=source,
        kind=kind,
        automatic_kind=auto if kind == "false positive" else "",
        score=score if kind == "false positive" else 0.0,
        xyxy=(1.0, 2.0, 3.0, 4.0),
        frame=frame,
    )


def test_gallery_picks_alternate_kinds_with_one_tile_per_image() -> None:
    items = [
        item("a", score=0.9, source="x"),
        item("a", "miss", source="x"),
        item("b", score=0.8, source="y"),
        item("c", "miss", source="z"),
        item("c", "miss", source="z"),
    ]
    picks = gallery_picks(items, 3, [])
    assert [(p.file_name, p.kind) for p in picks] == [
        ("a", "false positive"),
        ("c", "miss"),
        ("b", "false positive"),
    ]


def test_gallery_picks_spread_over_sources_before_repeating_one() -> None:
    items = [item("a1", score=0.99, source="a"), item("a2", score=0.98, source="a")]
    items.append(item("b1", score=0.5, source="b"))
    assert [p.file_name for p in gallery_picks(items, 2, [])] == ["a1", "b1"]


def test_gallery_picks_skip_excluded_images_and_close_frames() -> None:
    items = [
        item("f10", score=0.9, frame=10),
        item("f12", score=0.8, frame=12),
        item("f50", score=0.7, frame=50),
        item("bad", score=0.99, frame=100),
    ]
    picks = gallery_picks(items, 3, ["bad"], min_frame_gap=30)
    assert [p.file_name for p in picks] == ["f10", "f50"]


def test_gallery_fills_with_one_kind_when_the_other_runs_out() -> None:
    items = [item(n, score=s, source=n) for n, s in (("a", 0.9), ("b", 0.8), ("c", 0.7))]
    assert [p.file_name for p in gallery_picks(items, 5, [])] == ["a", "b", "c"]


def test_review_sample_is_seeded_capped_and_keeps_small_groups_whole() -> None:
    many = [item(f"i{n:02d}") for n in range(30)]
    few = [item("m1", "miss"), item("m2", "miss")]
    first = review_sample([*many, *few], 5, "seed")
    assert first == review_sample(list(reversed([*many, *few])), 5, "seed")
    assert len(first[("d", "false positive")]) == 5
    assert [i.file_name for i in first[("d", "miss")]] == ["m1", "m2"]


def write_review(path: Path, rows: list[list[object]]) -> Path:
    lines = ["dataset,file_name,kind,x1,y1,x2,y2,verdict,note"]
    lines += [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


VERDICTS = ["clear fuel", "unlabeled fuel", "other"]


def test_read_review_rejects_unknown_verdicts_and_repeats(tmp_path: Path) -> None:
    row = ["d", "a", "false positive", 1.0, 2.0, 3.0, 4.0, "unlabeled fuel", ""]
    assert read_review(write_review(tmp_path / "ok.csv", [row]), VERDICTS) == {
        item("a").key: "unlabeled fuel"
    }
    with pytest.raises(ConfigError, match="is not one of"):
        read_review(write_review(tmp_path / "bad.csv", [[*row[:7], "maybe", ""]]), VERDICTS)
    with pytest.raises(ConfigError, match="twice"):
        read_review(write_review(tmp_path / "twice.csv", [row, row]), VERDICTS)


def test_review_counts_tally_verdicts_by_automatic_kind() -> None:
    sample = {("d", "false positive"): [item("a"), item("b", auto="localization")]}
    verdicts = {item("a").key: "unlabeled fuel", item("b").key: "other"}
    totals: Counter[tuple[str, str]] = Counter({("d", "false positive"): 10})
    (count,) = review_counts(sample, totals, verdicts, VERDICTS)
    assert (count.total, count.sampled) == (10, 2)
    assert count.verdicts == {"unlabeled fuel": 1, "other": 1}
    assert count.by_automatic_kind == {
        "localization": {"other": 1},
        "background": {"unlabeled fuel": 1},
    }


def test_review_counts_need_a_verdict_for_every_sampled_error() -> None:
    sample = {("d", "false positive"): [item("a"), item("b")]}
    with pytest.raises(ConfigError, match="no verdict"):
        review_counts(sample, Counter(), {item("a").key: "other"}, VERDICTS)


def test_review_counts_reject_verdicts_outside_the_sample() -> None:
    sample = {("d", "false positive"): [item("a")]}
    verdicts = {item("a").key: "other", item("gone").key: "other"}
    with pytest.raises(ConfigError, match="not in the sample"):
        review_counts(sample, Counter(), verdicts, VERDICTS)


def test_crops_stay_inside_the_image_and_keep_their_minimum_size() -> None:
    image = gray(0, 200)
    crop, corner = crop_box(image, (190, 190, 198, 198), context=4, min_px=64)
    assert crop.size == (64, 64)
    assert corner == (136, 136)
    crop, corner = crop_box(image, (90, 90, 110, 110), context=4, min_px=64)
    assert (crop.size, corner) == ((80, 80), (60, 60))


def test_draw_errors_scales_the_image_to_fit() -> None:
    drawn = draw_errors(
        Image.new("RGB", (200, 100)), np.array([[0, 0, 10, 10.0]]), NO_BOXES, NO_BOXES, 50
    )
    assert drawn.size == (50, 25)


def test_a_gallery_needs_room_for_every_tile() -> None:
    project = load_yaml(REPO_ROOT / "configs" / "project.yaml", ProjectConfig)
    data = project.diagnose.gallery.model_dump()
    GalleryConfig(**{**data, "per_dataset": {"a": 16}})
    with pytest.raises(ValidationError):
        GalleryConfig(**{**data, "per_dataset": {"a": 16, "b": 1}})


def test_main_writes_the_gallery_and_counts_the_review(diagnosed: Path) -> None:
    assert evaluate.main([*ARGS, "--from-cache", "run1"]) == 0
    assert diagnose.main(["m", "--project-config", "configs/project.yaml"]) == 0
    assert (diagnosed / "docs" / "assets" / "failures.png").is_file()
    result = load_diagnosis(diagnosed / "reports" / "diagnosis" / "m" / "diagnosis.json")
    # The fake model is perfect at 0.5, so there is nothing to show or review.
    assert (result.gallery, result.review) == ([], [])


def test_review_sheet_writes_crops_and_a_template(
    diagnosed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = yaml.safe_load((diagnosed / "configs" / "project.yaml").read_text("utf-8"))
    # A threshold above the fake model's confidence turns every box into a miss.
    runs = diagnosed / "reports" / "runs"
    metrics = json.loads((runs / "run1" / "metrics.json").read_text("utf-8"))
    metrics["metrics"]["confidence"] = 0.95
    (fuel,) = metrics["metrics"]["classes"]
    fuel.update(true_positives=0, false_negatives=3)
    (runs / "run1" / "metrics.json").write_text(json.dumps(metrics), "utf-8")
    assert diagnose.main(["m", "--review-sheet", "--project-config", "configs/project.yaml"]) == 0
    sheets = diagnosed / project["paths"]["contact_sheet_dir"]
    assert (sheets / "review_alpha_miss_1.jpg").is_file()
    rows = (sheets / "review_template.csv").read_text("utf-8").splitlines()
    assert rows[0] == "dataset,file_name,kind,x1,y1,x2,y2,verdict,note"
    assert len(rows) == 4
    assert not (diagnosed / "reports" / "diagnosis").exists()


def test_main_fails_when_the_matching_disagrees_with_the_published_run(diagnosed: Path) -> None:
    path = diagnosed / "reports" / "runs" / "run1" / "metrics.json"
    metrics = json.loads(path.read_text("utf-8"))
    (fuel,) = metrics["metrics"]["classes"]
    fuel["false_positives"] += 1
    path.write_text(json.dumps(metrics), "utf-8")
    with pytest.raises(CountMismatchError, match="run1 fuel"):
        diagnose.main(["m", "--project-config", "configs/project.yaml"])


def _set_config(root: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = root / "configs" / "project.yaml"
    data = yaml.safe_load(path.read_text("utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data), "utf-8")


def _add_false_positive(root: Path) -> None:
    """Give c.jpg a confident box on empty ground, and count it in the run."""
    run = root / "reports" / "runs" / "run1"
    cache = json.loads((run / "predictions.json").read_text("utf-8"))
    cache["images"]["c.jpg"].append(["fuel", 0.9, 40.0, 40.0, 50.0, 50.0])
    (run / "predictions.json").write_text(json.dumps(cache), "utf-8")
    metrics = json.loads((run / "metrics.json").read_text("utf-8"))
    metrics["metrics"]["classes"][0]["false_positives"] += 1
    (run / "metrics.json").write_text(json.dumps(metrics), "utf-8")


def _slice_by_source(data: dict[str, Any]) -> None:
    d = data["diagnose"]
    d["sources"] = {"alpha": [{"name": "first two", "pattern": "^[ab]"}]}
    d["gallery"]["per_dataset"] = {"alpha": 2}
    d["gallery"]["exclude"] = {}
    d["gallery"]["exclude_sources"] = {}
    # A recording pattern the fixture's names do not match, so frames are unknown.
    data["splits"]["datasets"]["alpha"] = {
        "method": "temporal",
        "recording_pattern": r"^(?P<recording>.+?)_(?P<frame>\d+)\.jpg$",
    }


def test_a_false_positive_reaches_the_slices_gallery_and_review(
    diagnosed: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _add_false_positive(diagnosed)
    _set_config(diagnosed, _slice_by_source)
    args = ["m", "--project-config", "configs/project.yaml"]
    assert diagnose.main(args) == 0
    result = load_diagnosis(diagnosed / "reports" / "diagnosis" / "m" / "diagnosis.json")
    (fuel,) = result.splits[0].classes
    source = next(s for s in fuel.slicings if s.by == "source")
    assert [(x.name, x.images, x.false_positives) for x in source.slices] == [
        ("first two", 2, 0),
        ("other", 1, 1),
    ]
    assert [(t.file_name, t.kind, t.source) for t in result.gallery] == [
        ("c.jpg", "false positive", "other")
    ]
    # main configures logging itself, which replaces pytest's log capture.
    assert "the gallery shows 1 of 2 alpha tiles" in capsys.readouterr().err
    assert result.review == []

    assert diagnose.main([*args, "--review-sheet"]) == 0
    template = diagnosed / "data" / "contact_sheets" / "review_template.csv"
    (row,) = template.read_text("utf-8").splitlines()[1:]
    review = diagnosed / "reports" / "diagnosis" / "review.csv"
    review.write_text(
        template.read_text("utf-8").replace(row, row.replace(",,", ",other,")), "utf-8"
    )
    assert diagnose.main(args) == 0
    result = load_diagnosis(diagnosed / "reports" / "diagnosis" / "m" / "diagnosis.json")
    (count,) = result.review
    assert (count.kind, count.total, count.sampled, count.verdicts) == (
        "false positive",
        1,
        1,
        {"other": 1},
    )


def test_the_run_check_counts_boxes_matched_to_another_class(diagnosed: Path) -> None:
    run = RunResult.model_validate_json(
        (diagnosed / "reports" / "runs" / "run1" / "metrics.json").read_text("utf-8")
    )
    (fuel,) = run.metrics.classes
    robot = fuel.model_copy(update={"name": "robot"})

    def published(fuel_counts: tuple[int, int, int]) -> RunResult:
        tp, fp, fn = fuel_counts
        classes = [
            fuel.model_copy(
                update={"true_positives": tp, "false_positives": fp, "false_negatives": fn}
            ),
            robot.model_copy(
                update={"true_positives": 0, "false_positives": 1, "false_negatives": 0}
            ),
        ]
        metrics = run.metrics.model_copy(update={"classes": classes})
        return run.model_copy(update={"scored_classes": ["fuel", "robot"], "metrics": metrics})

    # A robot box on the labeled ball is a confusion: a false positive for
    # robot and a miss for fuel. The fuel box on empty ground is a false positive.
    labels = [dets([(0, 0, 10, 10)], [FUEL])]
    preds = [dets([(0, 0, 10, 10), (50, 50, 60, 60)], [ROBOT, FUEL], [0.9, 0.9])]

    def cases(result: RunResult) -> SplitCases:
        return SplitCases(result, diagnosed, [], ["fuel", "robot"], preds, labels)

    check_run_totals(cases(published((0, 1, 1))))
    with pytest.raises(CountMismatchError, match="fuel"):
        check_run_totals(cases(published((1, 1, 0))))


def test_a_missing_cache_is_an_error(diagnosed: Path) -> None:
    (diagnosed / "reports" / "runs" / "run1" / "predictions.json").unlink()
    with pytest.raises(ConfigError, match="which is missing"):
        diagnose.main(["m", "--project-config", "configs/project.yaml"])


def test_an_unknown_model_is_an_error(diagnosed: Path) -> None:
    with pytest.raises(ConfigError, match="is not in"):
        diagnose.main(["nope", "--project-config", "configs/project.yaml"])


def test_edges_need_values_and_no_edges_make_one_bin() -> None:
    with pytest.raises(ValueError, match="no values"):
        quantile_edges([], 3)
    assert bin_names([]) == ["all"]


def test_a_source_pattern_must_compile() -> None:
    with pytest.raises(ValidationError):
        SourcePattern(name="bad", pattern="(")


def test_main_applies_the_edge_rule_a_run_recorded(
    workspace: Path, fake: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = workspace / "configs/project.yaml"
    project = yaml.safe_load(config.read_text(encoding="utf-8"))
    project["evaluate"]["edge_ignore"]["tolerance_px"] = 5
    config.write_text(yaml.safe_dump(project), encoding="utf-8")
    assert evaluate.main(ARGS) == 0
    for name in ("git_is_dirty", "git_commit", "git_tree"):
        monkeypatch.setattr(diagnose, name, getattr(evaluate, name))
    train_dir = workspace / "data" / "harmonized" / "alpha" / "train"
    records = [ImageRecord(ImageRef("alpha", "train", "t.jpg"), "t.jpg", 64, 64, ())]
    write_split(train_dir, to_coco(records, ["fuel", "robot"]))
    smooth_image(9).save(train_dir / "t.jpg")
    assert diagnose.main(["m", "--project-config", "configs/project.yaml"]) == 0
    result = load_diagnosis(workspace / "reports" / "diagnosis" / "m" / "diagnosis.json")
    (fuel,) = result.splits[0].classes
    assert (fuel.overall.hits, fuel.overall.false_positives, fuel.overall.misses) == (2, 0, 0)
