import json
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from frc_xdata.config import SplitMethod
from frc_xdata.errors import ConfigError, PredictionCacheTooLargeError, UnmappedLabelError
from frc_xdata.evaluate import (
    Interval,
    Prediction,
    align,
    bootstrap,
    image_units,
    load_predictions,
    save_predictions,
    to_detections,
    unit_counts,
)
from frc_xdata.inspect_datasets import Box, ImageRecord, ImageRef

CLASSES = ["fuel", "robot"]
LIMIT = 10_000
TEMPORAL = SplitMethod(
    method="temporal", recording_pattern=r"^(?P<recording>.+?)[_-](?P<frame>\d+)\.jpg$"
)
GROUPED = SplitMethod(method="grouped", recording_pattern=r"^(?P<recording>.+[_-]mp4)[_-]\d+\.jpg$")
FAR_APART = [0, 2**64 - 1]


def pred(conf: float, cls: str = "fuel", box: tuple[float, ...] = (1.0, 2.0, 11.0, 12.0)):
    x1, y1, x2, y2 = box
    return Prediction(class_name=cls, confidence=conf, xyxy=(x1, y1, x2, y2))


def test_cache_round_trip_rounds_and_applies_the_floor(tmp_path: Path) -> None:
    path = tmp_path / "run" / "predictions.json"
    predictions = {
        "a.jpg": [pred(0.912345, box=(1.26, 2.0, 11.04, 12.55)), pred(0.004)],
        "b.jpg": [],
    }
    save_predictions(path, predictions, floor=0.01, max_bytes=LIMIT)
    loaded = load_predictions(path)
    assert list(loaded) == ["a.jpg", "b.jpg"]
    assert loaded["b.jpg"] == []
    (kept,) = loaded["a.jpg"]
    assert kept.confidence == 0.9123
    assert kept.xyxy == (1.3, 2.0, 11.0, 12.6)


def test_cache_is_one_image_per_line(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    save_predictions(path, {"a.jpg": [pred(0.5)], "b.jpg": []}, floor=0.01, max_bytes=LIMIT)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["confidence_floor"] == 0.01
    assert [line for line in text.splitlines() if line.startswith('"a.jpg"')]
    assert [line for line in text.splitlines() if line.startswith('"b.jpg": []')]


def test_cache_over_the_size_limit_raises_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    predictions = {f"{i}.jpg": [pred(0.5)] * 5 for i in range(20)}
    with pytest.raises(PredictionCacheTooLargeError, match="floor"):
        save_predictions(path, predictions, floor=0.01, max_bytes=500)
    assert not path.exists()


def test_to_detections_matches_classes_by_name() -> None:
    detections = to_detections([pred(0.9, "robot"), pred(0.8, "fuel")], CLASSES)
    assert detections.class_id.tolist() == [1, 0]
    assert detections.confidence.tolist() == [0.9, 0.8]
    assert len(to_detections([], CLASSES)) == 0


def test_to_detections_rejects_a_class_the_dataset_lacks() -> None:
    with pytest.raises(UnmappedLabelError, match="sports ball"):
        to_detections([pred(0.9, "sports ball")], CLASSES)


def test_align_follows_dataset_order_and_rejects_missing_images() -> None:
    predictions = {"a.jpg": [pred(0.9)], "b.jpg": []}
    aligned = align(predictions, ["b.jpg", "a.jpg"], CLASSES)
    assert [len(d) for d in aligned] == [0, 1]
    np.testing.assert_allclose(aligned[1].xyxy, [[1.0, 2.0, 11.0, 12.0]])
    with pytest.raises(ConfigError, match=r"c\.jpg"):
        align(predictions, ["a.jpg", "c.jpg"], CLASSES)


def record(source: str, labels: tuple[str, ...] = ()) -> ImageRecord:
    boxes = tuple(Box(label, 0, 0, 5, 5) for label in labels)
    return ImageRecord(ImageRef("alpha", "test", f"{source}.rf.jpg"), source, 64, 64, boxes)


def test_temporal_units_are_recordings() -> None:
    records = [record("run1_0001.jpg"), record("run1_0002.jpg"), record("run2-0007.jpg")]
    assert image_units(records, TEMPORAL, None, 4) == ["run1", "run1", "run2"]


def test_temporal_unit_needs_a_recording_name() -> None:
    with pytest.raises(ConfigError, match=r"photo\.jpg"):
        image_units([record("photo.jpg")], TEMPORAL, None, 4)


def test_grouped_units_join_recordings_and_near_duplicates() -> None:
    records = [
        record("match_mp4-1.jpg"),
        record("match_mp4-9.jpg"),
        record("a.jpg"),
        record("b.jpg"),
        record("c.jpg"),
    ]
    # a and b are one bit apart, c is far from both.
    hashes = [FAR_APART[0], FAR_APART[1], 12345, 12344, 0xFFFF << 48]
    units = image_units(records, GROUPED, hashes, 4)
    assert units[0] == units[1] == "match_mp4-1.jpg.rf.jpg"
    assert units[2] == units[3] == "a.jpg.rf.jpg"
    assert units[4] == "c.jpg.rf.jpg"


def test_grouped_units_need_hashes() -> None:
    with pytest.raises(ConfigError, match="hashes"):
        image_units([record("a.jpg")], GROUPED, None, 4)


def test_datasets_without_a_split_method_have_one_unit_per_image() -> None:
    records = [record("a.jpg"), record("b.jpg")]
    assert image_units(records, None, None, 4) == ["a.jpg.rf.jpg", "b.jpg.rf.jpg"]


def test_unit_counts_sum_to_the_split_and_skip_unscored_classes() -> None:
    records = [
        record("run1_1.jpg", ("fuel", "fuel")),
        record("run1_2.jpg", ("fuel", "robot")),
        record("run2_1.jpg", ()),
    ]
    counts = unit_counts(records, ["run1", "run1", "run2"], ["fuel"])
    assert [(c.unit, c.images, c.boxes) for c in counts] == [
        ("run1", 2, {"fuel": 3}),
        ("run2", 1, {"fuel": 0}),
    ]
    assert sum(c.images for c in counts) == len(records)


def fuel(*boxes: list[int], conf: float | None = None) -> sv.Detections:
    xyxy = np.array(boxes, dtype=float)
    confidence = None if conf is None else np.full(len(boxes), conf)
    return sv.Detections(xyxy=xyxy, class_id=np.zeros(len(boxes), dtype=int), confidence=confidence)


def run_bootstrap(units: list[str], seed: int = 7):
    hit, miss = [0, 0, 10, 10], [20, 20, 30, 30]
    targets = [fuel(hit), fuel(miss), fuel(hit), fuel(miss)]
    predictions = [
        fuel(hit, conf=0.9),
        sv.Detections.empty(),
        fuel(hit, conf=0.9),
        fuel([50, 50, 60, 60], conf=0.9),
    ]
    return bootstrap(
        predictions,
        targets,
        units,
        CLASSES,
        ["fuel"],
        confidence=0.5,
        resamples=50,
        level=0.9,
        seed=seed,
    )


def test_bootstrap_is_repeatable_for_a_seed() -> None:
    units = ["r1", "r1", "r2", "r3"]
    assert run_bootstrap(units) == run_bootstrap(units)


def test_bootstrap_interval_brackets_the_point_value() -> None:
    result = run_bootstrap(["r1", "r2", "r3", "r4"])
    (interval,) = result.classes
    assert result.units == 4
    assert interval.recall.low <= 0.5 <= interval.recall.high
    assert interval.recall.low < interval.recall.high


def test_bootstrap_over_one_unit_collapses_to_the_point_value() -> None:
    (interval,) = run_bootstrap(["r1"] * 4).classes
    assert interval.recall.low == interval.recall.high == 0.5


def test_bootstrap_skips_resamples_without_a_labeled_box() -> None:
    hit = [0, 0, 10, 10]
    result = bootstrap(
        [fuel(hit, conf=0.9), sv.Detections.empty()],
        [fuel(hit), sv.Detections.empty()],
        ["r1", "r2"],
        CLASSES,
        ["fuel", "robot"],
        confidence=0.5,
        resamples=50,
        level=0.9,
        seed=3,
    )
    fuel_interval, robot_interval = result.classes
    # A resample of only r2 has no fuel to find, which says nothing about recall.
    assert fuel_interval.recall == fuel_interval.map50 == Interval(low=1.0, high=1.0)
    assert (robot_interval.map50, robot_interval.recall) == (None, None)
