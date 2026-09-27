import json
from pathlib import Path

import numpy as np
import pytest

from frc_xdata.errors import ConfigError, PredictionCacheTooLargeError, UnmappedLabelError
from frc_xdata.evaluate import (
    Prediction,
    align,
    load_predictions,
    save_predictions,
    to_detections,
)

CLASSES = ["fuel", "robot"]
LIMIT = 10_000


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
