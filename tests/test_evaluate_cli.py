import json
import shutil
import sys
import types
from itertools import count
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml
from conftest import smooth_image, write_split

from frc_xdata import evaluate
from frc_xdata.config import API_KEY_VAR, ModelEntry
from frc_xdata.errors import ConfigError, DirtyTreeError
from frc_xdata.evaluate import Prediction, parse_response
from frc_xdata.harmonize import to_coco
from frc_xdata.inspect_datasets import Box, ImageRecord, ImageRef

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-test-secret"
ENTRY = ModelEntry(
    dataset="alpha",
    project="proj",
    version=1,
    model_id="proj-1-tiny",
    architecture="tiny",
    input_size="64x64",
)
# Three test images: two fuel boxes, one fuel box, and a background image.
IMAGES = {
    "a.jpg": (Box("fuel", 4, 4, 10, 10), Box("fuel", 30, 30, 12, 12)),
    "b.jpg": (Box("fuel", 20, 8, 10, 10),),
    "c.jpg": (),
}
ARGS = ["m", "alpha", "--project-config", "configs/project.yaml"]
ARGS += ["--datasets-config", "configs/datasets.yaml"]


class FakePredictor:
    """Predict every labeled box exactly, with one extra low-confidence box on c.jpg."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._server = {"backend": "fake"}

    @property
    def server(self) -> dict[str, str]:
        return self._server

    def predict(self, image: Path) -> list[Prediction]:
        self.calls.append(image.name)
        preds = [
            Prediction(class_name="fuel", confidence=0.9, xyxy=(b.x, b.y, b.x + b.w, b.y + b.h))
            for b in IMAGES[image.name]
        ]
        if image.name == "c.jpg":
            preds.append(Prediction(class_name="fuel", confidence=0.2, xyxy=(0, 0, 8, 8)))
        return preds


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A repo-shaped directory with one harmonized fuel-only dataset and one model."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, SECRET)
    monkeypatch.setattr(evaluate, "git_is_dirty", lambda repo: False)
    monkeypatch.setattr(evaluate, "git_commit", lambda repo: "abc123")
    ids = count(1)
    monkeypatch.setattr(evaluate, "_run_id", lambda *a: f"run{next(ids)}")

    configs = tmp_path / "configs"
    configs.mkdir()
    project = yaml.safe_load((REPO_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    project["evaluate"]["bootstrap"]["resamples"] = 20
    (configs / "project.yaml").write_text(yaml.safe_dump(project), encoding="utf-8")
    datasets = {"datasets": {"alpha": {"workspace": "team", "project": "p", "version": 3}}}
    datasets["datasets"]["alpha"]["license"] = "CC BY 4.0"
    (configs / "datasets.yaml").write_text(yaml.safe_dump(datasets), encoding="utf-8")

    split_dir = tmp_path / "data" / "harmonized" / "alpha" / "test"
    records = [
        ImageRecord(ImageRef("alpha", "test", name), name, 64, 64, boxes)
        for name, boxes in IMAGES.items()
    ]
    write_split(split_dir, to_coco(records, ["fuel", "robot"]))
    for seed, name in enumerate(IMAGES):
        smooth_image(seed).save(split_dir / name)

    reports = tmp_path / "reports"
    (reports / "data_manifests").mkdir(parents=True)
    (reports / "data_manifests" / "alpha.sha256").write_text("digest\n", encoding="utf-8")
    coverage = {"alpha": {"labeled": ["fuel"]}}
    (reports / "class_coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    model = {**ENTRY.model_dump(), "notes": "written by hand"}
    (reports / "models.yaml").write_text(yaml.safe_dump({"models": {"m": model}}), "utf-8")
    return tmp_path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakePredictor:
    predictor = FakePredictor()
    monkeypatch.setattr(evaluate, "HostedPredictor", lambda *a: predictor)
    return predictor


def read(run: str, name: str) -> dict[str, Any]:
    return json.loads((Path("reports/runs") / run / name).read_text(encoding="utf-8"))


def test_main_writes_predictions_metrics_and_meta(workspace: Path, fake: FakePredictor) -> None:
    assert evaluate.main(ARGS) == 0
    assert fake.calls == ["a.jpg", "b.jpg", "c.jpg"]
    assert sorted(p.name for p in (workspace / "reports/runs/run1").iterdir()) == [
        "meta.json",
        "metrics.json",
        "predictions.json",
    ]
    metrics = read("run1", "metrics.json")
    (fuel,) = metrics["metrics"]["classes"]
    assert (fuel["precision"], fuel["recall"], fuel["instances"]) == (1.0, 1.0, 3)
    assert metrics["predictions_from"] == "this run"
    assert [u["images"] for u in metrics["units"]] == [1, 1, 1]
    assert metrics["bootstrap"]["resamples"] == 20

    meta = read("run1", "meta.json")
    assert meta["git"] == {"commit": "abc123", "dirty": False}
    assert meta["dataset"]["universe"]["version"] == 3
    assert meta["dataset"]["manifest_sha256"]
    assert meta["model"]["notes"] == "written by hand"
    assert meta["model"]["server"] == {"backend": "fake"}
    assert meta["packages"]["supervision"]
    assert SECRET not in json.dumps(meta)


def test_from_cache_scores_the_same_without_calling_the_model(
    workspace: Path, fake: FakePredictor
) -> None:
    assert evaluate.main(ARGS) == 0
    calls = len(fake.calls)
    assert evaluate.main([*ARGS, "--from-cache", "run1"]) == 0
    assert len(fake.calls) == calls
    first, second = read("run1", "metrics.json"), read("run2", "metrics.json")
    for key in ("metrics", "units", "bootstrap"):
        assert first[key] == second[key]
    assert second["predictions_from"] == "run1"
    assert not (workspace / "reports/runs/run2/predictions.json").exists()


def test_from_cache_of_a_slice_cannot_score_the_whole_split(
    workspace: Path, fake: FakePredictor
) -> None:
    assert evaluate.main([*ARGS, "--limit", "2"]) == 0
    assert read("run1", "metrics.json")["metrics"]["images"] == 2
    with pytest.raises(ConfigError, match="no cached predictions"):
        evaluate.main([*ARGS, "--from-cache", "run1"])


def test_dirty_tree_is_refused_unless_allowed(
    workspace: Path, fake: FakePredictor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "git_is_dirty", lambda repo: True)
    with pytest.raises(DirtyTreeError):
        evaluate.main(ARGS)
    assert not fake.calls
    assert evaluate.main([*ARGS, "--allow-dirty"]) == 0
    assert read("run1", "meta.json")["git"]["dirty"] is True


def test_existing_run_directory_is_never_overwritten(
    workspace: Path, fake: FakePredictor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "_run_id", lambda *a: "same")
    assert evaluate.main(ARGS) == 0
    with pytest.raises(ConfigError, match="already exists"):
        evaluate.main(ARGS)


def test_unknown_model_is_refused(workspace: Path, fake: FakePredictor) -> None:
    with pytest.raises(ConfigError, match="nope"):
        evaluate.main(["nope", *ARGS[1:]])


def test_inference_errors_exit_cleanly_without_the_key(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Failing(FakePredictor):
        def predict(self, image: Path) -> list[Prediction]:
            raise requests.ConnectionError(f"https://example.test/proj/1?api_key={SECRET}")

    monkeypatch.setattr(evaluate, "HostedPredictor", lambda *a: Failing())
    assert evaluate.main(ARGS) == 1
    err = capsys.readouterr().err
    assert "ConnectionError" in err
    assert "<redacted>" in err
    assert SECRET not in err


def test_missing_inference_extra_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "inference_sdk", None)
    with pytest.raises(ConfigError, match="make setup-infer"):
        evaluate.HostedPredictor("https://example.test", SECRET, ENTRY, 0.01)


def response(model_id: str = "some-workspace/proj-1-tiny") -> dict[str, Any]:
    return {
        "resolved_model": {"model_id": model_id, "backend": "trt", "quantization": "fp16"},
        "image": {"width": 64, "height": 64},
        "predictions": [
            {"x": 20.0, "y": 30.0, "width": 10.0, "height": 6.0, "confidence": 0.8, "class": "fuel"}
        ],
    }


def test_parse_response_converts_centers_to_corners_and_keeps_only_backend() -> None:
    server: dict[str, str] = {}
    (p,) = parse_response(response(), "proj-1-tiny", server)
    assert p == Prediction(class_name="fuel", confidence=0.8, xyxy=(15.0, 27.0, 25.0, 33.0))
    assert server == {"backend": "trt", "quantization": "fp16"}


def test_parse_response_rejects_another_model_without_naming_the_workspace() -> None:
    with pytest.raises(ConfigError) as e:
        parse_response(response("some-workspace/other-model"), "proj-1-tiny", {})
    assert "other-model" in str(e.value)
    assert "some-workspace" not in str(e.value)


def test_hosted_predictor_calls_the_model_by_project_and_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class Client:
        def __init__(self, api_url: str, api_key: str) -> None:
            seen["api_url"] = api_url

        def configure(self, config: dict[str, Any]) -> None:
            seen["config"] = config

        def infer(self, inference_input: str, model_id: str) -> dict[str, Any]:
            seen["model_id"] = model_id
            return response()

    sdk = types.ModuleType("inference_sdk")
    sdk.InferenceHTTPClient = Client  # type: ignore[attr-defined]
    sdk.InferenceConfiguration = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "inference_sdk", sdk)

    predictor = evaluate.HostedPredictor("https://example.test", SECRET, ENTRY, 0.01)
    assert len(predictor.predict(tmp_path / "x.jpg")) == 1
    assert seen["model_id"] == "proj/1"
    assert seen["config"] == {"confidence_threshold": 0.01, "api_key_transport": "both"}
    assert predictor.server["backend"] == "trt"


def test_evaluate_needs_harmonized_data(workspace: Path, fake: FakePredictor) -> None:
    shutil.rmtree(workspace / "data")
    with pytest.raises(ConfigError, match="make harmonize"):
        evaluate.main(ARGS)


def test_run_id_names_the_model_dataset_split_and_slice() -> None:
    assert evaluate._run_id("m", "alpha", "test", None).startswith("m__alpha-test__")
    assert "__alpha-valid__first5__" in evaluate._run_id("m", "alpha", "valid", 5)


def test_missing_manifest_stops_before_the_model_is_called(
    workspace: Path, fake: FakePredictor
) -> None:
    (workspace / "reports/data_manifests/alpha.sha256").unlink()
    with pytest.raises(ConfigError, match="make download"):
        evaluate.main(ARGS)
    assert not fake.calls
    assert not (workspace / "reports/runs").exists()


def write_split_units(workspace: Path, units: int) -> None:
    report = {"datasets": {"alpha": {"splits": {"test": {"images": 3, "units": units}}}}}
    (workspace / "reports/splits.json").write_text(json.dumps(report), encoding="utf-8")


def use_grouped_split(workspace: Path) -> None:
    path = workspace / "configs/project.yaml"
    project = yaml.safe_load(path.read_text(encoding="utf-8"))
    project["splits"]["datasets"]["alpha"] = {
        "method": "grouped",
        "recording_pattern": r"^(?P<recording>.+[_-]mp4)[_-]\d+\.jpg$",
    }
    path.write_text(yaml.safe_dump(project), encoding="utf-8")


def test_grouped_units_are_rebuilt_from_image_hashes(workspace: Path, fake: FakePredictor) -> None:
    use_grouped_split(workspace)
    write_split_units(workspace, 3)
    assert evaluate.main(ARGS) == 0
    assert read("run1", "metrics.json")["bootstrap"]["units"] == 3


def test_units_that_disagree_with_the_split_report_stop_the_run(
    workspace: Path, fake: FakePredictor
) -> None:
    write_split_units(workspace, 2)
    with pytest.raises(ConfigError, match="rebuilt 3 units"):
        evaluate.main(ARGS)
    assert not fake.calls
    # A slice cannot be checked against whole-split counts.
    assert evaluate.main([*ARGS, "--limit", "1"]) == 0
