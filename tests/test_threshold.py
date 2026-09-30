import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from test_evaluate_cli import ARGS, fake, workspace

from frc_xdata import evaluate, threshold
from frc_xdata.config import ErrorCosts, ModelsFile, ProjectConfig, ThresholdConfig, load_yaml
from frc_xdata.errors import ConfigError, CountMismatchError, DirtyTreeError
from frc_xdata.threshold import (
    Counts,
    cost_per_image,
    flat_band,
    load_choice,
    pick_threshold,
)

# The evaluate CLI fixtures build a harmonized split and a model to run.
__all__ = ["fake", "workspace"]

COSTS = ErrorCosts(false_positive=3, false_negative=1)
SETTINGS: dict[str, Any] = {
    "output_dir": "reports/thresholds",
    "models": ["m"],
    "scored_class": "fuel",
    "grid": {"start": 0.05, "stop": 0.95, "step": 0.05},
    "costs": {"false_positive": 3, "false_negative": 1},
    "tuning": ["alpha"],
    "held_out": {"dataset": "beta", "out_of_scope": ["near"]},
    "band": 0.05,
}


def counts(fp: int, fn: int) -> Counts:
    return Counts(true_positives=0, false_positives=fp, false_negatives=fn)


def test_a_false_positive_costs_three_misses() -> None:
    assert cost_per_image(counts(1, 0), 1, COSTS) == cost_per_image(counts(0, 3), 1, COSTS)
    assert cost_per_image(counts(2, 4), 5, COSTS) == 2.0


def test_a_set_without_images_cannot_be_costed() -> None:
    with pytest.raises(ValueError):
        cost_per_image(counts(0, 0), 0, COSTS)


def test_the_lowest_cost_wins() -> None:
    assert pick_threshold([0.1, 0.2, 0.3], [2.0, 1.0, 3.0]) == 1


def test_a_tie_goes_to_the_higher_threshold_despite_float_noise() -> None:
    # 0.1 + 0.2 is a hair above 0.3 in floating point.
    assert pick_threshold([0.1, 0.2, 0.3], [1.0, 0.3, 0.1 + 0.2]) == 2


def test_costs_must_match_thresholds() -> None:
    with pytest.raises(ValueError):
        pick_threshold([0.1, 0.2], [1.0])


def test_the_band_stops_at_the_first_costlier_threshold() -> None:
    costs = [1.5, 1.02, 1.1, 1.0, 1.04, 1.2, 1.01]
    assert flat_band(costs, 3, 0.05) == (3, 4)
    assert flat_band(costs, 3, 0.15) == (1, 4)


def test_a_band_can_reach_both_ends() -> None:
    assert flat_band([1.0, 1.0, 1.0], 1, 0.0) == (0, 2)


def test_a_held_out_dataset_cannot_also_tune() -> None:
    with pytest.raises(ValidationError, match="both held out"):
        ThresholdConfig.model_validate({**SETTINGS, "tuning": ["beta"]})


def test_there_must_be_a_tuning_dataset() -> None:
    with pytest.raises(ValidationError):
        ThresholdConfig.model_validate({**SETTINGS, "tuning": []})


def test_the_grid_cannot_start_below_the_cached_floor(tmp_path: Path) -> None:
    project = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    project["threshold"] = {**SETTINGS, "grid": {"start": 0.005, "stop": 0.5, "step": 0.005}}
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(project), encoding="utf-8")
    with pytest.raises(ConfigError, match="confidence_floor"):
        load_yaml(path, ProjectConfig)


def _write_settings(workspace: Path, settings: dict[str, Any]) -> None:
    path = workspace / "configs" / "project.yaml"
    project = yaml.safe_load(path.read_text(encoding="utf-8"))
    project["threshold"] = settings
    # a.jpg and b.jpg hold the labeled fuel, and c.jpg only a false positive.
    patterns = [{"name": "far", "pattern": "^[ab]"}, {"name": "near", "pattern": "^c"}]
    project["diagnose"]["sources"] = {"beta": patterns}
    path.write_text(yaml.safe_dump(project), encoding="utf-8")


@pytest.fixture
def swept(workspace: Path, fake: object, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The evaluate workspace with runs on alpha, which tunes, and a copy of it, beta, held out."""
    harmonized = workspace / "data" / "harmonized"
    shutil.copytree(harmonized / "alpha", harmonized / "beta")
    reports = workspace / "reports"
    shutil.copy(
        reports / "data_manifests" / "alpha.sha256", reports / "data_manifests" / "beta.sha256"
    )
    coverage = {"alpha": {"labeled": ["fuel"]}, "beta": {"labeled": ["fuel"]}}
    (reports / "class_coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    assert evaluate.main(ARGS) == 0
    assert evaluate.main(["m", "beta", *ARGS[2:]]) == 0
    _write_settings(workspace, SETTINGS)
    for name in ("git_is_dirty", "git_commit", "git_tree"):
        monkeypatch.setattr(threshold, name, getattr(evaluate, name))
    return workspace


def test_main_chooses_the_highest_threshold_that_keeps_every_hit(swept: Path) -> None:
    assert threshold.main(["--project-config", "configs/project.yaml"]) == 0
    out = swept / "reports" / "thresholds" / "m"
    choice = load_choice(out / "threshold.json")
    # Every labeled box is predicted at 0.9 and the one false positive at 0.2,
    # so every threshold above 0.2 up to 0.9 costs nothing.
    assert (choice.threshold, choice.cost, choice.band_low, choice.band_high) == (
        0.9,
        0.0,
        0.25,
        0.9,
    )
    assert [(s.name, s.role, s.images) for s in choice.sets] == [
        ("alpha test", "tuning", 3),
        ("beta far", "held out", 2),
        ("beta near", "out of scope", 1),
    ]
    assert [p.threshold for p in choice.grid] == pytest.approx(
        [round(0.05 * i, 2) for i in range(1, 20)]
    )
    # Only the tuning set is costed on the grid.
    assert {c.name for p in choice.grid for c in p.sets} == {"alpha test"}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["runs"] == ["run1", "run2"]
    assert meta["git"] == {"commit": "abc123", "tree": "def456", "dirty": False}


def test_a_held_out_set_with_nothing_out_of_scope_is_scored_whole(swept: Path) -> None:
    _write_settings(swept, {**SETTINGS, "held_out": {"dataset": "beta"}})
    assert threshold.main(["--project-config", "configs/project.yaml"]) == 0
    choice = load_choice(swept / "reports" / "thresholds" / "m" / "threshold.json")
    assert [(s.name, s.role, s.images) for s in choice.sets] == [
        ("alpha test", "tuning", 3),
        ("beta test", "held out", 3),
    ]


def test_an_unknown_out_of_scope_kind_is_an_error(swept: Path) -> None:
    held_out = {"dataset": "beta", "out_of_scope": ["closeup"]}
    _write_settings(swept, {**SETTINGS, "held_out": held_out})
    with pytest.raises(ConfigError, match="not kinds of image"):
        threshold.main(["--project-config", "configs/project.yaml"])


def test_main_refuses_a_dirty_tree(swept: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threshold, "git_is_dirty", lambda repo: True)
    with pytest.raises(DirtyTreeError):
        threshold.main(["--project-config", "configs/project.yaml"])


def test_a_model_without_a_published_run_is_an_error(swept: Path) -> None:
    with pytest.raises(ConfigError, match="no published run"):
        threshold.main(["--model", "m2", "--project-config", "configs/project.yaml"])


def test_each_recorded_deploy_threshold_matches_its_committed_choice() -> None:
    root = Path(__file__).resolve().parents[1]
    models = load_yaml(root / "reports" / "models.yaml", ModelsFile).models
    project = load_yaml(root / "configs" / "project.yaml", ProjectConfig)
    assert project.threshold is not None
    for name in project.threshold.models:
        chosen = load_choice(root / project.threshold.output_dir / name / "threshold.json")
        assert models[name].deploy_confidence == chosen.threshold


def test_parts_that_lose_an_image_are_an_error(
    swept: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = threshold.split_held_out

    def lossy(*args: Any) -> Any:
        in_scope, *rest = split(*args)
        keep = [i > 0 for i in range(in_scope.images)]
        return [in_scope.subset(in_scope.name, "held out", keep), *rest]

    monkeypatch.setattr(threshold, "split_held_out", lossy)
    with pytest.raises(CountMismatchError, match="add up to"):
        threshold.main(["--project-config", "configs/project.yaml"])


def test_an_empty_part_is_an_error(swept: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    split = threshold.split_held_out

    def emptied(*args: Any) -> Any:
        in_scope, *rest = split(*args)
        return [in_scope, *(r.subset(r.name, r.role, [False] * r.images) for r in rest)]

    monkeypatch.setattr(threshold, "split_held_out", emptied)
    with pytest.raises(CountMismatchError, match="has no images"):
        threshold.main(["--project-config", "configs/project.yaml"])


def test_units_that_disagree_with_the_run_are_an_error(
    swept: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(threshold, "image_units", lambda records, *a: ["one"] * len(records))
    with pytest.raises(ConfigError, match="rebuilt 1 units"):
        threshold.main(["--project-config", "configs/project.yaml"])
