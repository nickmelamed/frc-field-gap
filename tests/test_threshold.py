import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from test_evaluate_cli import ARGS, fake, workspace

from frc_xdata import evaluate, threshold
from frc_xdata.config import ErrorCosts, ModelsFile, ProjectConfig, ThresholdConfig, load_yaml
from frc_xdata.errors import ConfigError, DirtyTreeError
from frc_xdata.threshold import (
    Counts,
    cost_per_image,
    flat_band,
    load_choice,
    pick_threshold,
    tuning_units,
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
    "tuning": [],
    "lockbox": {"dataset": "alpha", "tune_fraction": 0.5},
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


UNITS = ["w1", "w1", "w2", "w3", "w4", "v1", "v2", "p1"]
KINDS = ["web", "web", "web", "web", "web", "video", "video", "photo"]


def test_the_tuning_half_takes_a_share_of_every_kind() -> None:
    picked = tuning_units(UNITS, KINDS, 0.5, 2026)
    assert len(picked & {"w1", "w2", "w3", "w4"}) == 2
    assert len(picked & {"v1", "v2"}) == 1
    # Half of one unit rounds to none, so a kind that small stays held out.
    assert "p1" not in picked


def test_the_tuning_half_depends_only_on_the_units_and_the_seed() -> None:
    order = [4, 0, 7, 2, 5, 1, 6, 3]
    shuffled = tuning_units([UNITS[i] for i in order], [KINDS[i] for i in order], 0.5, 2026)
    assert shuffled == tuning_units(UNITS, KINDS, 0.5, 2026)
    many = [f"u{i}" for i in range(40)]
    assert tuning_units(many, ["k"] * 40, 0.5, 1) != tuning_units(many, ["k"] * 40, 0.5, 2)


def test_the_lockbox_cannot_also_tune() -> None:
    with pytest.raises(ValidationError, match="both the lockbox"):
        ThresholdConfig.model_validate({**SETTINGS, "tuning": ["alpha"]})


def test_the_grid_cannot_start_below_the_cached_floor(tmp_path: Path) -> None:
    project = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    project["threshold"] = {**SETTINGS, "grid": {"start": 0.005, "stop": 0.5, "step": 0.005}}
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(project), encoding="utf-8")
    with pytest.raises(ConfigError, match="confidence_floor"):
        load_yaml(path, ProjectConfig)


@pytest.fixture
def swept(workspace: Path, fake: object, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The evaluate workspace after one live run, with alpha as a lockbox of one kind."""
    assert evaluate.main(ARGS) == 0
    path = workspace / "configs" / "project.yaml"
    project = yaml.safe_load(path.read_text(encoding="utf-8"))
    project["threshold"] = SETTINGS
    project["diagnose"]["sources"] = {"alpha": [{"name": "all", "pattern": "."}]}
    path.write_text(yaml.safe_dump(project), encoding="utf-8")
    for name in ("git_is_dirty", "git_commit", "git_tree"):
        monkeypatch.setattr(threshold, name, getattr(evaluate, name))
    return workspace


def test_main_chooses_the_highest_threshold_that_keeps_every_hit(swept: Path) -> None:
    assert threshold.main(["--project-config", "configs/project.yaml"]) == 0
    out = swept / "reports" / "thresholds" / "m"
    choice = load_choice(out / "threshold.json")
    # Every labeled box is predicted at 0.9 and the one false positive at 0.2,
    # so every threshold above 0.2 up to 0.9 costs nothing.
    assert (choice.threshold, choice.cost, choice.band_high) == (0.9, 0.0, 0.9)
    tuning, held_out = choice.sets
    assert (tuning.role, held_out.role) == ("tuning", "held out")
    assert tuning.images + held_out.images == 3
    assert set(choice.lockbox.tuning).isdisjoint(choice.lockbox.held_out)
    assert len(choice.lockbox.tuning) + len(choice.lockbox.held_out) == 3
    assert [p.threshold for p in choice.grid] == pytest.approx(
        [round(0.05 * i, 2) for i in range(1, 20)]
    )
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["runs"] == ["run1"]
    assert meta["git"] == {"commit": "abc123", "tree": "def456", "dirty": False}


def test_main_refuses_a_dirty_tree(swept: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threshold, "git_is_dirty", lambda repo: True)
    with pytest.raises(DirtyTreeError):
        threshold.main(["--project-config", "configs/project.yaml"])


def test_a_model_without_a_lockbox_run_is_an_error(swept: Path) -> None:
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
