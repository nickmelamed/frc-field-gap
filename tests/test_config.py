from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from frc_xdata.config import (
    API_KEY_VAR,
    DatasetsConfig,
    DatasetSpec,
    ProjectConfig,
    load_yaml,
    roboflow_api_key,
)
from frc_xdata.errors import ConfigError


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    name: str


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sample.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_yaml_returns_validated_model(tmp_path: Path) -> None:
    cfg = load_yaml(write(tmp_path, "seed: 1515\nname: mortorq\n"), Sample)
    assert cfg == Sample(seed=1515, name="mortorq")


def test_load_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "absent.yaml", Sample)


def test_load_yaml_bad_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_yaml(write(tmp_path, "seed: [1, 2\n"), Sample)


def test_load_yaml_schema_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not match Sample"):
        load_yaml(write(tmp_path, "seed: one\nname: x\n"), Sample)


def test_load_yaml_unreadable_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_yaml(tmp_path, Sample)


def test_load_yaml_empty_file_raises(tmp_path: Path) -> None:
    # safe_load returns None for an empty file, which must not slip through
    # as a default config.
    with pytest.raises(ConfigError, match="does not match Sample"):
        load_yaml(write(tmp_path, ""), Sample)


def test_load_yaml_top_level_list_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not match Sample"):
        load_yaml(write(tmp_path, "- 1\n- 2\n"), Sample)


@pytest.mark.parametrize(
    "text",
    ["seed: [1, 2\n", "seed: one\nname: x\n", ""],
    ids=["bad-yaml", "schema", "empty"],
)
def test_load_yaml_error_names_the_file(tmp_path: Path, text: str) -> None:
    path = write(tmp_path, text)
    with pytest.raises(ConfigError) as excinfo:
        load_yaml(path, Sample)
    assert str(path) in str(excinfo.value)


def test_load_yaml_keeps_the_original_error_as_cause(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_yaml(tmp_path / "absent.yaml", Sample)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_load_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_yaml(write(tmp_path, "seed: 1\nname: x\nextra: 2\n"), Sample)


def test_api_key_read_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, "  test-key  ")
    assert roboflow_api_key() == "test-key"


def test_api_key_read_from_dotenv_in_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(f"{API_KEY_VAR}=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    assert roboflow_api_key() == "from-file"


def test_api_key_environment_wins_over_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(f"{API_KEY_VAR}=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, "from-env")
    assert roboflow_api_key() == "from-env"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_api_key_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str | None
) -> None:
    # Run from an empty directory so load_dotenv cannot find a developer's
    # real .env file.
    monkeypatch.chdir(tmp_path)
    if value is None:
        monkeypatch.delenv(API_KEY_VAR, raising=False)
    else:
        monkeypatch.setenv(API_KEY_VAR, value)
    with pytest.raises(ConfigError, match=API_KEY_VAR):
        roboflow_api_key()


REPO_ROOT = Path(__file__).resolve().parents[1]

SPEC = {"workspace": "robot-zftp2", "license": "CC BY 4.0"}


def test_committed_project_config_loads() -> None:
    cfg = load_yaml(REPO_ROOT / "configs" / "project.yaml", ProjectConfig)
    assert cfg.paths.raw_dir == Path("data/raw")


def test_project_config_rejects_absolute_paths(tmp_path: Path) -> None:
    text = (REPO_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8")
    path = write(tmp_path, text.replace("raw_dir: data/raw", "raw_dir: /data/raw"))
    with pytest.raises(ConfigError, match="must be relative"):
        load_yaml(path, ProjectConfig)


def test_dataset_spec_defaults_to_unpinned_coco() -> None:
    spec = DatasetSpec.model_validate(SPEC)
    assert spec.format == "coco"
    assert not spec.pinned
    assert spec.url == "https://universe.roboflow.com/robot-zftp2"


def test_dataset_spec_pinned_needs_project_and_version() -> None:
    assert not DatasetSpec.model_validate({**SPEC, "project": "rebuilt"}).pinned
    assert not DatasetSpec.model_validate({**SPEC, "version": 3}).pinned
    spec = DatasetSpec.model_validate({**SPEC, "project": "rebuilt", "version": 3})
    assert spec.pinned
    assert spec.url == "https://universe.roboflow.com/robot-zftp2/rebuilt"


@pytest.mark.parametrize(
    "change",
    [
        {"version": 0},
        {"format": "yolov8"},
        {"workspace": "Robot ZFTP2"},
        {"project": "../escape"},
        {"classes": ["fuel"]},
    ],
    ids=["version-zero", "format", "workspace-case", "project-path", "unknown-field"],
)
def test_dataset_spec_rejects_bad_values(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DatasetSpec.model_validate({**SPEC, **change})


@pytest.mark.parametrize("key", ["MarsWars", "lava-2026", "", "../raw"])
def test_datasets_config_rejects_unsafe_keys(key: str) -> None:
    with pytest.raises(ValidationError):
        DatasetsConfig.model_validate({"datasets": {key: SPEC}})


def test_datasets_config_accepts_snake_case_keys() -> None:
    cfg = DatasetsConfig.model_validate({"datasets": {"robot_zftp2": SPEC}})
    assert list(cfg.datasets) == ["robot_zftp2"]
