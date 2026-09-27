from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from frc_xdata.config import API_KEY_VAR, load_yaml, roboflow_api_key
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


def test_load_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_yaml(write(tmp_path, "seed: 1\nname: x\nextra: 2\n"), Sample)


def test_api_key_read_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, "  test-key  ")
    assert roboflow_api_key() == "test-key"


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
