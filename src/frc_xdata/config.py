"""Load and validate configuration from YAML files and the environment."""

import os
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from frc_xdata.errors import ConfigError

API_KEY_VAR = "ROBOFLOW_API_KEY"
UNIVERSE_URL = "https://universe.roboflow.com"

ModelT = TypeVar("ModelT", bound=BaseModel)

# Keys become directory and file names, so keep them path-safe.
DatasetKey = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$")]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]


def _relative(path: Path) -> Path:
    if path.is_absolute():
        raise ValueError(f"{path} must be relative to the repo root")
    return path


RelativePath = Annotated[Path, AfterValidator(_relative)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Paths(_Frozen):
    """Directories the pipeline reads and writes, relative to the repo root."""

    raw_dir: RelativePath
    reports_dir: RelativePath
    manifests_dir: RelativePath
    assets_dir: RelativePath
    field_test_dir: RelativePath


class AreaBuckets(_Frozen):
    """Upper edges of the small and medium box sizes, as a fraction of image area."""

    small_max: Annotated[float, Field(gt=0, lt=1)]
    medium_max: Annotated[float, Field(gt=0, lt=1)]

    @model_validator(mode="after")
    def _ordered(self) -> "AreaBuckets":
        if self.small_max >= self.medium_max:
            raise ValueError("small_max must be below medium_max")
        return self


class GridConfig(_Frozen):
    """Layout of the per-dataset sample grid."""

    rows: PositiveInt
    cols: PositiveInt
    tile_px: PositiveInt
    max_bytes: PositiveInt
    exclude: dict[DatasetKey, list[str]] = {}


class InspectConfig(_Frozen):
    """Settings for dataset inspection."""

    area_buckets: AreaBuckets
    box_tolerance_px: Annotated[float, Field(ge=0)]
    near_duplicate_max_distance: Annotated[int, Field(ge=0, le=64)]
    max_examples: PositiveInt
    grid: GridConfig


class ProjectConfig(_Frozen):
    """Settings shared by every stage, from ``configs/project.yaml``."""

    seed: int
    paths: Paths
    inspect: InspectConfig


class DatasetSpec(_Frozen):
    """One Roboflow Universe dataset and the exact version the pipeline uses.

    ``project`` and ``version`` may be left empty while a candidate is still
    being looked up. Such a dataset can be resolved but not downloaded.
    """

    workspace: Slug
    project: Slug | None = None
    version: PositiveInt | None = None
    format: Literal["coco"] = "coco"
    license: str
    note: str = ""

    @property
    def pinned(self) -> bool:
        """Return True if both the project and the version are set."""
        return self.project is not None and self.version is not None

    @property
    def url(self) -> str:
        """Return the Universe page for the project, or the workspace if unset."""
        if self.project is None:
            return f"{UNIVERSE_URL}/{self.workspace}"
        return f"{UNIVERSE_URL}/{self.workspace}/{self.project}"


class DatasetsConfig(_Frozen):
    """Every candidate dataset by key, from ``configs/datasets.yaml``."""

    datasets: dict[DatasetKey, DatasetSpec]


def load_yaml(path: Path, model: type[ModelT]) -> ModelT:
    """Read a YAML file and validate it against a pydantic model.

    Raises:
        ConfigError: If the file is missing or unreadable, is not valid YAML,
            or does not match the model. The message names the file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except OSError as e:
        raise ConfigError(f"cannot read config file {path}: {e}") from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path} does not match {model.__name__}:\n{e}") from e


def roboflow_api_key() -> str:
    """Return the Roboflow API key from the environment or a .env file.

    The .env file is searched for from the current directory upward, so
    commands run from the repo root find the one there. A variable already
    set in the environment wins over the file.

    Raises:
        ConfigError: If ``ROBOFLOW_API_KEY`` is unset or empty.
    """
    # Without usecwd, python-dotenv searches upward from this module's file,
    # which ignores the working directory and misses the repo when the
    # package is installed into site-packages.
    load_dotenv(find_dotenv(usecwd=True))
    key = os.environ.get(API_KEY_VAR, "").strip()
    if not key:
        raise ConfigError(f"{API_KEY_VAR} is not set. Copy .env.example to .env and fill it in.")
    return key
