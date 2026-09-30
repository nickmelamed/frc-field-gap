"""Load and validate configuration from YAML files and the environment."""

import os
import re
from itertools import pairwise
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
    contact_sheet_dir: RelativePath
    harmonized_dir: RelativePath
    platform_dir: RelativePath


class AreaBuckets(_Frozen):
    """Upper edges of the small and medium box sizes, as a fraction of image area."""

    small_max: Annotated[float, Field(gt=0, lt=1)]
    medium_max: Annotated[float, Field(gt=0, lt=1)]

    @model_validator(mode="after")
    def _ordered(self) -> "AreaBuckets":
        if self.small_max >= self.medium_max:
            raise ValueError("small_max must be below medium_max")
        return self


class TileLayout(_Frozen):
    """Rows and columns of square cells, each ``tile_px`` on a side."""

    rows: PositiveInt
    cols: PositiveInt
    tile_px: PositiveInt


class GridConfig(TileLayout):
    """Layout of the per-dataset sample grid.

    ``exclude`` lists file names, and ``exclude_patterns`` holds patterns
    matched against source names, for a set of images too large to list.
    """

    max_bytes: PositiveInt
    exclude: dict[DatasetKey, list[str]] = {}
    exclude_patterns: dict[DatasetKey, list[str]] = {}

    @model_validator(mode="after")
    def _patterns_compile(self) -> "GridConfig":
        for patterns in self.exclude_patterns.values():
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise ValueError(f"exclude pattern {pattern!r} does not compile: {e}") from e
        return self


class InspectConfig(_Frozen):
    """Settings for dataset inspection."""

    area_buckets: AreaBuckets
    box_tolerance_px: Annotated[float, Field(ge=0)]
    near_duplicate_max_distance: Annotated[int, Field(ge=0, le=64)]
    max_examples: PositiveInt
    grid: GridConfig
    contact_sheet: TileLayout


class SplitFractions(_Frozen):
    """Share of images that goes to each split."""

    train: Annotated[float, Field(gt=0, lt=1)]
    valid: Annotated[float, Field(gt=0, lt=1)]
    test: Annotated[float, Field(gt=0, lt=1)]

    @model_validator(mode="after")
    def _sum_to_one(self) -> "SplitFractions":
        if abs(self.train + self.valid + self.test - 1) > 1e-9:
            raise ValueError("train, valid, and test must sum to 1")
        return self


class SplitMethod(_Frozen):
    """How one dataset is re-split.

    ``temporal`` cuts each recording in frame order, so the pattern needs a
    ``frame`` group, and recordings whose name matches ``train_only_pattern``
    go whole to train. ``grouped`` assigns whole groups of related images,
    and a name the pattern does not match is its own recording.
    ``eval_only`` puts every image in test, for a dataset that is only
    scored, and groups related images as ``grouped`` does, so the error bars
    resample whole groups.
    """

    method: Literal["temporal", "grouped", "eval_only"]
    recording_pattern: str
    train_only_pattern: str | None = None
    dedupe_copies: bool = False

    @model_validator(mode="after")
    def _pattern_groups(self) -> "SplitMethod":
        try:
            groups = re.compile(self.recording_pattern).groupindex
            if self.train_only_pattern is not None:
                re.compile(self.train_only_pattern)
        except re.error as e:
            raise ValueError(f"pattern does not compile: {e}") from e
        needed = {"recording", "frame"} if self.method == "temporal" else {"recording"}
        if missing := needed - set(groups):
            raise ValueError(f"recording_pattern needs named groups {sorted(missing)}")
        if self.train_only_pattern is not None and self.method != "temporal":
            raise ValueError("train_only_pattern only applies to the temporal method")
        return self


class SplitsConfig(_Frozen):
    """Settings for re-splitting datasets whose own splits leak."""

    fractions: SplitFractions
    buffer_frames: Annotated[int, Field(ge=0)]
    # Cutting needs at least one image each for train, valid, and test.
    min_recording_images: Annotated[int, Field(ge=3)]
    datasets: dict[DatasetKey, SplitMethod]


class PlatformConfig(_Frozen):
    """The Roboflow project each harmonized dataset is uploaded to, by dataset key."""

    projects: dict[DatasetKey, Slug]
    upload_retries: Annotated[int, Field(ge=0)]

    def project(self, key: str) -> str:
        """Return the project for dataset ``key``.

        Raises:
            ConfigError: If ``key`` has no project, so nothing is uploaded to
                a project meant for another dataset.
        """
        if key not in self.projects:
            raise ConfigError(f"platform.projects has no project for {key}")
        return self.projects[key]


Probability = Annotated[float, Field(gt=0, lt=1)]


class ThresholdRange(_Frozen):
    """Evenly spaced confidence thresholds, from ``start`` to ``stop`` inclusive."""

    start: Probability
    stop: Probability
    step: Probability

    @model_validator(mode="after")
    def _ordered(self) -> "ThresholdRange":
        if self.start > self.stop:
            raise ValueError("start must not be above stop")
        return self

    @property
    def values(self) -> list[float]:
        """Return the thresholds, rounded so float steps do not drift."""
        count = round((self.stop - self.start) / self.step) + 1
        return [round(self.start + i * self.step, 6) for i in range(count)]


class BootstrapConfig(_Frozen):
    """How many resamples to draw, and the share of them an interval covers."""

    resamples: PositiveInt
    level: Probability


class EdgeIgnoreConfig(_Frozen):
    """Labels at the frame edge that are left out of scoring (D-027)."""

    classes: list[str]
    tolerance_px: Annotated[float, Field(ge=0)]
    min_overlap: Probability


class EvaluateConfig(_Frozen):
    """Settings for scoring a model on a harmonized split."""

    api_url: str
    models_file: RelativePath
    coverage_file: RelativePath
    runs_dir: RelativePath
    confidence: Probability
    iou: Probability
    confidence_floor: Probability
    max_predictions_per_image: PositiveInt
    pr_thresholds: ThresholdRange
    max_prediction_bytes: PositiveInt
    bootstrap: BootstrapConfig
    edge_ignore: EdgeIgnoreConfig | None = None

    @model_validator(mode="after")
    def _floor_below_thresholds(self) -> "EvaluateConfig":
        if self.confidence_floor > min(self.confidence, self.pr_thresholds.start):
            raise ValueError("confidence_floor must not be above any threshold it is scored at")
        return self


class SourcePattern(_Frozen):
    """A named group of images whose source name matches ``pattern``."""

    name: str
    pattern: str

    @model_validator(mode="after")
    def _compiles(self) -> "SourcePattern":
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"pattern does not compile: {e}") from e
        return self


class GalleryConfig(GridConfig):
    """The failure gallery, with how many tiles each dataset gets."""

    per_dataset: dict[DatasetKey, PositiveInt]
    min_frame_gap: Annotated[int, Field(ge=0)] = 0
    # Sources left out whole, when every frame of one broadcast shows the same faces.
    exclude_sources: dict[DatasetKey, list[str]] = {}

    @model_validator(mode="after")
    def _fits(self) -> "GalleryConfig":
        if sum(self.per_dataset.values()) > self.rows * self.cols:
            raise ValueError("per_dataset asks for more tiles than the gallery has")
        return self


class ReviewConfig(TileLayout):
    """The sample of errors that is checked by eye, and the verdicts allowed."""

    per_kind: PositiveInt
    # Crops show this many box widths of context around the box.
    context: Annotated[float, Field(ge=1)]
    min_crop_px: PositiveInt
    # May hold ``{model}``, since each model's errors get their own verdicts.
    file: RelativePath
    verdicts: list[str]

    def path_for(self, model: str) -> Path:
        """Return the review file for ``model``."""
        return Path(str(self.file).format(model=model))


class DiagnoseConfig(_Frozen):
    """Settings for slicing a model's errors."""

    output_dir: RelativePath
    feature_px: PositiveInt
    localization_floor: Probability
    feature_bins: Annotated[int, Field(ge=2)]
    crowding_edges: list[Annotated[int, Field(ge=0)]]
    quantiles: list[Probability]
    gallery: GalleryConfig
    review: ReviewConfig
    sources: dict[DatasetKey, list[SourcePattern]] = {}

    @model_validator(mode="after")
    def _edges_start_at_zero(self) -> "DiagnoseConfig":
        edges = self.crowding_edges
        if not edges or edges[0] != 0 or any(a >= b for a, b in pairwise(edges)):
            raise ValueError("crowding_edges must start at 0 and increase")
        return self


class MergeSource(_Frozen):
    """How one harmonized dataset enters the merged training set.

    ``keep_splits`` keeps the dataset's train, valid, and test as they are,
    and its valid and test are protected. ``train_only`` puts every image in
    train, once it matches no protected image. Source names matching
    ``exclude_pattern``, and harmonized files in ``exclude_files``, are left
    out of train and valid. ``dedupe_copies``
    keeps one augmented copy of each training photo, as D-014 does.
    """

    role: Literal["keep_splits", "train_only"]
    exclude_pattern: str | None = None
    exclude_files: list[str] = []
    dedupe_copies: bool = False

    @model_validator(mode="after")
    def _pattern_compiles(self) -> "MergeSource":
        if self.exclude_pattern is not None:
            try:
                re.compile(self.exclude_pattern)
            except re.error as e:
                raise ValueError(f"exclude_pattern does not compile: {e}") from e
        return self


class MergeConfig(_Frozen):
    """Settings for merging harmonized datasets into one training set (D-028).

    ``protected`` names datasets that are never trained on, such as the
    lockbox. None of their images, and nothing from the field test set, may
    match a merged training image.
    """

    name: DatasetKey
    sources: dict[DatasetKey, MergeSource]
    protected: list[DatasetKey] = []
    report: RelativePath

    @model_validator(mode="after")
    def _roles_apart(self) -> "MergeConfig":
        if self.name in self.sources or self.name in self.protected:
            raise ValueError(f"{self.name} cannot be both the merged dataset and a source")
        if both := sorted(set(self.sources) & set(self.protected)):
            raise ValueError(f"{both} are both merged and protected")
        if not any(s.role == "keep_splits" for s in self.sources.values()):
            raise ValueError("at least one source must keep its splits, or test would be empty")
        return self


class ErrorCosts(_Frozen):
    """What one false positive and one miss cost the robot, in the same unit."""

    false_positive: Annotated[float, Field(gt=0)]
    false_negative: Annotated[float, Field(gt=0)]


class HeldOutCheck(_Frozen):
    """The dataset scored only at the chosen threshold, and the kinds of image left out (D-032).

    Kinds are the source names under ``diagnose.sources`` for the dataset.
    Out-of-scope kinds show cases the robot never acts on, so they are
    reported apart from the check.
    """

    dataset: DatasetKey
    out_of_scope: list[str] = []


class ThresholdConfig(_Frozen):
    """Settings for choosing a deploy threshold from cached predictions (D-031)."""

    output_dir: RelativePath
    models: list[str]
    # The class the robot acts on. Its errors are the only ones costed.
    scored_class: str
    grid: ThresholdRange
    costs: ErrorCosts
    # Datasets whose test split is scored whole when choosing.
    tuning: Annotated[list[DatasetKey], Field(min_length=1)]
    held_out: HeldOutCheck
    # Thresholds whose mean cost is within this share of the lowest.
    band: Probability

    @model_validator(mode="after")
    def _held_out_apart(self) -> "ThresholdConfig":
        if self.held_out.dataset in self.tuning:
            raise ValueError(f"{self.held_out.dataset} is both held out and a tuning dataset")
        return self


class ProjectConfig(_Frozen):
    """Settings shared by every stage, from ``configs/project.yaml``."""

    seed: int
    paths: Paths
    inspect: InspectConfig
    splits: SplitsConfig
    platform: PlatformConfig
    evaluate: EvaluateConfig
    diagnose: DiagnoseConfig
    merge: MergeConfig | None = None
    threshold: ThresholdConfig | None = None

    @model_validator(mode="after")
    def _grid_above_floor(self) -> "ProjectConfig":
        t = self.threshold
        if t is not None and t.grid.start < self.evaluate.confidence_floor:
            raise ValueError("threshold.grid must not start below evaluate.confidence_floor")
        return self


class ModelEntry(BaseModel):
    """The fields of one ``reports/models.yaml`` entry that evaluation needs.

    The file also records training settings by hand, which are kept in each
    run's metadata as written.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    dataset: DatasetKey
    project: Slug
    version: PositiveInt
    model_id: Slug
    architecture: str
    input_size: str
    # The threshold frc-threshold chose for running the model on the robot
    # (D-031). Scores in the report stay at evaluate.confidence.
    deploy_confidence: Probability | None = None


class ModelsFile(BaseModel):
    """Every platform-trained model by run name, from ``reports/models.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: dict[str, ModelEntry]


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


DROP = "DROP"


class ClassMapConfig(_Frozen):
    """Target classes and each dataset's source label mapping.

    Category ids in harmonized files follow the order of ``classes``,
    starting from 1.
    """

    classes: Annotated[list[str], Field(min_length=1)]
    datasets: dict[DatasetKey, dict[str, str]]

    @model_validator(mode="after")
    def _targets_known(self) -> "ClassMapConfig":
        if len(set(self.classes)) != len(self.classes) or DROP in self.classes:
            raise ValueError(f"classes must be unique and must not include {DROP}")
        allowed = {*self.classes, DROP}
        for key, mapping in self.datasets.items():
            for label, target in mapping.items():
                if target not in allowed:
                    raise ValueError(f"{key}: {label!r} maps to unknown class {target!r}")
        return self


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
