"""Choose the confidence threshold the robot runs a model at.

``frc-threshold`` reads the cached predictions of a model's published test
runs and makes no hosted calls. At every threshold on a grid it counts the
scored class's hits, false positives, and misses the way a run does, prices
them with the configured costs, and picks the threshold with the lowest mean
cost per image. The lockbox is cut by unit into a half that helps choose and
a half that is scored only at the chosen threshold, so one result is free of
the choice (D-031, D-032).
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import supervision as sv
from pydantic import BaseModel, ConfigDict

from frc_xdata.config import ErrorCosts, ModelsFile, ProjectConfig, ThresholdConfig, load_yaml
from frc_xdata.diagnose import check_run_totals, error_counts, image_sources, load_cases
from frc_xdata.download import PROJECT_CONFIG
from frc_xdata.errors import ConfigError, CountMismatchError, DirtyTreeError
from frc_xdata.evaluate import (
    DECIMALS,
    META_NAME,
    Interval,
    Prediction,
    RunResult,
    above,
    bootstrap,
    image_units,
    limit_predictions,
    load_predictions,
    per_image_limit,
    precision_recall,
    restrict,
)
from frc_xdata.harmonize import SPLITS_NAME
from frc_xdata.inspect_datasets import ImageRecord, phash
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.provenance import (
    git_commit,
    git_is_dirty,
    git_tree,
    package_versions,
    sha256_file,
    utc_timestamp,
)
from frc_xdata.runs import load_runs, predictions_path, select_runs

CHOICE_NAME = "threshold.json"
TEST = "test"

Role = Literal["tuning", "held out"]

logger = logging.getLogger(__name__)


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Counts(_Record):
    """Hits, false positives, and misses of one class."""

    true_positives: int
    false_positives: int
    false_negatives: int


class SetCost(_Record):
    """The scored class's errors on one set at one threshold, and what they cost per image."""

    name: str
    counts: Counts
    cost: float


class GridPoint(_Record):
    """Every tuning set's cost at one threshold, and their mean."""

    threshold: float
    cost: float
    sets: list[SetCost]


class ClassScore(_Record):
    """One class's scores on one set at one threshold."""

    name: str
    # None when the model predicts no box of the class at this threshold.
    precision: float | None
    recall: float
    counts: Counts


class SetScore(_Record):
    """Every scored class on one set at one threshold."""

    threshold: float
    cost: float
    classes: list[ClassScore]
    # Images where the per-class prediction limit may have dropped boxes at
    # or above the threshold, so false positives are a lower bound (D-020).
    images_above_limit: int


class SetResult(_Record):
    """One set scored at the chosen threshold and at the neutral one."""

    name: str
    dataset: str
    role: Role
    run_id: str
    images: int
    units: int
    chosen: SetScore
    neutral: SetScore


class LockboxSplit(_Record):
    """Which lockbox units helped choose and which were held out."""

    dataset: str
    tune_fraction: float
    seed: int
    tuning: list[str]
    held_out: list[str]


class ThresholdChoice(_Record):
    """The chosen threshold for one model and what backs it."""

    model: str
    scored_class: str
    costs: ErrorCosts
    threshold: float
    cost: float
    # The run of thresholds around the choice whose mean cost stays within
    # ``band`` of the lowest.
    band: float
    band_low: float
    band_high: float
    neutral_threshold: float
    neutral_cost: float
    sets: list[SetResult]
    # Recall of the scored class on the held-out half at the chosen
    # threshold, resampling whole units.
    held_out_recall: Interval | None
    lockbox: LockboxSplit
    grid: list[GridPoint]


@dataclass(frozen=True)
class ScoredSet:
    """Images whose errors are counted together, taken from one published run."""

    name: str
    role: Role
    result: RunResult
    classes: list[str]
    records: list[ImageRecord]
    predictions: list[sv.Detections]
    labels: list[sv.Detections]
    units: list[str]
    # Each image's cached predictions under the run's limit, for D-020.
    cached: list[list[Prediction]]

    @property
    def images(self) -> int:
        """Return how many images the set holds."""
        return len(self.labels)

    def subset(self, name: str, role: Role, keep: Sequence[bool]) -> "ScoredSet":
        """Return the images where ``keep`` is True."""
        picked = [i for i, k in enumerate(keep) if k]
        return ScoredSet(
            name=name,
            role=role,
            result=self.result,
            classes=self.classes,
            records=[self.records[i] for i in picked],
            predictions=[self.predictions[i] for i in picked],
            labels=[self.labels[i] for i in picked],
            units=[self.units[i] for i in picked],
            cached=[self.cached[i] for i in picked],
        )


def cost_per_image(counts: Counts, images: int, costs: ErrorCosts) -> float:
    """Return what a set's false positives and misses cost, divided by its images."""
    if images < 1:
        raise ValueError("a set needs at least one image to be costed")
    total = costs.false_positive * counts.false_positives
    total += costs.false_negative * counts.false_negatives
    return total / images


def pick_threshold(thresholds: Sequence[float], costs: Sequence[float]) -> int:
    """Return the index of the lowest cost, taking the higher threshold on a tie.

    A tie goes to the higher threshold, since at equal cost the robot is
    better off chasing fewer boxes. Costs are compared after rounding away
    float noise, so equal counts tie exactly.
    """
    if not thresholds or len(thresholds) != len(costs):
        raise ValueError("need one cost for each threshold")
    return min(range(len(costs)), key=lambda i: (round(costs[i], 9), -thresholds[i]))


def flat_band(costs: Sequence[float], best: int, band: float) -> tuple[int, int]:
    """Return the first and last index of the run around ``best`` within ``band`` of its cost."""
    limit = costs[best] * (1 + band) + 1e-12
    low = best
    while low > 0 and costs[low - 1] <= limit:
        low -= 1
    high = best
    while high < len(costs) - 1 and costs[high + 1] <= limit:
        high += 1
    return low, high


def tuning_units(
    units: Sequence[str], kinds: Sequence[str], fraction: float, seed: int
) -> frozenset[str]:
    """Pick the lockbox units that help choose the threshold, a share of each kind.

    Units of each kind of image are ordered by a hash of the seed and the
    unit name, and the first ``fraction`` of them are taken, so both halves
    hold every kind. A unit's kind is that of its first image by position.

    Args:
        units: The unit of each image.
        kinds: The kind of each image, such as webcam frames, in the same order.
        fraction: Share of each kind's units to take.
        seed: Seed for the ordering, so the halves are repeatable.
    """
    kind_of: dict[str, str] = {}
    for unit, kind in zip(units, kinds, strict=True):
        kind_of.setdefault(unit, kind)
    picked: set[str] = set()
    for kind in sorted(set(kind_of.values())):
        members = sorted(
            (u for u, k in kind_of.items() if k == kind),
            key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).hexdigest(),
        )
        picked.update(members[: round(fraction * len(members))])
    return frozenset(picked)


def _counts(values: tuple[int, int, int]) -> Counts:
    tp, fp, fn = values
    return Counts(true_positives=tp, false_positives=fp, false_negatives=fn)


def scored_ids(s: ScoredSet) -> list[int]:
    """Return the class ids the set's run scored."""
    return [s.classes.index(c) for c in s.result.scored_classes]


def set_cost(s: ScoredSet, scored_class: str, threshold: float, cfg: ThresholdConfig) -> SetCost:
    """Return the scored class's errors on ``s`` at ``threshold`` and their cost."""
    ids = scored_ids(s)
    found = error_counts(s.predictions, s.labels, ids, threshold, s.result.metrics.iou)
    counts = _counts(found[s.classes.index(scored_class)])
    return SetCost(name=s.name, counts=counts, cost=cost_per_image(counts, s.images, cfg.costs))


def score_set(s: ScoredSet, threshold: float, cfg: ThresholdConfig) -> SetScore:
    """Return every scored class's precision, recall, and counts on ``s`` at ``threshold``."""
    ids = scored_ids(s)
    found = error_counts(s.predictions, s.labels, ids, threshold, s.result.metrics.iou)
    preds = [above(restrict(p, ids), threshold) for p in s.predictions]
    labels = [restrict(t, ids) for t in s.labels]
    precision, recall = precision_recall(preds, labels, ids)
    classes = [
        ClassScore(
            name=s.classes[c],
            precision=None if p is None else round(p, DECIMALS),
            recall=round(r, DECIMALS),
            counts=_counts(found[c]),
        )
        for c, p, r in zip(ids, precision, recall, strict=True)
    ]
    main = _counts(found[s.classes.index(cfg.scored_class)])
    limit = per_image_limit(s.cached, _per_image(s.result), threshold)
    return SetScore(
        threshold=threshold,
        cost=round(cost_per_image(main, s.images, cfg.costs), DECIMALS),
        classes=classes,
        images_above_limit=limit.images_above_threshold,
    )


def _per_image(result: RunResult) -> int:
    if result.per_image_limit is None:
        raise ConfigError(f"{result.run_id} does not record its per-image limit")
    return result.per_image_limit.per_image


def load_set(project: ProjectConfig, result: RunResult, role: Role) -> ScoredSet:
    """Load one published run as a set, after checking it reproduces the run's counts.

    Raises:
        CountMismatchError: If the counts rebuilt from the cache differ from
            the published ones.
        ConfigError: If the rebuilt units disagree with the run's.
    """
    cases = load_cases(project, result)
    check_run_totals(cases)
    names = [r.ref.file_name for r in cases.records]
    method = project.splits.datasets.get(result.dataset)
    hashes = (
        [phash(cases.split_dir / n) for n in names]
        if method is not None and method.method != "temporal"
        else None
    )
    units = image_units(cases.records, method, hashes, project.inspect.near_duplicate_max_distance)
    if len(set(units)) != result.bootstrap.units:
        raise ConfigError(
            f"{result.run_id} rebuilt {len(set(units))} units, "
            f"but the run resampled {result.bootstrap.units}"
        )
    cache = load_predictions(predictions_path(project.evaluate.runs_dir, result))
    per_image = _per_image(result)
    cached = [limit_predictions(cache[n], per_image, result.scored_classes) for n in names]
    return ScoredSet(
        name=f"{result.dataset} {result.split}",
        role=role,
        result=result,
        classes=cases.classes,
        records=cases.records,
        predictions=cases.predictions,
        labels=cases.labels,
        units=units,
        cached=cached,
    )


def split_lockbox(
    project: ProjectConfig, whole: ScoredSet, cfg: ThresholdConfig
) -> tuple[ScoredSet, ScoredSet, LockboxSplit]:
    """Cut the lockbox into its tuning half and its held-out half.

    Raises:
        ConfigError: If the lockbox has no source patterns to stratify by.
    """
    sources = image_sources(project, whole.result.dataset, whole.records)
    if sources is None:
        raise ConfigError(f"diagnose.sources has no patterns for {whole.result.dataset}")
    kinds, _ = sources
    picked = tuning_units(whole.units, kinds, cfg.lockbox.tune_fraction, project.seed)
    keep = [u in picked for u in whole.units]
    dataset = whole.result.dataset
    tuning = whole.subset(f"{dataset} tuning half", "tuning", keep)
    held_out = whole.subset(f"{dataset} held-out half", "held out", [not k for k in keep])
    split = LockboxSplit(
        dataset=dataset,
        tune_fraction=cfg.lockbox.tune_fraction,
        seed=project.seed,
        tuning=sorted(picked),
        held_out=sorted(set(whole.units) - picked),
    )
    return tuning, held_out, split


def _published(project: ProjectConfig, model: str, dataset: str) -> RunResult:
    runs = select_runs(load_runs(project.evaluate.runs_dir))
    for run in runs:
        r = run.result
        if (r.model, r.dataset, r.split) == (model, dataset, TEST):
            return r
    raise ConfigError(f"{model} has no published run on {dataset} {TEST}")


def _result(s: ScoredSet, chosen: float, neutral: float, cfg: ThresholdConfig) -> SetResult:
    return SetResult(
        name=s.name,
        dataset=s.result.dataset,
        role=s.role,
        run_id=s.result.run_id,
        images=s.images,
        units=len(set(s.units)),
        chosen=score_set(s, chosen, cfg),
        neutral=score_set(s, neutral, cfg),
    )


def choose_threshold(project: ProjectConfig, model: str) -> ThresholdChoice:
    """Sweep the grid for ``model`` and return the chosen threshold with its scores.

    Raises:
        ConfigError: If there are no threshold settings, the model is
            unknown, or a run it needs is not published.
    """
    cfg = project.threshold
    if cfg is None:
        raise ConfigError("configs/project.yaml has no threshold section")
    ev = project.evaluate
    if model not in load_yaml(ev.models_file, ModelsFile).models:
        raise ConfigError(f"{model} is not in {ev.models_file}")

    whole_sets = [load_set(project, _published(project, model, d), "tuning") for d in cfg.tuning]
    lockbox = load_set(project, _published(project, model, cfg.lockbox.dataset), "held out")
    lock_tuning, held_out, split = split_lockbox(project, lockbox, cfg)
    tuning = [*whole_sets, lock_tuning]
    for s in tuning:
        if cfg.scored_class not in s.result.scored_classes:
            raise ConfigError(f"{s.result.run_id} does not score {cfg.scored_class}")

    thresholds = cfg.grid.values
    grid = []
    for t in thresholds:
        sets = [set_cost(s, cfg.scored_class, t, cfg) for s in tuning]
        grid.append((t, float(np.mean([c.cost for c in sets])), sets))
    means = [cost for _, cost, _ in grid]
    best = pick_threshold(thresholds, means)
    low, high = flat_band(means, best, cfg.band)
    chosen = thresholds[best]
    logger.info("%s: chose %s, mean cost %.3f per image", model, chosen, means[best])

    neutral = ev.confidence
    neutral_mean = float(
        np.mean([set_cost(s, cfg.scored_class, neutral, cfg).cost for s in tuning])
    )
    cls = held_out.classes.index(cfg.scored_class)
    resampled = bootstrap(
        held_out.predictions,
        held_out.labels,
        held_out.units,
        held_out.classes,
        [cfg.scored_class],
        confidence=chosen,
        resamples=ev.bootstrap.resamples,
        level=ev.bootstrap.level,
        seed=project.seed,
    )
    (interval,) = [c.recall for c in resampled.classes if c.name == held_out.classes[cls]]

    _check_halves(lockbox, [lock_tuning, held_out], cfg)
    return ThresholdChoice(
        model=model,
        scored_class=cfg.scored_class,
        costs=cfg.costs,
        threshold=chosen,
        cost=round(means[best], DECIMALS),
        band=cfg.band,
        band_low=thresholds[low],
        band_high=thresholds[high],
        neutral_threshold=neutral,
        neutral_cost=round(neutral_mean, DECIMALS),
        sets=[_result(s, chosen, neutral, cfg) for s in [*tuning, held_out]],
        held_out_recall=interval,
        lockbox=split,
        grid=[
            GridPoint(
                threshold=t,
                cost=round(cost, DECIMALS),
                sets=[c.model_copy(update={"cost": round(c.cost, DECIMALS)}) for c in sets],
            )
            for t, cost, sets in grid
        ],
    )


def _check_halves(whole: ScoredSet, halves: Sequence[ScoredSet], cfg: ThresholdConfig) -> None:
    """Fail unless the halves' counts at the run's threshold add up to the whole lockbox's.

    Raises:
        CountMismatchError: If any image was lost or counted twice.
    """
    t = whole.result.metrics.confidence
    whole_counts = set_cost(whole, cfg.scored_class, t, cfg).counts
    parts = [set_cost(h, cfg.scored_class, t, cfg).counts for h in halves]
    summed = tuple(
        sum(getattr(p, f) for p in parts)
        for f in ("true_positives", "false_positives", "false_negatives")
    )
    want = (whole_counts.true_positives, whole_counts.false_positives, whole_counts.false_negatives)
    if summed != want or sum(h.images for h in halves) != whole.images:
        raise CountMismatchError(f"lockbox halves add up to {summed}, not {want}")


def threshold_meta(
    repo: Path,
    project: ProjectConfig,
    project_config: Path,
    choice: ThresholdChoice,
    argv: Sequence[str],
    dirty: bool,
) -> dict[str, object]:
    """Return the inputs, hashes, and versions behind one choice."""
    ev = project.evaluate
    configs = [project_config, ev.models_file, project.paths.reports_dir / SPLITS_NAME]
    return {
        "created": utc_timestamp(),
        "command": ["frc-threshold", *argv],
        "git": {"commit": git_commit(repo), "tree": git_tree(repo), "dirty": dirty},
        "configs": {str(p): sha256_file(p) if p.is_file() else None for p in configs},
        "runs": sorted({s.run_id for s in choice.sets}),
        "packages": package_versions(),
    }


def write_choice(out_dir: Path, choice: ThresholdChoice, meta: dict[str, object]) -> None:
    """Write ``threshold.json`` and ``meta.json``, rendering both before writing either."""
    files = {
        CHOICE_NAME: json.dumps(choice.model_dump(mode="json"), indent=2) + "\n",
        META_NAME: json.dumps(meta, indent=2) + "\n",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    logger.info("wrote %s", ", ".join(str(out_dir / n) for n in files))


def load_choice(path: Path) -> ThresholdChoice:
    """Read a ``threshold.json`` written by :func:`write_choice`."""
    return ThresholdChoice.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Run the threshold command line and return the exit code."""
    parser = argparse.ArgumentParser(
        description="Choose each model's deploy threshold from its cached test predictions."
    )
    parser.add_argument("--model", help="choose for this model only, such as merged-noaug")
    parser.add_argument("--allow-dirty", action="store_true", help="run from a dirty tree")
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    add_log_level_argument(parser)
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)
    setup_logging(args.log_level)

    repo = Path.cwd()
    dirty = git_is_dirty(repo)
    if dirty and not args.allow_dirty:
        raise DirtyTreeError("commit or stash changes first, or pass --allow-dirty")
    project = load_yaml(args.project_config, ProjectConfig)
    if project.threshold is None:
        raise ConfigError(f"{args.project_config} has no threshold section")
    models = [args.model] if args.model else project.threshold.models
    for model in models:
        choice = choose_threshold(project, model)
        meta = threshold_meta(repo, project, args.project_config, choice, raw_args, dirty)
        write_choice(project.threshold.output_dir / model, choice, meta)
    return 0
