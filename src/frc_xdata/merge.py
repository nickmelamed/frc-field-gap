"""Merge harmonized datasets into one training set without leaking any test image.

Reads ``data/harmonized/<key>/`` for every source under ``merge`` in
``configs/project.yaml`` and writes ``data/harmonized/<name>/``. Sources that
keep their splits bring train, valid, and test as they are, so each one's
test split is the same set of images the baseline was scored on. Sources
used for training only bring every image to train, after removing any image
that matches a valid or test image of a kept source or an image of a
protected dataset such as the lockbox. Matches are found by source name and
by perceptual hash under all 8 flips and 90 degree rotations, since some
sources hold flipped and rotated copies (D-028). A field-test image anywhere
in the output stops the run. What was kept and dropped goes to
``reports/merge.json``, and the merged dataset's classes to
``reports/class_coverage.json``.
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from frc_xdata.config import (
    ClassMapConfig,
    MergeConfig,
    ProjectConfig,
    SplitMethod,
    load_yaml,
)
from frc_xdata.download import PROJECT_CONFIG
from frc_xdata.errors import ConfigError, DataLeakError, SplitLeakError
from frc_xdata.harmonize import (
    CLASS_MAP_CONFIG,
    COVERAGE_NAME,
    class_coverage,
    link_or_copy,
    to_coco,
)
from frc_xdata.inspect_datasets import (
    ANNOTATIONS_NAME,
    ImageRecord,
    ImageRef,
    field_test_images,
    load_split,
    phash_image,
)
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.splits import SPLITS, copy_groups, pick_copy

# Separates the source key from the original file name in merged file names.
NAME_SEPARATOR = "__"
# Candidates are compared with protected hashes in blocks of this many rows,
# which keeps the distance matrix small.
BLOCK_ROWS = 2048

logger = logging.getLogger(__name__)


def dihedral(image: Image.Image) -> list[Image.Image]:
    """Return the image under each of the 8 flips and 90 degree rotations, itself first."""
    turns = [image, *(image.rotate(angle, expand=True) for angle in (90, 180, 270))]
    return [*turns, *(ImageOps.mirror(t) for t in turns)]


def dihedral_hashes(path: Path) -> tuple[int, ...]:
    """Return the perceptual hash of an image file under each of the 8 transforms."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    return tuple(phash_image(t) for t in dihedral(rgb))


def near_any(
    candidates: npt.NDArray[np.uint64], protected: npt.NDArray[np.uint64], max_distance: int
) -> npt.NDArray[np.bool_]:
    """Return which candidates have any hash within ``max_distance`` bits of a protected hash.

    Args:
        candidates: One row per image, one column per transform.
        protected: One hash per protected image, untransformed. Comparing
            every transform of a candidate with the protected image as it is
            finds a protected image that is any transform of the candidate.
        max_distance: Largest Hamming distance that counts as a match.
    """
    rows = candidates.reshape(len(candidates), -1)
    found = np.zeros(len(rows), dtype=bool)
    if not len(protected):
        return found
    for start in range(0, len(rows), BLOCK_ROWS):
        block = rows[start : start + BLOCK_ROWS]
        distances = np.bitwise_count(block[:, :, None] ^ protected[None, None, :])
        found[start : start + BLOCK_ROWS] = (distances <= max_distance).any(axis=(1, 2))
    return found


def sha256(path: Path) -> str:
    """Return the SHA256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class MergedImage:
    """One image of the merged dataset and the harmonized file it comes from."""

    record: ImageRecord
    origin: ImageRef


def merged_name(key: str, file_name: str) -> str:
    """Return the file name an image gets in the merged dataset."""
    return f"{key}{NAME_SEPARATOR}{file_name}"


def excluded(record: ImageRecord, pattern: str | None) -> bool:
    """Return True if the record's source name matches ``pattern``."""
    return pattern is not None and re.match(pattern, record.source_name) is not None


@dataclass(frozen=True)
class HeldOutScenes:
    """The recordings that held-out images come from, per dataset.

    ``first_frame`` gives, for a recording cut in frame order, the first
    frame that is not in train. ``recordings`` lists the recordings of
    datasets split by groups, whose held-out recordings have no frame in
    train.
    """

    first_frame: dict[tuple[str, str], int]
    recordings: set[tuple[str, str]]


def held_out_scenes(
    held_out: Iterable[ImageRecord], methods: Mapping[str, SplitMethod]
) -> HeldOutScenes:
    """Collect the recordings behind valid, test, and protected images.

    ``methods`` holds the split method of each dataset that has one.
    """
    first_frame: dict[tuple[str, str], int] = {}
    recordings: set[tuple[str, str]] = set()
    for r in held_out:
        method = methods.get(r.ref.dataset)
        if method is None or (m := re.match(method.recording_pattern, r.source_name)) is None:
            continue
        scene = (r.ref.dataset, m["recording"])
        if method.method == "temporal":
            frame = int(m["frame"])
            first_frame[scene] = min(first_frame.get(scene, frame), frame)
        else:
            recordings.add(scene)
    return HeldOutScenes(first_frame, recordings)


def same_scene(
    record: ImageRecord,
    scenes: HeldOutScenes,
    methods: Mapping[str, SplitMethod],
    buffer_frames: int,
) -> bool:
    """Return True if a training-only image is a frame of a held-out recording.

    Its source name is read with each dataset's recording pattern. For a
    recording cut in frame order, a frame counts when harmonize would have
    kept it out of train, meaning it is within ``buffer_frames`` of the
    first held-out frame or later (D-013). For a grouped dataset, any frame
    of a held-out recording counts.
    """
    for dataset, method in methods.items():
        m = re.match(method.recording_pattern, record.source_name)
        if m is None:
            continue
        scene = (dataset, m["recording"])
        if method.method == "temporal":
            first = scenes.first_frame.get(scene)
            if first is not None and int(m["frame"]) >= first - buffer_frames:
                return True
        elif scene in scenes.recordings:
            return True
    return False


@dataclass(frozen=True)
class MergePlan:
    """Where every source image goes, and why the others were dropped.

    ``dropped`` maps an image to ``copy``, ``excluded``, or the reason it
    was blocked (see :func:`plan_merge`).
    """

    images: dict[str, list[MergedImage]]
    dropped: dict[ImageRef, str]


def plan_merge(
    sources: Mapping[str, Mapping[str, Sequence[ImageRecord]]],
    cfg: MergeConfig,
    blocked: Mapping[ImageRef, str],
) -> MergePlan:
    """Decide the merged split of every image, from matches found beforehand.

    A copy group of a source that keeps one copy per photo is dropped whole
    when any copy is blocked, since its copies show the same photo.

    Args:
        sources: Each source's records by harmonized split.
        cfg: The merge settings.
        blocked: Training-only images that match a held-out image, with the
            reason: ``source_name``, ``same_scene``, or ``near_protected``.

    Raises:
        ConfigError: If a source kept with its splits lacks one of them.
    """
    images: dict[str, list[MergedImage]] = defaultdict(list)
    dropped: dict[ImageRef, str] = {}
    for key, by_split in sorted(sources.items()):
        source = cfg.sources[key]
        if source.role == "keep_splits" and (missing := set(SPLITS) - set(by_split)):
            raise ConfigError(f"{key} keeps its splits but has no {sorted(missing)}")
        if source.dedupe_copies:
            everything = [r for records in by_split.values() for r in records]
            for copies in copy_groups(everything).values():
                reasons = sorted(blocked[c.ref] for c in copies if c.ref in blocked)
                if reasons:
                    dropped.update({c.ref: blocked.get(c.ref, reasons[0]) for c in copies})
                    continue
                keep = pick_copy(copies)
                dropped.update({c.ref: "copy" for c in copies if c is not keep})
        for split, records in sorted(by_split.items()):
            for r in sorted(records, key=lambda r: r.ref.file_name):
                if r.ref in dropped:
                    continue
                target = split if source.role == "keep_splits" else "train"
                if target != "test" and excluded(r, source.exclude_pattern):
                    dropped[r.ref] = "excluded"
                    continue
                if source.role == "train_only" and r.ref in blocked:
                    dropped[r.ref] = blocked[r.ref]
                    continue
                ref = ImageRef(cfg.name, target, merged_name(key, r.ref.file_name))
                images[target].append(MergedImage(replace(r, ref=ref), r.ref))
    return MergePlan(dict(images), dropped)


def check_field_test(
    merged: Mapping[ImageRef, tuple[int, ...]],
    digests: Mapping[ImageRef, str],
    field_hashes: Mapping[ImageRef, int],
    field_digests: Mapping[ImageRef, str],
    max_distance: int,
) -> None:
    """Raise if any merged image is a field-test image, exactly or under a transform.

    Raises:
        DataLeakError: If one is found. The field test set is evaluation
            only, so the run stops instead of dropping the image.
    """
    same = set(field_digests.values())
    leaks = {ref for ref, digest in digests.items() if digest in same}
    refs = list(merged)
    if refs and field_hashes:
        hashes = np.array([merged[r] for r in refs], dtype=np.uint64)
        field = np.array(list(field_hashes.values()), dtype=np.uint64)
        leaks |= {
            r for r, hit in zip(refs, near_any(hashes, field, max_distance), strict=True) if hit
        }
    if leaks:
        first = sorted(map(str, leaks))[0]
        raise DataLeakError(f"{len(leaks)} merged images match the field test set, first {first}")


def check_train_apart(
    train: Mapping[ImageRef, tuple[int, ...]], scored: npt.NDArray[np.uint64], max_distance: int
) -> None:
    """Raise if any merged train image matches a scored image under a transform.

    ``scored`` holds the hashes of every test image and every image of a
    protected dataset. Valid images are left out, since a kept source's
    train may hold frames close to its own valid split (D-013).

    Raises:
        SplitLeakError: If one is found. Harmonize checks the kept sources
            without transforms, so this catches a flipped or rotated copy it
            could not see.
    """
    refs = list(train)
    if not refs:
        return
    hashes = np.array([train[r] for r in refs], dtype=np.uint64)
    leaks = [r for r, hit in zip(refs, near_any(hashes, scored, max_distance), strict=True) if hit]
    if leaks:
        raise SplitLeakError(
            f"{len(leaks)} merged train images match a test or protected image, first {leaks[0]}"
        )


def _hash_all(paths: Sequence[Path]) -> list[tuple[int, ...]]:
    with ThreadPoolExecutor() as pool:
        return list(pool.map(dihedral_hashes, paths))


def _load(harmonized_dir: Path, key: str) -> dict[str, list[ImageRecord]]:
    dataset_dir = harmonized_dir / key
    splits: dict[str, list[ImageRecord]] = {
        s: load_split(dataset_dir / s, key)
        for s in SPLITS
        if (dataset_dir / s / ANNOTATIONS_NAME).is_file()
    }
    if not splits:
        raise ConfigError(f"{dataset_dir} has no harmonized splits, run make harmonize first")
    return splits


def _path(harmonized_dir: Path, ref: ImageRef) -> Path:
    return harmonized_dir / ref.dataset / ref.split / ref.file_name


def write_merged(plan: MergePlan, harmonized_dir: Path, name: str, classes: Sequence[str]) -> None:
    """Write the merged splits, hard-linking each image from its harmonized file.

    Each image's ``extra`` records the source dataset, its source name, and
    the harmonized split it came from.
    """
    out = harmonized_dir / name
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        images = plan.images.get(split, [])
        split_dir = out / split
        split_dir.mkdir(parents=True)
        for m in images:
            link_or_copy(_path(harmonized_dir, m.origin), split_dir / m.record.ref.file_name)
        coco = to_coco([m.record for m in images], classes)
        origins = {m.record.ref.file_name: m.origin for m in images}
        for entry in coco["images"]:
            origin = origins[entry["file_name"]]
            entry["extra"] = {
                "name": entry["extra"]["name"],
                "source_dataset": origin.dataset,
                "source_split": origin.split,
            }
        (split_dir / ANNOTATIONS_NAME).write_text(
            json.dumps(coco, indent=1) + "\n", encoding="utf-8"
        )


def merge_report(
    plan: MergePlan, cfg: MergeConfig, field_images: int, max_distance: int
) -> dict[str, Any]:
    """Build ``reports/merge.json``.

    The merged dataset's split sizes sit under ``datasets.<name>.splits``,
    the same place ``reports/splits.json`` keeps them, so the upload check
    reads either file the same way.
    """
    from_source: dict[str, Counter[str]] = defaultdict(Counter)
    for split, images in plan.images.items():
        for m in images:
            from_source[m.origin.dataset][split] += 1
    dropped: dict[str, Counter[str]] = defaultdict(Counter)
    for ref, reason in plan.dropped.items():
        dropped[ref.dataset][reason] += 1
    return {
        "near_duplicate_max_distance": max_distance,
        "field_test_images": field_images,
        "protected": list(cfg.protected),
        "datasets": {
            cfg.name: {
                "sources": {k: s.model_dump() for k, s in sorted(cfg.sources.items())},
                "splits": {s: {"images": len(plan.images.get(s, []))} for s in SPLITS},
                "from_source": {
                    k: {s: counts[s] for s in SPLITS} for k, counts in sorted(from_source.items())
                },
                "dropped": {k: dict(sorted(c.items())) for k, c in sorted(dropped.items())},
            }
        },
    }


def run_merge(project: ProjectConfig, class_map: ClassMapConfig) -> MergePlan:
    """Merge the configured sources, check for leaks, and write the dataset and reports.

    Raises:
        ConfigError: If merging is not configured or a source is missing.
        DataLeakError: If a merged image matches a field-test image.
        SplitLeakError: If a merged train image matches a test image or an
            image of a protected dataset.
    """
    cfg = project.merge
    if cfg is None:
        raise ConfigError("configs/project.yaml has no merge section")
    harmonized = project.paths.harmonized_dir
    max_distance = project.inspect.near_duplicate_max_distance
    sources = {key: _load(harmonized, key) for key in sorted(cfg.sources)}

    held_out = [
        (split, r)
        for key, by_split in sources.items()
        if cfg.sources[key].role == "keep_splits"
        for split in ("valid", "test")
        for r in by_split[split]
    ]
    for key in cfg.protected:
        held_out += [("test", r) for records in _load(harmonized, key).values() for r in records]
    protected_names = {r.source_name for _, r in held_out}
    methods = {
        key: method
        for key, method in project.splits.datasets.items()
        if key in cfg.sources or key in cfg.protected
    }
    scenes = held_out_scenes((r for _, r in held_out), methods)
    logger.info("hashing %d valid, test, and protected images", len(held_out))
    held_hashes = [h[0] for h in _hash_all([_path(harmonized, r.ref) for _, r in held_out])]
    protected = np.array(held_hashes, dtype=np.uint64)
    scored = np.array(
        [h for (split, _), h in zip(held_out, held_hashes, strict=True) if split == "test"],
        dtype=np.uint64,
    )

    candidates = [
        r
        for key, by_split in sources.items()
        if cfg.sources[key].role == "train_only"
        for records in by_split.values()
        for r in records
    ]
    logger.info("hashing %d training-only images under 8 transforms", len(candidates))
    candidate_hashes = np.array(
        _hash_all([_path(harmonized, r.ref) for r in candidates]), dtype=np.uint64
    ).reshape(len(candidates), -1)
    hits = near_any(candidate_hashes, protected, max_distance)
    blocked: dict[ImageRef, str] = {}
    for r, hit in zip(candidates, hits, strict=True):
        if r.source_name in protected_names:
            blocked[r.ref] = "source_name"
        elif same_scene(r, scenes, methods, project.splits.buffer_frames):
            blocked[r.ref] = "same_scene"
        elif hit:
            blocked[r.ref] = "near_protected"

    plan = plan_merge(sources, cfg, blocked)
    by_origin = dict(zip((r.ref for r in candidates), map(tuple, candidate_hashes), strict=True))
    kept_train = [m.origin for m in plan.images.get("train", []) if m.origin not in by_origin]
    logger.info("hashing %d kept-split train images under 8 transforms", len(kept_train))
    by_origin |= dict(
        zip(kept_train, _hash_all([_path(harmonized, r) for r in kept_train]), strict=True)
    )
    check_train_apart(
        {m.origin: by_origin[m.origin] for m in plan.images.get("train", [])},
        scored,
        max_distance,
    )

    field = field_test_images(project.paths.field_test_dir)
    if field:
        rest = [
            m.origin
            for s in ("valid", "test")
            for m in plan.images.get(s, [])
            if m.origin not in by_origin
        ]
        by_origin |= dict(zip(rest, _hash_all([_path(harmonized, r) for r in rest]), strict=True))
        merged = [m.origin for images in plan.images.values() for m in images]
        check_field_test(
            {r: by_origin[r] for r in merged},
            {r: sha256(_path(harmonized, r)) for r in merged},
            {ref: dihedral_hashes(p)[0] for ref, p in field},
            {ref: sha256(p) for ref, p in field},
            max_distance,
        )
    else:
        logger.warning(
            "%s has no images, so there is nothing to check against", project.paths.field_test_dir
        )

    write_merged(plan, harmonized, cfg.name, class_map.classes)
    report = merge_report(plan, cfg, len(field), max_distance)
    cfg.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    coverage_path = project.paths.reports_dir / COVERAGE_NAME
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    by_split: dict[str, list[ImageRecord]] = {
        s: [m.record for m in plan.images.get(s, [])] for s in SPLITS
    }
    coverage[cfg.name] = class_coverage({cfg.name: by_split}, class_map.classes)[cfg.name]
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "wrote %s: %s, dropped %s",
        harmonized / cfg.name,
        report["datasets"][cfg.name]["splits"],
        report["datasets"][cfg.name]["dropped"],
    )
    return plan


def main(argv: list[str] | None = None) -> int:
    """Run the merge command line and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    parser.add_argument("--class-map", type=Path, default=CLASS_MAP_CONFIG)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    run_merge(
        load_yaml(args.project_config, ProjectConfig), load_yaml(args.class_map, ClassMapConfig)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
