"""Rewrite every downloaded dataset to the two-class schema.

Reads ``data/raw/<key>/``, maps each source label through
``configs/class_map.yaml``, re-splits the datasets listed under ``splits`` in
``configs/project.yaml``, and writes COCO files with hard-linked images to
``data/harmonized/<key>/<split>/``. Other datasets keep their own splits.
Label counts before and after mapping go to ``reports/harmonize_counts.csv``,
the classes each dataset labels to ``reports/class_coverage.json``, and what
the re-split kept and dropped to ``reports/splits.json``.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from frc_xdata.config import (
    DROP,
    ClassMapConfig,
    DatasetsConfig,
    ProjectConfig,
    SplitMethod,
    load_yaml,
)
from frc_xdata.download import DATASETS_CONFIG, PROJECT_CONFIG, read_manifest
from frc_xdata.errors import SplitLeakError, UnmappedLabelError
from frc_xdata.inspect_datasets import (
    ANNOTATIONS_NAME,
    ImageRecord,
    ImageRef,
    find_splits,
    load_split,
    near_duplicate_pairs,
    phash,
    write_csv,
)
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.splits import (
    DROP_REASONS,
    SPLITS,
    SplitResult,
    check_no_leak,
    copy_groups,
    split_dataset,
)

CLASS_MAP_CONFIG = Path("configs/class_map.yaml")
COUNTS_NAME = "harmonize_counts.csv"
COVERAGE_NAME = "class_coverage.json"
SPLITS_NAME = "splits.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelCount:
    """Boxes of one source label in one split, and the class they became."""

    dataset: str
    split: str
    source_label: str
    target: str
    instances: int


def map_labels(
    records: Sequence[ImageRecord], mapping: Mapping[str, str]
) -> tuple[list[ImageRecord], list[LabelCount]]:
    """Map every box label to a target class, removing boxes mapped to DROP.

    Images keep their place even when all their boxes are dropped, so an
    image that only showed a dropped label becomes a background image.

    Raises:
        UnmappedLabelError: If a box label is missing from ``mapping``.
            Labels are never dropped silently, so a new source label stops
            the run. The message names every missing label.
    """
    found = Counter((r.ref.dataset, r.ref.split, b.label) for r in records for b in r.boxes)
    missing = sorted({(d, label) for d, _, label in found if label not in mapping})
    if missing:
        names = ", ".join(f"{d}: {label!r}" for d, label in missing)
        raise UnmappedLabelError(f"labels missing from the class map: {names}")
    mapped = [
        replace(
            r,
            boxes=tuple(
                replace(b, label=mapping[b.label]) for b in r.boxes if mapping[b.label] != DROP
            ),
        )
        for r in records
    ]
    counts = [
        LabelCount(dataset, split, label, mapping[label], n)
        for (dataset, split, label), n in sorted(found.items())
    ]
    return mapped, counts


def to_coco(records: Sequence[ImageRecord], classes: Sequence[str]) -> dict[str, Any]:
    """Build a COCO file with exactly one category per class, ids from 1.

    supervision orders classes by category id, so leaving out Roboflow's
    placeholder category keeps class ids the same in every dataset. Each
    image keeps its source name and original split under ``extra``. The
    empty ``info`` and ``licenses`` are there because the Roboflow SDK's
    uploader reads both keys and fails without them.
    """
    category_ids = {name: n for n, name in enumerate(classes, start=1)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for image_id, r in enumerate(sorted(records, key=lambda r: r.ref.file_name)):
        images.append(
            {
                "id": image_id,
                "file_name": r.ref.file_name,
                "width": r.width,
                "height": r.height,
                "extra": {"name": r.source_name, "source_split": r.ref.split},
            }
        )
        for b in r.boxes:
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": image_id,
                    "category_id": category_ids[b.label],
                    "bbox": [b.x, b.y, b.w, b.h],
                    "area": b.w * b.h,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
    return {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": n, "name": name, "supercategory": "none"} for name, n in category_ids.items()
        ],
        "images": images,
        "annotations": annotations,
    }


def class_coverage(
    splits: Mapping[str, Mapping[str, Sequence[ImageRecord]]], classes: Sequence[str]
) -> dict[str, Any]:
    """Return, per dataset, the classes it labels and their counts per split.

    A class with no boxes anywhere in a dataset is left out of ``labeled``,
    so evaluation can skip it instead of counting every detection of it as
    a false positive.
    """
    coverage: dict[str, Any] = {}
    for dataset, by_split in splits.items():
        per_split: dict[str, Any] = {}
        totals: Counter[str] = Counter()
        for split, records in by_split.items():
            instances = Counter(b.label for r in records for b in r.boxes)
            images = Counter(label for r in records for label in {b.label for b in r.boxes})
            totals.update(instances)
            per_split[split] = {
                "images": len(records),
                "classes": {c: {"instances": instances[c], "images": images[c]} for c in classes},
            }
        coverage[dataset] = {
            "labeled": [c for c in classes if totals[c]],
            "splits": per_split,
        }
    return coverage


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst``, copying instead across file systems."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(
    records: Sequence[ImageRecord],
    source_dirs: Mapping[str, Path],
    out_dir: Path,
    classes: Sequence[str],
) -> None:
    """Write one split's images and COCO file to ``out_dir``.

    ``source_dirs`` maps each original split name to the directory holding
    its images.

    Raises:
        ValueError: If two images in the split share a file name.
    """
    names = Counter(r.ref.file_name for r in records)
    if clashes := sorted(n for n, k in names.items() if k > 1):
        raise ValueError(f"{out_dir}: file names used more than once: {clashes[:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        link_or_copy(source_dirs[r.ref.split] / r.ref.file_name, out_dir / r.ref.file_name)
    (out_dir / ANNOTATIONS_NAME).write_text(
        json.dumps(to_coco(records, classes), indent=1) + "\n", encoding="utf-8"
    )


def _log_counts(counts: Iterable[LabelCount]) -> None:
    for c in counts:
        logger.info("%s/%s: %d %r -> %s", c.dataset, c.split, c.instances, c.source_label, c.target)


def hash_images(
    records: Sequence[ImageRecord], source_dirs: Mapping[str, Path]
) -> dict[ImageRef, int]:
    """Return the perceptual hash of every image, keyed by where it came from."""
    paths = [source_dirs[r.ref.split] / r.ref.file_name for r in records]
    with ThreadPoolExecutor() as pool:
        return dict(zip((r.ref for r in records), pool.map(phash, paths), strict=True))


def split_summary(
    records: Sequence[ImageRecord], result: SplitResult, method: SplitMethod
) -> dict[str, Any]:
    """Summarize one re-split dataset for ``reports/splits.json``.

    A unit is a recording for the temporal method and a group of related
    images for the grouped one, so ``units`` per split counts the
    independent scenes behind each split's images.
    """
    images: Counter[str] = Counter(result.split.values())
    units: dict[str, set[str]] = defaultdict(set)
    for ref, s in result.split.items():
        units[s].add(result.unit[ref])
    unit_sizes = Counter(result.unit[ref] for ref in result.split)
    from_source: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        from_source[r.ref.split][result.split.get(r.ref, "dropped")] += 1
    dropped = Counter(result.dropped.values())
    summary: dict[str, Any] = {
        "method": method.method,
        "unit": "recording" if method.method == "temporal" else "group",
        "units": len(unit_sizes),
        "largest_unit": max(unit_sizes.values(), default=0),
        "dropped": {reason: dropped[reason] for reason in DROP_REASONS},
        "splits": {s: {"images": images[s], "units": len(units[s])} for s in SPLITS},
        "from_source_split": {
            src: dict(sorted(counts.items())) for src, counts in sorted(from_source.items())
        },
    }
    if method.dedupe_copies:
        sizes = Counter(len(g) for g in copy_groups(records).values())
        summary["copy_group_sizes"] = {str(k): n for k, n in sorted(sizes.items())}
    return summary


def cross_dataset_pairs(hashes: Mapping[ImageRef, int], max_distance: int) -> int:
    """Count near-duplicate pairs whose two images come from different datasets."""
    refs = list(hashes)
    values = np.array([hashes[r] for r in refs], dtype=np.uint64)
    return sum(
        refs[i].dataset != refs[j].dataset for i, j in near_duplicate_pairs(values, max_distance)
    )


def run_harmonize(keys: Sequence[str], project: ProjectConfig, class_map: ClassMapConfig) -> None:
    """Harmonize every downloaded dataset in ``keys`` and write the reports.

    Each dataset's output directory is replaced as a whole, so files from an
    earlier run never linger. A dataset without a manifest has not finished
    downloading and is skipped with a warning.

    Raises:
        UnmappedLabelError: If a source label is missing from the class map.
        SplitLeakError: If a re-split test image has a near duplicate in its
            own train or valid split, or if two re-split datasets share a
            near duplicate. Either would let a test image be seen in
            training.
    """
    paths = project.paths
    max_distance = project.inspect.near_duplicate_max_distance
    counts: list[LabelCount] = []
    written: dict[str, dict[str, list[ImageRecord]]] = {}
    summaries: dict[str, Any] = {}
    kept_hashes: dict[ImageRef, int] = {}
    for key in keys:
        dataset_dir = paths.raw_dir / key
        if read_manifest(dataset_dir) is None:
            logger.warning("%s: no manifest, run frc-download first", key)
            continue
        source_dirs = find_splits(dataset_dir)
        records = [r for d in source_dirs.values() for r in load_split(d, key)]
        mapped, dataset_counts = map_labels(records, class_map.datasets.get(key, {}))
        counts.extend(dataset_counts)
        _log_counts(dataset_counts)

        by_split: dict[str, list[ImageRecord]] = defaultdict(list)
        method = project.splits.datasets.get(key)
        if method is None:
            for r in mapped:
                by_split[r.ref.split].append(r)
        else:
            hashes = hash_images(mapped, source_dirs)
            seed = f"{project.seed}:{key}"
            result = split_dataset(mapped, hashes, method, project.splits, max_distance, seed)
            for r in mapped:
                if r.ref in result.split:
                    by_split[result.split[r.ref]].append(r)
            # Checked on the layout about to be written, not on the split
            # result, so a mistake between the two is caught as well.
            layout = {r.ref: s for s, split_records in by_split.items() for r in split_records}
            check_no_leak(layout, hashes, max_distance)
            kept_hashes.update({ref: hashes[ref] for ref in result.split})
            summaries[key] = split_summary(mapped, result, method)
            logger.info(
                "%s: re-split %s, dropped %s", key, method.method, summaries[key]["dropped"]
            )
        out = paths.harmonized_dir / key
        if out.exists():
            shutil.rmtree(out)
        for split, split_records in sorted(by_split.items()):
            write_split(split_records, source_dirs, out / split, class_map.classes)
            logger.info("%s/%s: wrote %d images", key, split, len(split_records))
        written[key] = dict(sorted(by_split.items()))

    cross = cross_dataset_pairs(kept_hashes, max_distance)
    if cross:
        raise SplitLeakError(f"{cross} near-duplicate pairs across re-split datasets")
    report = {
        "fractions": project.splits.fractions.model_dump(),
        "buffer_frames": project.splits.buffer_frames,
        "min_recording_images": project.splits.min_recording_images,
        "near_duplicate_max_distance": max_distance,
        "cross_dataset_near_duplicates": cross,
        "datasets": summaries,
    }
    write_csv(paths.reports_dir / COUNTS_NAME, counts)
    (paths.reports_dir / COVERAGE_NAME).write_text(
        json.dumps(class_coverage(written, class_map.classes), indent=2) + "\n",
        encoding="utf-8",
    )
    (paths.reports_dir / SPLITS_NAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the harmonize command line and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    parser.add_argument("--datasets-config", type=Path, default=DATASETS_CONFIG)
    parser.add_argument("--class-map", type=Path, default=CLASS_MAP_CONFIG)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    project = load_yaml(args.project_config, ProjectConfig)
    datasets = load_yaml(args.datasets_config, DatasetsConfig)
    class_map = load_yaml(args.class_map, ClassMapConfig)
    run_harmonize(list(datasets.datasets), project, class_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
