"""Measure each downloaded dataset and find duplicate images across all of them.

Writes per-split stats and class counts to ``reports/``, exact and
perceptual-hash duplicates to ``reports/duplicates.json``, and one sample
grid per dataset to ``docs/assets/samples_<key>.png``. With
``--contact-sheet``, writes only larger annotated sheets to
``data/contact_sheets/`` for checking labels by eye.
"""

import argparse
import csv
import io
import json
import logging
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import imagehash
import numpy as np
import numpy.typing as npt
import supervision as sv
from PIL import Image, ImageDraw

from frc_xdata.config import (
    AreaBuckets,
    DatasetsConfig,
    GridConfig,
    InspectConfig,
    ProjectConfig,
    TileLayout,
    load_yaml,
)
from frc_xdata.download import (
    DATASETS_CONFIG,
    IMAGE_SUFFIXES,
    PROJECT_CONFIG,
    read_manifest,
)
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.provenance import sha256_file

ANNOTATIONS_NAME = "_annotations.coco.json"
STATS_NAME = "dataset_stats.csv"
CLASS_COUNTS_NAME = "class_counts.csv"
DUPLICATES_NAME = "duplicates.json"
FIELD_TEST_KEY = "field_test"
DIGITS = 4
MARGIN_PX = 4
BACKGROUND = (217, 217, 217)

logger = logging.getLogger(__name__)

PairKind = Literal["within_split", "cross_split", "cross_dataset"]


@dataclass(frozen=True)
class Box:
    """One labeled box in COCO ``[x, y, width, height]`` pixel form."""

    label: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class ImageRef:
    """Where an image lives: dataset key, split, and file name."""

    dataset: str
    split: str
    file_name: str

    def __str__(self) -> str:
        """Return the path relative to the raw data directory."""
        return f"{self.dataset}/{self.split}/{self.file_name}"


@dataclass(frozen=True)
class ImageRecord:
    """One image and its labels, as read from a COCO file."""

    ref: ImageRef
    source_name: str
    width: int
    height: int
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class SplitStats:
    """Summary of one split, written as one row of the stats CSV."""

    dataset: str
    split: str
    images: int
    source_images: int
    boxes: int
    zero_label_images: int
    boxes_per_image_mean: float
    boxes_per_image_median: float
    boxes_per_image_max: int
    resolutions: int
    top_resolution: str
    top_resolution_images: int
    width_min: int
    width_max: int
    height_min: int
    height_max: int
    small_box_fraction: float
    medium_box_fraction: float
    large_box_fraction: float
    invalid_boxes: int


@dataclass(frozen=True)
class ClassCount:
    """Instances of one class in one split, and how many images contain it."""

    dataset: str
    split: str
    label: str
    instances: int
    images: int


def parse_coco(data: Any, dataset: str, split: str) -> list[ImageRecord]:
    """Turn a COCO annotation file's contents into image records.

    Class names come from the categories that annotations point to, so the
    unused placeholder category Roboflow adds at id 0 never shows up. The
    source name is Roboflow's record of the original upload, which its
    augmented copies share.

    Raises:
        ValueError: If an annotation points to a category or image that the
            file does not define.
    """
    names = {c["id"]: c["name"] for c in data["categories"]}
    boxes: dict[int, list[Box]] = defaultdict(list)
    image_ids = {img["id"] for img in data["images"]}
    for ann in data["annotations"]:
        if ann["category_id"] not in names:
            raise ValueError(f"annotation {ann['id']} has unknown category {ann['category_id']}")
        if ann["image_id"] not in image_ids:
            raise ValueError(f"annotation {ann['id']} has unknown image {ann['image_id']}")
        x, y, w, h = (float(v) for v in ann["bbox"])
        boxes[ann["image_id"]].append(Box(names[ann["category_id"]], x, y, w, h))
    return [
        ImageRecord(
            ref=ImageRef(dataset, split, img["file_name"]),
            source_name=(img.get("extra") or {}).get("name", img["file_name"]),
            width=int(img["width"]),
            height=int(img["height"]),
            boxes=tuple(boxes[img["id"]]),
        )
        for img in data["images"]
    ]


def load_split(split_dir: Path, dataset: str) -> list[ImageRecord]:
    """Read the COCO annotation file in ``split_dir``."""
    data = json.loads((split_dir / ANNOTATIONS_NAME).read_text(encoding="utf-8"))
    return parse_coco(data, dataset, split_dir.name)


def area_bucket(box: Box, width: int, height: int, buckets: AreaBuckets) -> str:
    """Return small, medium, or large for a box's share of the image area."""
    fraction = box.w * box.h / (width * height)
    if fraction < buckets.small_max:
        return "small"
    if fraction < buckets.medium_max:
        return "medium"
    return "large"


def is_invalid(box: Box, width: int, height: int, tolerance: float) -> bool:
    """Return True if a box has no area or reaches past the image edge."""
    return (
        box.w <= 0
        or box.h <= 0
        or box.x < -tolerance
        or box.y < -tolerance
        or box.x + box.w > width + tolerance
        or box.y + box.h > height + tolerance
    )


def split_stats(records: Sequence[ImageRecord], cfg: InspectConfig) -> SplitStats:
    """Summarize one split.

    Raises:
        ValueError: If ``records`` is empty or mixes datasets or splits.
    """
    if not records:
        raise ValueError("cannot summarize an empty split")
    refs = {(r.ref.dataset, r.ref.split) for r in records}
    if len(refs) != 1:
        raise ValueError(f"records mix splits: {sorted(refs)}")
    dataset, split = refs.pop()
    per_image = [len(r.boxes) for r in records]
    sizes = Counter(f"{r.width}x{r.height}" for r in records)
    top, top_count = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    buckets: Counter[str] = Counter()
    invalid = 0
    for r in records:
        for b in r.boxes:
            buckets[area_bucket(b, r.width, r.height, cfg.area_buckets)] += 1
            invalid += is_invalid(b, r.width, r.height, cfg.box_tolerance_px)
    total = sum(per_image)

    def share(name: str) -> float:
        return round(buckets[name] / total, DIGITS) if total else 0.0

    return SplitStats(
        dataset=dataset,
        split=split,
        images=len(records),
        source_images=len({r.source_name for r in records}),
        boxes=total,
        zero_label_images=per_image.count(0),
        boxes_per_image_mean=round(statistics.fmean(per_image), DIGITS),
        boxes_per_image_median=float(statistics.median(per_image)),
        boxes_per_image_max=max(per_image),
        resolutions=len(sizes),
        top_resolution=top,
        top_resolution_images=top_count,
        width_min=min(r.width for r in records),
        width_max=max(r.width for r in records),
        height_min=min(r.height for r in records),
        height_max=max(r.height for r in records),
        small_box_fraction=share("small"),
        medium_box_fraction=share("medium"),
        large_box_fraction=share("large"),
        invalid_boxes=invalid,
    )


def class_counts(records: Iterable[ImageRecord]) -> list[ClassCount]:
    """Count instances and images per class for each dataset and split, sorted."""
    instances: Counter[tuple[str, str, str]] = Counter()
    images: Counter[tuple[str, str, str]] = Counter()
    for r in records:
        labels = [b.label for b in r.boxes]
        for label in labels:
            instances[(r.ref.dataset, r.ref.split, label)] += 1
        for label in set(labels):
            images[(r.ref.dataset, r.ref.split, label)] += 1
    return [
        ClassCount(*key, instances=instances[key], images=images[key]) for key in sorted(instances)
    ]


def source_name_overlap(records: Iterable[ImageRecord]) -> dict[str, list[str]]:
    """Return, per dataset, the source names that appear in more than one split.

    Augmented copies of one upload share a source name, so an overlap means
    the same original photo sits in two splits.
    """
    splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in records:
        splits[(r.ref.dataset, r.source_name)].add(r.ref.split)
    overlap: dict[str, list[str]] = defaultdict(list)
    for (dataset, name), found in sorted(splits.items()):
        if len(found) > 1:
            overlap[dataset].append(name)
    return dict(overlap)


def pair_kind(a: ImageRef, b: ImageRef) -> PairKind:
    """Classify a duplicate pair by whether it crosses a split or a dataset."""
    if a.dataset != b.dataset:
        return "cross_dataset"
    if a.split != b.split:
        return "cross_split"
    return "within_split"


def exact_duplicate_groups(digests: Sequence[tuple[ImageRef, str]]) -> list[list[ImageRef]]:
    """Group images with the same SHA256, keeping only groups of two or more."""
    groups: dict[str, list[ImageRef]] = defaultdict(list)
    for ref, digest in digests:
        groups[digest].append(ref)
    return sorted(
        (sorted(g, key=str) for g in groups.values() if len(g) > 1),
        key=lambda g: str(g[0]),
    )


def near_duplicate_pairs(
    hashes: npt.NDArray[np.uint64], max_distance: int
) -> list[tuple[int, int]]:
    """Return index pairs ``(i, j)``, ``i < j``, within ``max_distance`` bits.

    Each row compares one hash against every later one in a single vector
    operation instead of a Python loop.
    """
    pairs = []
    for i in range(len(hashes) - 1):
        distances = np.bitwise_count(hashes[i] ^ hashes[i + 1 :])
        for offset in np.flatnonzero(distances <= max_distance):
            pairs.append((i, i + 1 + int(offset)))
    return pairs


def phash(path: Path) -> int:
    """Return the 64-bit perceptual hash of an image as an integer."""
    with Image.open(path) as image:
        bits = imagehash.phash(image).hash.flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def summarize_pairs(
    pairs: Iterable[tuple[ImageRef, ImageRef]], max_examples: int
) -> dict[str, Any]:
    """Count duplicate pairs by kind and by the datasets and splits involved.

    Returns:
        ``counts`` maps each kind to a list of rows naming both sides and the
        number of pairs. ``examples`` keeps up to ``max_examples`` pairs per
        kind, in the order they were found.
    """
    counts: Counter[tuple[PairKind, str, str]] = Counter()
    examples: dict[str, list[list[str]]] = defaultdict(list)
    for a, b in pairs:
        a, b = sorted((a, b), key=str)
        kind = pair_kind(a, b)
        counts[(kind, f"{a.dataset}/{a.split}", f"{b.dataset}/{b.split}")] += 1
        if len(examples[kind]) < max_examples:
            examples[kind].append([str(a), str(b)])
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (kind, first, second), n in sorted(counts.items()):
        rows[kind].append({"a": first, "b": second, "pairs": n})
    return {"counts": dict(rows), "examples": dict(examples)}


def group_pairs(groups: Iterable[Sequence[ImageRef]]) -> list[tuple[ImageRef, ImageRef]]:
    """Expand each group of identical images into every pair within it."""
    return [(g[i], g[j]) for g in groups for i in range(len(g)) for j in range(i + 1, len(g))]


def _fit(
    image: Image.Image, boxes: Sequence[Box], tile_px: int
) -> tuple[Image.Image, sv.Detections, list[str]]:
    scale = tile_px / max(image.size)
    resized = image.resize((round(image.width * scale), round(image.height * scale)))
    labels = sorted({b.label for b in boxes})
    xyxy = np.array(
        [[b.x * scale, b.y * scale, (b.x + b.w) * scale, (b.y + b.h) * scale] for b in boxes],
        dtype=float,
    ).reshape(-1, 4)
    class_id = np.array([labels.index(b.label) for b in boxes], dtype=int)
    return resized, sv.Detections(xyxy=xyxy, class_id=class_id), [b.label for b in boxes]


def grid_order(records: Sequence[ImageRecord], grid: GridConfig, seed: str) -> list[ImageRecord]:
    """Return the images a sample grid shows, row by row.

    Excluding a file swaps in the next image in the seeded order, so the
    rest of the grid stays the same.
    """
    excluded = set(grid.exclude.get(records[0].ref.dataset, [])) if records else set()
    order = sorted(records, key=lambda r: str(r.ref))
    random.Random(seed).shuffle(order)
    return [r for r in order if r.ref.file_name not in excluded][: grid.rows * grid.cols]


def sample_grid(
    records: Sequence[ImageRecord],
    split_dirs: dict[str, Path],
    grid: GridConfig,
    seed: str,
) -> Image.Image:
    """Draw the images from ``grid_order`` with their boxes and labels."""
    chosen = grid_order(records, grid, seed)
    return tile(_annotate(chosen, split_dirs, grid.tile_px), [r.ref.split for r in chosen], grid)


def _annotate(
    records: Sequence[ImageRecord], split_dirs: dict[str, Path], tile_px: int
) -> list[Image.Image]:
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.35, text_padding=2)
    tiles = []
    for r in records:
        with Image.open(split_dirs[r.ref.split] / r.ref.file_name) as opened:
            image, detections, labels = _fit(opened.convert("RGB"), r.boxes, tile_px)
        image = box_annotator.annotate(image, detections)
        tiles.append(label_annotator.annotate(image, detections, labels=labels))
    return tiles


def _one_per_source(records: Iterable[ImageRecord], limit: int) -> list[ImageRecord]:
    # Augmented copies share a source name. Otherwise one scene can fill a sheet.
    seen: set[str] = set()
    kept = []
    for r in records:
        if r.source_name not in seen:
            seen.add(r.source_name)
            kept.append(r)
    return kept[:limit]


def contact_sheet_picks(
    records: Sequence[ImageRecord], layout: TileLayout, seed: str
) -> dict[str, list[ImageRecord]]:
    """Return the images for each of one dataset's contact sheets, by sheet name.

    ``dense`` holds the images with the most boxes, ``zero`` a seeded sample
    of images with no boxes, and ``label_<name>`` a seeded sample of images
    containing that label. Sheets with no images are left out.
    """
    limit = layout.rows * layout.cols
    ordered = sorted(records, key=lambda r: str(r.ref))
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    sheets = {
        "dense": _one_per_source(
            sorted((r for r in ordered if r.boxes), key=lambda r: -len(r.boxes)), limit
        ),
        "zero": _one_per_source((r for r in shuffled if not r.boxes), limit),
    }
    for label in sorted({b.label for r in records for b in r.boxes}):
        name = "label_" + re.sub(r"[^A-Za-z0-9_-]", "_", label)
        sheets[name] = _one_per_source(
            (r for r in shuffled if any(b.label == label for b in r.boxes)), limit
        )
    return {name: picks for name, picks in sheets.items() if picks}


def tile(images: Sequence[Image.Image], captions: Sequence[str], grid: TileLayout) -> Image.Image:
    """Lay images out row by row, each centered in a square cell with a caption.

    The grid always has ``rows * cols`` cells, so every dataset's grid has the
    same size even when it has fewer images than cells.
    """
    step = grid.tile_px + MARGIN_PX
    mosaic = Image.new(
        "RGB", (grid.cols * step + MARGIN_PX, grid.rows * step + MARGIN_PX), BACKGROUND
    )
    draw = ImageDraw.Draw(mosaic)
    for n, (image, caption) in enumerate(zip(images, captions, strict=True)):
        row, col = divmod(n, grid.cols)
        left, top = MARGIN_PX + col * step, MARGIN_PX + row * step
        mosaic.paste(
            image,
            (left + (grid.tile_px - image.width) // 2, top + (grid.tile_px - image.height) // 2),
        )
        draw.text((left + 2, top + 2), caption, fill=(0, 0, 0))
    return mosaic


def encode_png(image: Image.Image, max_bytes: int) -> bytes:
    """Encode ``image`` as a 256-color PNG, shrinking it until it fits.

    Raises:
        ValueError: If even a tiny version does not fit in ``max_bytes``.
    """
    while True:
        buffer = io.BytesIO()
        image.quantize(colors=256).save(buffer, format="PNG", optimize=True)
        if buffer.tell() <= max_bytes:
            return buffer.getvalue()
        if min(image.size) < 64:
            raise ValueError(f"cannot fit the grid in {max_bytes} bytes")
        image = image.resize((image.width * 9 // 10, image.height * 9 // 10))


def _grid_seed(project: ProjectConfig, key: str) -> str:
    # One seed per dataset, so adding a dataset leaves the other grids alone.
    return f"{project.seed}:{key}"


def find_splits(dataset_dir: Path) -> dict[str, Path]:
    """Return each split directory that holds a COCO file, by split name."""
    return {
        d.name: d
        for d in sorted(dataset_dir.iterdir())
        if d.is_dir() and (d / ANNOTATIONS_NAME).is_file()
    }


def field_test_images(field_test_dir: Path) -> list[tuple[ImageRef, Path]]:
    """Return every field-test image under ``field_test_dir``, or none if it is missing."""
    if not field_test_dir.is_dir():
        return []
    return [
        (ImageRef(FIELD_TEST_KEY, p.parent.relative_to(field_test_dir).as_posix(), p.name), p)
        for p in sorted(field_test_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


def write_csv(path: Path, rows: Sequence[Any]) -> None:
    """Write dataclass rows to a CSV file with a header from their fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([fld.name for fld in fields(rows[0])] if rows else [])
        writer.writerows([list(asdict(r).values()) for r in rows])


def run_inspection(
    keys: Sequence[str],
    project: ProjectConfig,
    grids: bool = True,
) -> None:
    """Inspect every downloaded dataset in ``keys`` and write the reports.

    A dataset without a manifest has not finished downloading and is
    skipped with a warning. Field-test images are included in the duplicate
    search only, since they are never shown or summarized with training data.
    """
    paths, cfg = project.paths, project.inspect
    records: list[ImageRecord] = []
    stats: list[SplitStats] = []
    files: list[tuple[ImageRef, Path]] = []
    digests: list[tuple[ImageRef, str]] = []
    for key in keys:
        dataset_dir = paths.raw_dir / key
        manifest = read_manifest(dataset_dir)
        if manifest is None:
            logger.warning("%s: no manifest, run frc-download first", key)
            continue
        sha = {entry.path: entry.sha256 for entry in manifest.files}
        split_dirs = find_splits(dataset_dir)
        dataset_records = []
        for split, split_dir in split_dirs.items():
            split_records = load_split(split_dir, key)
            if not split_records:
                logger.warning("%s/%s: no images", key, split)
                continue
            stats.append(split_stats(split_records, cfg))
            dataset_records.extend(split_records)
            for r in split_records:
                files.append((r.ref, split_dir / r.ref.file_name))
                digests.append((r.ref, sha[f"{split}/{r.ref.file_name}"]))
        records.extend(dataset_records)
        logger.info("%s: %d images in %s", key, len(dataset_records), sorted(split_dirs))
        if grids and dataset_records:
            png = encode_png(
                sample_grid(dataset_records, split_dirs, cfg.grid, _grid_seed(project, key)),
                cfg.grid.max_bytes,
            )
            paths.assets_dir.mkdir(parents=True, exist_ok=True)
            (paths.assets_dir / f"samples_{key}.png").write_bytes(png)

    field = field_test_images(paths.field_test_dir)
    files.extend(field)
    digests.extend((ref, sha256_file(path)) for ref, path in field)

    logger.info("hashing %d images for near duplicates", len(files))
    with ThreadPoolExecutor() as pool:
        hashes = np.array(list(pool.map(phash, [p for _, p in files])), dtype=np.uint64)
    refs = [ref for ref, _ in files]
    near = [
        (refs[i], refs[j]) for i, j in near_duplicate_pairs(hashes, cfg.near_duplicate_max_distance)
    ]
    exact = exact_duplicate_groups(digests)
    report = {
        "near_duplicate_max_distance": cfg.near_duplicate_max_distance,
        "field_test_images": len(field),
        "exact": {"groups": len(exact), **summarize_pairs(group_pairs(exact), cfg.max_examples)},
        "near": {"pairs": len(near), **summarize_pairs(near, cfg.max_examples)},
        "source_name_overlap": {
            dataset: {"names": len(names), "examples": names[: cfg.max_examples]}
            for dataset, names in source_name_overlap(records).items()
        },
    }
    write_csv(paths.reports_dir / STATS_NAME, stats)
    write_csv(paths.reports_dir / CLASS_COUNTS_NAME, class_counts(records))
    (paths.reports_dir / DUPLICATES_NAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("%d exact groups, %d near-duplicate pairs", len(exact), len(near))


def list_grids(keys: Sequence[str], project: ProjectConfig) -> None:
    """Log the file behind each grid tile, for checking the grids by eye.

    Positions read ``r<row>c<col>``, counted from 1 at the top left.
    """
    grid = project.inspect.grid
    for key in keys:
        dataset_dir = project.paths.raw_dir / key
        if read_manifest(dataset_dir) is None:
            continue
        records = [r for d in find_splits(dataset_dir).values() for r in load_split(d, key)]
        for n, r in enumerate(grid_order(records, grid, _grid_seed(project, key))):
            row, col = divmod(n, grid.cols)
            logger.info("%s r%dc%d %s/%s", key, row + 1, col + 1, r.ref.split, r.ref.file_name)


def write_contact_sheets(keys: Sequence[str], project: ProjectConfig) -> None:
    """Write each dataset's contact sheets and log the file behind every tile.

    Positions read ``r<row>c<col>``, counted from 1 at the top left. Sheets are
    not face-checked, so they stay out of ``docs/``.
    """
    layout = project.inspect.contact_sheet
    out_dir = project.paths.contact_sheet_dir
    for key in keys:
        dataset_dir = project.paths.raw_dir / key
        if read_manifest(dataset_dir) is None:
            logger.warning("%s: no manifest, run frc-download first", key)
            continue
        split_dirs = find_splits(dataset_dir)
        records = [r for d in split_dirs.values() for r in load_split(d, key)]
        for name, picks in contact_sheet_picks(records, layout, _grid_seed(project, key)).items():
            sheet = tile(
                _annotate(picks, split_dirs, layout.tile_px), [r.ref.split for r in picks], layout
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            sheet.save(out_dir / f"{key}_{name}.jpg")
            for n, r in enumerate(picks):
                row, col = divmod(n, layout.cols)
                logger.info(
                    "%s %s r%dc%d %s/%s", key, name, row + 1, col + 1, r.ref.split, r.ref.file_name
                )
    logger.info("contact sheets in %s are not face-checked, keep them out of docs/", out_dir)


def main(argv: list[str] | None = None) -> int:
    """Run the inspection command line and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    parser.add_argument("--datasets-config", type=Path, default=DATASETS_CONFIG)
    parser.add_argument("--no-grids", action="store_true", help="skip the sample grids")
    parser.add_argument(
        "--list-grid", action="store_true", help="log the file behind each grid tile, write nothing"
    )
    parser.add_argument(
        "--contact-sheet",
        nargs="+",
        metavar="KEY",
        help="write label-checking sheets for these datasets instead of the reports",
    )
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    project = load_yaml(args.project_config, ProjectConfig)
    datasets = load_yaml(args.datasets_config, DatasetsConfig)
    if args.list_grid:
        list_grids(list(datasets.datasets), project)
        return 0
    if args.contact_sheet:
        unknown = sorted(set(args.contact_sheet) - set(datasets.datasets))
        if unknown:
            parser.error(f"unknown dataset keys: {', '.join(unknown)}")
        write_contact_sheets(args.contact_sheet, project)
        return 0
    run_inspection(list(datasets.datasets), project, grids=not args.no_grids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
