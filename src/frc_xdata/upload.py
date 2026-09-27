"""Upload a harmonized dataset to Roboflow for training, and check the upload.

``frc-upload`` first checks that each harmonized split loads with supervision,
holds as many images as ``reports/splits.json`` records, and has no near
duplicate in the field test set. It then uploads each split in its own call
with the split named. Left to itself, the SDK guesses each image's split from
its path and puts anything it cannot place in train. It uploads only to a
project that already exists, because the SDK would otherwise create one under
an MIT license.

``frc-verify-upload`` downloads a generated version in COCO format and matches
every image to its harmonized file by the file name recorded at upload. It
writes ``reports/platform_upload_<key>.json``, with the preprocessing and
augmentation Roboflow reports for the version, and fails if an image is
missing, extra, in another split, or has a different number of boxes.
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import supervision as sv

from frc_xdata.config import ProjectConfig, load_yaml
from frc_xdata.download import PROJECT_CONFIG, redact, redacted_api_key, sdk_errors
from frc_xdata.errors import ConfigError, DataLeakError, DirtyTreeError, UploadCheckError
from frc_xdata.harmonize import SPLITS_NAME
from frc_xdata.inspect_datasets import (
    ANNOTATIONS_NAME,
    ImageRecord,
    field_test_images,
    find_splits,
    load_split,
    phash,
)
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.provenance import git_commit, git_is_dirty, utc_timestamp
from frc_xdata.splits import SPLITS

REPORT_NAME = "platform_upload_{key}.json"
EXPORT_FORMAT = "coco"

logger = logging.getLogger(__name__)

Upload = Callable[[Path, str], None]


@dataclass(frozen=True)
class VersionInfo:
    """What Roboflow reports about a generated dataset version."""

    splits: dict[str, Any]
    preprocessing: dict[str, Any]
    augmentation: dict[str, Any]


Export = Callable[[Path], VersionInfo]


@dataclass(frozen=True)
class UploadComparison:
    """How a downloaded version differs from the harmonized dataset.

    Images are matched by the name Roboflow recorded at upload. Entries in
    ``moved`` are ``(file name, harmonized split, exported split)``. A file
    exported more than once is listed in ``extra`` once per copy after the
    first.
    """

    expected: dict[str, int]
    exported: dict[str, int]
    missing: list[str]
    extra: list[str]
    moved: list[tuple[str, str, str]]
    box_count_changed: list[str]

    @property
    def ok(self) -> bool:
        """Return True if every image arrived once, in its split, with its boxes."""
        return not (self.missing or self.extra or self.moved or self.box_count_changed)


def split_sizes(dataset_dir: Path) -> dict[str, int]:
    """Load each split with supervision and return its number of images.

    Raises:
        UploadCheckError: If a split has no COCO file.
    """
    sizes: dict[str, int] = {}
    for split in SPLITS:
        coco = dataset_dir / split / ANNOTATIONS_NAME
        if not coco.is_file():
            raise UploadCheckError(f"{coco} is missing, run make harmonize first")
        dataset = sv.DetectionDataset.from_coco(
            images_directory_path=str(coco.parent), annotations_path=str(coco)
        )
        sizes[split] = len(dataset)
    return sizes


def reported_sizes(splits_report: Path, key: str) -> dict[str, int]:
    """Return the number of images per split that the split report records for ``key``.

    Raises:
        UploadCheckError: If the report has no split sizes for ``key``. Only
            datasets re-split by harmonize have them, and only those carry
            the guarantee that no test image has a near duplicate in train.
    """
    data = json.loads(splits_report.read_text(encoding="utf-8"))
    try:
        splits = data["datasets"][key]["splits"]
        return {split: int(splits[split]["images"]) for split in SPLITS}
    except KeyError as e:
        raise UploadCheckError(f"{splits_report} has no split sizes for {key}") from e


def check_sizes(found: Mapping[str, int], expected: Mapping[str, int]) -> None:
    """Check the harmonized split sizes against the split report.

    Raises:
        UploadCheckError: If any split differs, naming each one.
    """
    wrong = [
        f"{split} has {found.get(split)}, the report says {expected[split]}"
        for split in SPLITS
        if found.get(split) != expected[split]
    ]
    if wrong:
        raise UploadCheckError(
            "harmonized data does not match the split report, run make harmonize: "
            + "; ".join(wrong)
        )


def field_test_leaks(
    images: Sequence[Path], field_test_dir: Path, max_distance: int
) -> list[tuple[Path, Path]]:
    """Return each ``(image, field-test image)`` pair within ``max_distance`` pHash bits."""
    field = [path for _, path in field_test_images(field_test_dir)]
    if not field:
        return []
    with ThreadPoolExecutor() as pool:
        field_hashes = list(pool.map(phash, field))
        image_hashes = list(pool.map(phash, images))
    return [
        (image, test_image)
        for image, h in zip(images, image_hashes, strict=True)
        for test_image, th in zip(field, field_hashes, strict=True)
        if (h ^ th).bit_count() <= max_distance
    ]


def check_before_upload(dataset_dir: Path, key: str, project: ProjectConfig) -> dict[str, int]:
    """Run every local check on a harmonized dataset and return its split sizes.

    Raises:
        UploadCheckError: If a split is missing or its size differs from
            ``reports/splits.json``.
        DataLeakError: If an image is a near duplicate of a field-test image.
    """
    sizes = split_sizes(dataset_dir)
    check_sizes(sizes, reported_sizes(project.paths.reports_dir / SPLITS_NAME, key))
    field_dir = project.paths.field_test_dir
    if not field_dir.is_dir():
        logger.warning("%s does not exist, so there are no field-test images to check", field_dir)
    images = [
        dataset_dir / split / r.ref.file_name
        for split in SPLITS
        for r in load_split(dataset_dir / split, key)
    ]
    leaks = field_test_leaks(images, field_dir, project.inspect.near_duplicate_max_distance)
    if leaks:
        image, test_image = leaks[0]
        raise DataLeakError(
            f"{len(leaks)} image pairs match the field test set, such as {image} and {test_image}"
        )
    return sizes


def upload_splits(dataset_dir: Path, upload: Upload) -> None:
    """Upload each split in its own call, naming the split."""
    for split in SPLITS:
        logger.info("uploading %s as %s", dataset_dir / split, split)
        upload(dataset_dir / split, split)


# Universe exports name files <stem>_<ext>.rf.<hash>.jpg.
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}")
IMAGE_EXTENSION = re.compile(r"\.(?:jpe?g|png)$", re.IGNORECASE)
JPG_TAG = re.compile(r"_jpe?g$", re.IGNORECASE)


def match_key(name: str) -> str:
    """Return the part of an image name that survives a Roboflow upload.

    Roboflow drops the ``.rf.<hash>`` suffix of an uploaded Universe file,
    and a ``_jpg`` tag before it, so ``x_jpg.rf.<hash>.jpg`` is exported
    as ``x.jpg`` and ``x_png.rf.<hash>.jpg`` as ``x_png.jpg``. Both map to
    the same key as the file they came from.
    """
    stem = IMAGE_EXTENSION.sub("", RF_SUFFIX.sub("", name))
    return JPG_TAG.sub("", stem)


def _by_key(records: Sequence[ImageRecord]) -> dict[str, ImageRecord]:
    by_key = {match_key(r.ref.file_name): r for r in records}
    if len(by_key) < len(records):
        raise UploadCheckError(
            "a file name appears in more than one harmonized split, or two names "
            "differ only in the suffix Roboflow drops on upload"
        )
    return by_key


def compare_export(export_dir: Path, dataset_dir: Path, key: str) -> UploadComparison:
    """Match every image in a downloaded version to its harmonized file.

    Missing, moved, and changed images are listed by their harmonized file
    name, and extra ones by their exported name.

    Raises:
        UploadCheckError: If two harmonized images share a match key, which
            would make the match ambiguous.
    """
    expected = _by_key([r for split in SPLITS for r in load_split(dataset_dir / split, key)])
    exported = sorted(
        (r for d in find_splits(export_dir).values() for r in load_split(d, key)),
        key=lambda r: r.source_name,
    )
    seen: set[str] = set()
    extra = []
    moved = []
    box_count_changed = []
    for r in exported:
        k = match_key(r.source_name)
        original = expected.get(k)
        if original is None or k in seen:
            extra.append(r.source_name)
            continue
        seen.add(k)
        name = original.ref.file_name
        if original.ref.split != r.ref.split:
            moved.append((name, original.ref.split, r.ref.split))
        if len(original.boxes) != len(r.boxes):
            box_count_changed.append(name)
    exported_sizes = Counter(r.ref.split for r in exported)
    return UploadComparison(
        expected=dict(Counter(r.ref.split for r in expected.values())),
        exported={split: exported_sizes[split] for split in SPLITS},
        missing=sorted(r.ref.file_name for k, r in expected.items() if k not in seen),
        extra=extra,
        moved=moved,
        box_count_changed=box_count_changed,
    )


def upload_report(
    comparison: UploadComparison,
    info: VersionInfo,
    key: str,
    project_slug: str,
    version: int,
    max_examples: int,
    repo: Path,
) -> dict[str, Any]:
    """Build the record written to ``reports/platform_upload_<key>.json``.

    Each mismatch list is cut to ``max_examples`` entries next to its full
    count.
    """
    problems: dict[str, Sequence[Any]] = {
        "missing": comparison.missing,
        "extra": comparison.extra,
        "moved": comparison.moved,
        "box_count_changed": comparison.box_count_changed,
    }
    return {
        "key": key,
        "project": project_slug,
        "version": version,
        "checked": utc_timestamp(),
        "git_commit": git_commit(repo),
        "dirty": git_is_dirty(repo),
        "roboflow": package_version("roboflow"),
        "ok": comparison.ok,
        "expected": comparison.expected,
        "exported": comparison.exported,
        "roboflow_splits": info.splits,
        "preprocessing": info.preprocessing,
        "augmentation": info.augmentation,
        **{
            name: {
                "count": len(items),
                "examples": [list(i) if isinstance(i, tuple) else i for i in items[:max_examples]],
            }
            for name, items in problems.items()
        },
    }


def _roboflow_project(api_key: str, slug: str) -> tuple[Any, Any]:
    from roboflow import Roboflow
    from roboflow.adapters.rfapi import RoboflowError

    workspace = Roboflow(api_key=api_key).workspace()
    try:
        return workspace, workspace.project(slug)
    except RoboflowError as e:
        raise ConfigError(
            f"no Roboflow project {slug} in this key's workspace. "
            "Create it in the web app first, see docs/RETRAINING.md"
        ) from e


def _roboflow_upload(api_key: str, slug: str, key: str, retries: int) -> Upload:
    workspace, _ = _roboflow_project(api_key, slug)

    def upload(split_dir: Path, split: str) -> None:
        workspace.upload_dataset(
            str(split_dir),
            slug,
            num_retries=retries,
            batch_name=f"harmonized-{key}-{split}",
            split=split,
        )

    return upload


def _roboflow_export(api_key: str, slug: str, version: int) -> Export:
    _, project = _roboflow_project(api_key, slug)

    def export(dest: Path) -> VersionInfo:
        generated = project.version(version)
        # overwrite=True because the SDK otherwise returns early whenever the
        # directory exists, even if an earlier download stopped halfway.
        generated.download(EXPORT_FORMAT, location=str(dest), overwrite=True)
        return VersionInfo(
            splits=dict(generated.splits),
            preprocessing=dict(generated.preprocessing),
            augmentation=dict(generated.augmentation),
        )

    return export


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("key", help="harmonized dataset key, such as marswars")
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    add_log_level_argument(parser)
    return parser


def upload_main(argv: list[str] | None = None) -> int:
    """Run the upload command line and return the exit code."""
    parser = _parser("Check a harmonized dataset and upload it to Roboflow split by split.")
    parser.add_argument("--dry-run", action="store_true", help="run the local checks only")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    project = load_yaml(args.project_config, ProjectConfig)
    dataset_dir = project.paths.harmonized_dir / args.key
    sizes = check_before_upload(dataset_dir, args.key, project)
    logger.info("%s passed the local checks: %s", args.key, sizes)
    if args.dry_run:
        return 0

    api_key = redacted_api_key()
    slug = project.platform.project
    try:
        upload_splits(
            dataset_dir,
            _roboflow_upload(api_key, slug, args.key, project.platform.upload_retries),
        )
    except sdk_errors() as e:
        logger.error("roboflow failed: %s", redact(f"{type(e).__name__}: {e}", api_key))
        return 1
    logger.info("uploaded %s to %s. Generate a version, then run frc-verify-upload", args.key, slug)
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    """Run the upload check command line and return the exit code."""
    parser = _parser("Download a Roboflow version and check it image by image.")
    parser.add_argument("--version", type=int, required=True, help="generated version number")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="write the report from a dirty tree"
    )
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    repo = Path.cwd()
    if git_is_dirty(repo) and not args.allow_dirty:
        raise DirtyTreeError("commit or stash changes first, or pass --allow-dirty")
    project = load_yaml(args.project_config, ProjectConfig)
    slug = project.platform.project
    api_key = redacted_api_key()
    export_dir = project.paths.platform_dir / f"{slug}-v{args.version}"
    try:
        info = _roboflow_export(api_key, slug, args.version)(export_dir)
    except sdk_errors() as e:
        logger.error("roboflow failed: %s", redact(f"{type(e).__name__}: {e}", api_key))
        return 1

    comparison = compare_export(export_dir, project.paths.harmonized_dir / args.key, args.key)
    report = upload_report(
        comparison, info, args.key, slug, args.version, project.inspect.max_examples, repo
    )
    path = project.paths.reports_dir / REPORT_NAME.format(key=args.key)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "wrote %s. Exported %s, expected %s", path, comparison.exported, comparison.expected
    )
    if not comparison.ok:
        for name in ("missing", "extra", "moved", "box_count_changed"):
            if report[name]["count"]:
                logger.error(
                    "%s: %d images, such as %s",
                    name,
                    report[name]["count"],
                    report[name]["examples"][:3],
                )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(upload_main())
