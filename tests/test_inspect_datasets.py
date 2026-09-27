import csv
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import coco, image_entry, smooth_image
from PIL import Image

from frc_xdata import inspect_datasets
from frc_xdata.config import GridConfig, ProjectConfig, load_yaml
from frc_xdata.inspect_datasets import (
    Box,
    ImageRef,
    area_bucket,
    class_counts,
    encode_png,
    exact_duplicate_groups,
    group_pairs,
    is_invalid,
    load_split,
    near_duplicate_pairs,
    pair_kind,
    parse_coco,
    phash,
    sample_grid,
    source_name_overlap,
    split_stats,
    summarize_pairs,
    tile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project() -> ProjectConfig:
    return load_yaml(REPO_ROOT / "configs" / "project.yaml", ProjectConfig)


def ref(dataset: str, split: str, name: str) -> ImageRef:
    return ImageRef(dataset, split, name)


def test_parse_coco_skips_placeholder_and_falls_back_to_file_name() -> None:
    data = coco(
        [image_entry(0, "x.jpg", source="orig.jpg"), image_entry(1, "y.jpg")],
        [(0, 1, [1, 2, 3, 4])],
    )
    records = parse_coco(data, "ds", "train")
    assert records[0].boxes == (Box("fuel", 1, 2, 3, 4),)
    assert records[0].source_name == "orig.jpg"
    assert records[1].source_name == "y.jpg"
    assert records[1].boxes == ()


@pytest.mark.parametrize(
    ("anns", "message"),
    [([(0, 9, [0, 0, 1, 1])], "unknown category 9"), ([(5, 1, [0, 0, 1, 1])], "unknown image 5")],
)
def test_parse_coco_rejects_dangling_references(
    anns: list[tuple[int, int, list[float]]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_coco(coco([image_entry(0, "x.jpg")], anns), "ds", "train")


def test_split_stats_known_answers(raw_dir: Path, project: ProjectConfig) -> None:
    stats = split_stats(load_split(raw_dir / "alpha" / "train", "alpha"), project.inspect)
    assert (stats.dataset, stats.split) == ("alpha", "train")
    assert (stats.images, stats.source_images, stats.boxes) == (3, 2, 3)
    assert stats.zero_label_images == 1
    assert (stats.boxes_per_image_mean, stats.boxes_per_image_median) == (1.0, 1.0)
    assert stats.boxes_per_image_max == 2
    assert (stats.resolutions, stats.top_resolution, stats.top_resolution_images) == (1, "64x64", 3)
    # fuel 4x4 is 0.0039 of the image (medium), robot 40x40 and fuel 10x10
    # are 0.39 and 0.024 (large).
    assert stats.small_box_fraction == 0.0
    assert stats.medium_box_fraction == 0.3333
    assert stats.large_box_fraction == 0.6667
    assert stats.invalid_boxes == 1


def test_split_stats_rejects_empty_and_mixed(raw_dir: Path, project: ProjectConfig) -> None:
    with pytest.raises(ValueError, match="empty"):
        split_stats([], project.inspect)
    mixed = load_split(raw_dir / "alpha" / "train", "alpha") + load_split(
        raw_dir / "alpha" / "valid", "alpha"
    )
    with pytest.raises(ValueError, match="mix"):
        split_stats(mixed, project.inspect)


def test_area_bucket_edges_belong_to_the_larger_bucket(project: ProjectConfig) -> None:
    buckets = project.inspect.area_buckets
    # A 100x100 image makes the box area in pixels equal to fraction * 10000.
    assert area_bucket(Box("f", 0, 0, 24, 1), 100, 100, buckets) == "small"
    assert area_bucket(Box("f", 0, 0, 25, 1), 100, 100, buckets) == "medium"
    assert area_bucket(Box("f", 0, 0, 224, 1), 100, 100, buckets) == "medium"
    assert area_bucket(Box("f", 0, 0, 225, 1), 100, 100, buckets) == "large"


@pytest.mark.parametrize(
    ("box", "invalid"),
    [
        (Box("f", 0, 0, 10, 10), False),
        (Box("f", -1, -1, 11, 11), False),
        (Box("f", 90, 90, 11, 11), False),
        (Box("f", 0, 0, 0, 10), True),
        (Box("f", 0, 0, 10, -1), True),
        (Box("f", -2, 0, 10, 10), True),
        (Box("f", 0, 90, 10, 12), True),
    ],
)
def test_is_invalid(box: Box, invalid: bool) -> None:
    assert is_invalid(box, 100, 100, tolerance=1) is invalid


def test_class_counts_count_instances_and_images(raw_dir: Path) -> None:
    rows = class_counts(load_split(raw_dir / "alpha" / "train", "alpha"))
    assert [(r.label, r.instances, r.images) for r in rows] == [("fuel", 2, 2), ("robot", 1, 1)]


def test_source_name_overlap_finds_the_same_upload_in_two_splits(raw_dir: Path) -> None:
    records = [
        *load_split(raw_dir / "alpha" / "train", "alpha"),
        *load_split(raw_dir / "alpha" / "valid", "alpha"),
        *load_split(raw_dir / "beta" / "train", "beta"),
    ]
    assert source_name_overlap(records) == {"alpha": ["a.png"]}


def test_pair_kind() -> None:
    a = ref("x", "train", "1.jpg")
    assert pair_kind(a, ref("x", "train", "2.jpg")) == "within_split"
    assert pair_kind(a, ref("x", "valid", "2.jpg")) == "cross_split"
    assert pair_kind(a, ref("y", "train", "2.jpg")) == "cross_dataset"


def test_exact_duplicate_groups_keep_only_repeated_digests() -> None:
    a, b, c = ref("x", "t", "a"), ref("x", "t", "b"), ref("y", "t", "c")
    groups = exact_duplicate_groups([(c, "1"), (a, "1"), (b, "2")])
    assert groups == [[a, c]]
    assert group_pairs([*groups, [a, b, c]]) == [(a, c), (a, b), (a, c), (b, c)]


def test_near_duplicate_pairs_respect_the_distance() -> None:
    hashes = np.array([0b0000, 0b0001, 0b0111, 0xFFFF], dtype=np.uint64)
    assert near_duplicate_pairs(hashes, 1) == [(0, 1)]
    assert near_duplicate_pairs(hashes, 2) == [(0, 1), (1, 2)]
    assert near_duplicate_pairs(hashes, 0) == []
    assert near_duplicate_pairs(np.array([], dtype=np.uint64), 4) == []


def test_near_duplicate_pairs_use_all_64_bits() -> None:
    top = np.uint64(1 << 63)
    assert near_duplicate_pairs(np.array([0, top], dtype=np.uint64), 0) == []
    assert near_duplicate_pairs(np.array([top, top], dtype=np.uint64), 0) == [(0, 1)]


def test_phash_is_close_for_a_resized_copy_and_far_otherwise(tmp_path: Path) -> None:
    smooth_image(1).save(tmp_path / "a.png")
    smooth_image(1).resize((48, 48)).save(tmp_path / "resized.png")
    smooth_image(2).save(tmp_path / "other.png")
    a, resized, other = (phash(tmp_path / f"{n}.png") for n in ("a", "resized", "other"))
    assert (a ^ resized).bit_count() <= 4
    assert (a ^ other).bit_count() > 16


def test_summarize_pairs_counts_by_side_and_caps_examples() -> None:
    a, b = ref("x", "train", "a"), ref("x", "valid", "b")
    c, d = ref("x", "train", "c"), ref("y", "train", "d")
    summary = summarize_pairs([(b, a), (a, b), (a, c), (a, d)], max_examples=1)
    assert summary["counts"] == {
        "cross_dataset": [{"a": "x/train", "b": "y/train", "pairs": 1}],
        "cross_split": [{"a": "x/train", "b": "x/valid", "pairs": 2}],
        "within_split": [{"a": "x/train", "b": "x/train", "pairs": 1}],
    }
    assert summary["examples"]["cross_split"] == [["x/train/a", "x/valid/b"]]


def grid_config(**changes: object) -> GridConfig:
    base = {"rows": 2, "cols": 2, "tile_px": 32, "max_bytes": 100_000}
    return GridConfig.model_validate({**base, **changes})


def alpha_records(raw_dir: Path) -> tuple[list[inspect_datasets.ImageRecord], dict[str, Path]]:
    dirs = {s: raw_dir / "alpha" / s for s in ("train", "valid")}
    return [r for s, d in dirs.items() for r in load_split(d, "alpha")], dirs


def test_sample_grid_is_deterministic_and_sized(raw_dir: Path) -> None:
    records, dirs = alpha_records(raw_dir)
    first = sample_grid(records, dirs, grid_config(), seed="1:alpha")
    second = sample_grid(records, dirs, grid_config(), seed="1:alpha")
    assert first.tobytes() == second.tobytes()
    # Two 32 px cells per side with a 4 px margin around and between them.
    assert first.size == (2 * 32 + 3 * 4, 2 * 32 + 3 * 4)


def test_tile_keeps_the_full_grid_size_with_fewer_images() -> None:
    config = grid_config(rows=2, cols=3, tile_px=10)
    one = tile([Image.new("RGB", (10, 5), (255, 0, 0))], ["train"], config)
    assert one.size == (3 * 10 + 4 * 4, 2 * 10 + 3 * 4)
    # The 10x5 image sits centered vertically in the first cell.
    assert one.getpixel((4 + 9, 4 + 3)) == (255, 0, 0)
    assert one.getpixel((4 + 9, 4 + 1)) == inspect_datasets.BACKGROUND


def test_sample_grid_never_shows_excluded_images(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records, dirs = alpha_records(raw_dir)
    opened: list[str] = []
    real_open = Image.open

    def spy(path: Path) -> Image.Image:
        opened.append(Path(path).name)
        return real_open(path)

    monkeypatch.setattr(inspect_datasets.Image, "open", spy)
    config = grid_config(exclude={"alpha": ["a.png", "c.png"]})
    sample_grid(records, dirs, config, seed="1:alpha")
    assert sorted(opened) == ["b.png", "d.png"]


def test_excluding_one_image_keeps_the_others(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records, dirs = alpha_records(raw_dir)
    opened: list[str] = []
    real_open = Image.open

    def spy(path: Path) -> Image.Image:
        opened.append(Path(path).name)
        return real_open(path)

    monkeypatch.setattr(inspect_datasets.Image, "open", spy)
    small = grid_config(rows=1, cols=2)
    sample_grid(records, dirs, small, seed="1:alpha")
    before = list(opened)
    opened.clear()
    sample_grid(
        records, dirs, grid_config(rows=1, cols=2, exclude={"alpha": [before[0]]}), "1:alpha"
    )
    assert opened[0] == before[1]
    assert before[0] not in opened


def test_encode_png_fits_the_limit_by_shrinking() -> None:
    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 256, (300, 300, 3), np.uint8))
    data = encode_png(noisy, max_bytes=20_000)
    assert len(data) <= 20_000
    with Image.open(io.BytesIO(data)) as decoded:
        assert decoded.width < 300


def test_encode_png_gives_up_when_nothing_fits() -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        encode_png(smooth_image(1), max_bytes=10)


def run_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    monkeypatch.chdir(tmp_path)
    shutil.copy(REPO_ROOT / "configs" / "project.yaml", tmp_path / "project.yaml")
    (tmp_path / "datasets.yaml").write_text(
        "datasets:\n"
        + "".join(
            f"  {k}:\n    workspace: team\n    project: p\n    version: 1\n    license: CC BY 4.0\n"
            for k in ("alpha", "beta", "missing")
        ),
        encoding="utf-8",
    )
    code = inspect_datasets.main(
        ["--project-config", "project.yaml", "--datasets-config", "datasets.yaml", *extra]
    )
    assert code == 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_main_writes_every_report(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    field = tmp_path / "data" / "field_test" / "event1"
    field.mkdir(parents=True)
    shutil.copy(raw_dir / "beta" / "train" / "e.png", field / "frame.png")
    run_cli(tmp_path, monkeypatch)

    stats = read_csv(tmp_path / "reports" / "dataset_stats.csv")
    assert [(r["dataset"], r["split"], r["images"]) for r in stats] == [
        ("alpha", "train", "3"),
        ("alpha", "valid", "1"),
        ("beta", "train", "2"),
    ]
    counts = read_csv(tmp_path / "reports" / "class_counts.csv")
    assert ("beta", "train", "robot", "1") in [
        (r["dataset"], r["split"], r["label"], r["instances"]) for r in counts
    ]

    report = json.loads((tmp_path / "reports" / "duplicates.json").read_text())
    assert report["field_test_images"] == 1
    assert report["exact"]["groups"] == 3
    assert report["exact"]["counts"] == {
        "cross_dataset": [
            {"a": "alpha/train", "b": "beta/train", "pairs": 1},
            {"a": "beta/train", "b": "field_test/event1", "pairs": 1},
        ],
        "within_split": [{"a": "alpha/train", "b": "alpha/train", "pairs": 1}],
    }
    near = report["near"]["counts"]
    assert near["cross_split"] == [{"a": "alpha/train", "b": "alpha/valid", "pairs": 2}]
    assert report["source_name_overlap"] == {"alpha": {"names": 1, "examples": ["a.png"]}}

    for key in ("alpha", "beta"):
        grid = tmp_path / "docs" / "assets" / f"samples_{key}.png"
        assert 0 < grid.stat().st_size <= 450_000
    assert not (tmp_path / "docs" / "assets" / "samples_missing.png").exists()


def test_main_without_grids_writes_no_images(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cli(tmp_path, monkeypatch, "--no-grids")
    assert (tmp_path / "reports" / "dataset_stats.csv").is_file()
    assert not (tmp_path / "docs" / "assets").exists()
