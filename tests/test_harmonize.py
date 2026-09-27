import csv
import json
from pathlib import Path
from typing import Any

import pytest
import supervision as sv
import yaml
from conftest import coco, image_entry

from frc_xdata import harmonize
from frc_xdata.config import ClassMapConfig, load_yaml
from frc_xdata.errors import SplitLeakError, UnmappedLabelError
from frc_xdata.harmonize import class_coverage, map_labels, to_coco
from frc_xdata.inspect_datasets import Box, ImageRecord, ImageRef, parse_coco

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["fuel", "robot"]


def record(name: str, *labels: str, split: str = "train", dataset: str = "ds") -> ImageRecord:
    boxes = tuple(Box(label, 1, 1, 2, 3) for label in labels)
    return ImageRecord(ImageRef(dataset, split, name), name, 64, 64, boxes)


def test_map_labels_maps_every_label_and_counts_before_and_after() -> None:
    records = [record("a", "FUEL", "robot"), record("b", "FUEL", split="valid")]
    mapped, counts = map_labels(records, {"FUEL": "fuel", "robot": "robot"})
    assert [[b.label for b in r.boxes] for r in mapped] == [["fuel", "robot"], ["fuel"]]
    assert [(c.split, c.source_label, c.target, c.instances) for c in counts] == [
        ("train", "FUEL", "fuel", 1),
        ("train", "robot", "robot", 1),
        ("valid", "FUEL", "fuel", 1),
    ]


def test_map_labels_drop_removes_boxes_and_keeps_the_image() -> None:
    mapped, counts = map_labels(
        [record("a", "game_piece", "inactive"), record("b", "inactive")],
        {"game_piece": "fuel", "inactive": "DROP"},
    )
    assert [r.ref.file_name for r in mapped] == ["a", "b"]
    assert [[b.label for b in r.boxes] for r in mapped] == [["fuel"], []]
    assert ("inactive", "DROP", 2) in [(c.source_label, c.target, c.instances) for c in counts]


def test_map_labels_keeps_box_geometry() -> None:
    (mapped,), _ = map_labels([record("a", "FUEL")], {"FUEL": "fuel"})
    assert mapped.boxes == (Box("fuel", 1, 1, 2, 3),)


def test_map_labels_unmapped_label_raises_and_names_it() -> None:
    records = [record("a", "fuel", "Fuel", dataset="lava"), record("b", "robot", dataset="lava")]
    with pytest.raises(UnmappedLabelError, match=r"lava: 'Fuel', lava: 'robot'"):
        map_labels(records, {"fuel": "fuel"})


def test_map_labels_ignores_labels_absent_from_the_data() -> None:
    mapped, counts = map_labels([record("a")], {"fuel": "fuel"})
    assert mapped == [record("a")]
    assert counts == []


def test_placeholder_sharing_a_real_name_is_ignored() -> None:
    # robotzftp2 names both its unused placeholder (id 0) and its real class
    # (id 1) "fuel".
    data = coco([image_entry(0, "x.jpg")], [(0, 1, [1, 2, 3, 4])])
    data["categories"] = [
        {"id": 0, "name": "fuel", "supercategory": "none"},
        {"id": 1, "name": "fuel", "supercategory": "fuel"},
    ]
    mapped, _ = map_labels(parse_coco(data, "robotzftp2", "train"), {"fuel": "fuel"})
    out = to_coco(mapped, CLASSES)
    assert out["categories"] == [
        {"id": 1, "name": "fuel", "supercategory": "none"},
        {"id": 2, "name": "robot", "supercategory": "none"},
    ]
    assert [a["category_id"] for a in out["annotations"]] == [1]


def test_to_coco_writes_boxes_and_provenance() -> None:
    out = to_coco([record("b", "robot", split="valid"), record("a", "fuel", "fuel")], CLASSES)
    assert [(i["id"], i["file_name"]) for i in out["images"]] == [(0, "a"), (1, "b")]
    assert out["images"][1]["extra"] == {"name": "b", "source_split": "valid"}
    assert [(a["image_id"], a["category_id"]) for a in out["annotations"]] == [
        (0, 1),
        (0, 1),
        (1, 2),
    ]
    assert out["annotations"][0]["bbox"] == [1, 1, 2, 3]
    assert out["annotations"][0]["area"] == 6
    assert len({a["id"] for a in out["annotations"]}) == 3


def test_class_coverage_leaves_out_classes_a_dataset_never_labels() -> None:
    splits = {
        "fuel_only": {"train": [record("a", "fuel")], "test": [record("b")]},
        "both": {"train": [record("c", "fuel", "robot", "robot")]},
    }
    coverage = class_coverage(splits, CLASSES)
    assert coverage["fuel_only"]["labeled"] == ["fuel"]
    assert coverage["both"]["labeled"] == ["fuel", "robot"]
    assert coverage["fuel_only"]["splits"]["test"] == {
        "images": 1,
        "classes": {
            "fuel": {"instances": 0, "images": 0},
            "robot": {"instances": 0, "images": 0},
        },
    }
    assert coverage["both"]["splits"]["train"]["classes"]["robot"] == {
        "instances": 2,
        "images": 1,
    }


def test_committed_class_map_drops_only_marswars_state_classes() -> None:
    cfg = load_yaml(REPO_ROOT / "configs" / "class_map.yaml", ClassMapConfig)
    dropped = {(k, label) for k, m in cfg.datasets.items() for label, t in m.items() if t == "DROP"}
    assert dropped == {
        ("marswars", "blue_active"),
        ("marswars", "red_active"),
        ("marswars", "inactive"),
    }


def run_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    class_map: str,
    splits: dict[str, Any] | None = None,
) -> None:
    monkeypatch.chdir(tmp_path)
    project = yaml.safe_load((REPO_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    project["splits"]["datasets"] = splits or {}
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(project), encoding="utf-8")
    (tmp_path / "datasets.yaml").write_text(
        "datasets:\n"
        + "".join(
            f"  {k}:\n    workspace: team\n    project: p\n    version: 1\n    license: CC BY 4.0\n"
            for k in ("alpha", "beta", "missing")
        ),
        encoding="utf-8",
    )
    (tmp_path / "class_map.yaml").write_text(class_map, encoding="utf-8")
    code = harmonize.main(
        [
            "--project-config",
            "project.yaml",
            "--datasets-config",
            "datasets.yaml",
            "--class-map",
            "class_map.yaml",
        ]
    )
    assert code == 0


CLASS_MAP = """
classes: [fuel, robot]
datasets:
  alpha: {fuel: fuel, robot: DROP}
  beta: {robot: robot}
"""


def test_main_writes_harmonized_splits_that_supervision_loads(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cli(tmp_path, monkeypatch, CLASS_MAP)
    out = tmp_path / "data" / "harmonized"
    assert sorted(p.relative_to(out).as_posix() for p in out.glob("*/*")) == [
        "alpha/train",
        "alpha/valid",
        "beta/train",
    ]
    train = out / "alpha" / "train"
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=str(train),
        annotations_path=str(train / "_annotations.coco.json"),
    )
    assert ds.classes == ["fuel", "robot"]
    assert len(ds) == 3
    labels = {Path(path).name: list(det.class_id) for path, _, det in ds}
    assert labels == {"a.png": [0], "b.png": [], "c.png": [0]}
    assert (train / "a.png").samefile(raw_dir / "alpha" / "train" / "a.png")


def test_main_writes_counts_and_coverage(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cli(tmp_path, monkeypatch, CLASS_MAP)
    with (tmp_path / "reports" / "harmonize_counts.csv").open(encoding="utf-8") as f:
        rows = [tuple(r.values()) for r in csv.DictReader(f)]
    assert rows == [
        ("alpha", "train", "fuel", "fuel", "2"),
        ("alpha", "train", "robot", "DROP", "1"),
        ("alpha", "valid", "fuel", "fuel", "1"),
        ("beta", "train", "robot", "robot", "1"),
    ]
    coverage = json.loads((tmp_path / "reports" / "class_coverage.json").read_text())
    assert coverage["alpha"]["labeled"] == ["fuel"]
    assert coverage["beta"]["labeled"] == ["robot"]
    assert "missing" not in coverage


def test_main_replaces_stale_output(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "data" / "harmonized" / "alpha" / "test" / "old.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"")
    run_cli(tmp_path, monkeypatch, CLASS_MAP)
    assert not stale.parent.exists()


def test_main_stops_on_an_unmapped_label(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(UnmappedLabelError, match="alpha: 'robot'"):
        run_cli(tmp_path, monkeypatch, "classes: [fuel, robot]\ndatasets:\n  alpha: {fuel: fuel}\n")


GROUPED = {
    "alpha": {
        "method": "grouped",
        "recording_pattern": "^(?P<recording>no-recordings)$",
        "dedupe_copies": True,
    }
}


def test_main_resplits_configured_datasets_and_reports_it(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With robot dropped, c is a copy of a (same source name, one fuel box),
    # and a has the smaller box. d shares a's source name, so they stay
    # together, and the larger group fills train first.
    run_cli(tmp_path, monkeypatch, CLASS_MAP, GROUPED)
    out = tmp_path / "data" / "harmonized"
    names = {
        p.relative_to(out).as_posix()
        for p in out.glob("*/*/*")
        if p.name != "_annotations.coco.json"
    }
    assert names == {
        "alpha/train/a.png",
        "alpha/train/d.png",
        "alpha/valid/b.png",
        "beta/train/e.png",
        "beta/train/f.png",
    }
    train = json.loads((out / "alpha" / "train" / "_annotations.coco.json").read_text())
    assert [i["extra"]["source_split"] for i in train["images"]] == ["train", "valid"]

    report = json.loads((tmp_path / "reports" / "splits.json").read_text())
    assert list(report["datasets"]) == ["alpha"]
    alpha = report["datasets"]["alpha"]
    assert alpha["dropped"] == {"copy": 1, "buffer": 0, "near_test": 0}
    assert alpha["splits"] == {
        "train": {"images": 2, "units": 1},
        "valid": {"images": 1, "units": 1},
        "test": {"images": 0, "units": 0},
    }
    assert alpha["from_source_split"] == {
        "train": {"dropped": 1, "train": 1, "valid": 1},
        "valid": {"train": 1},
    }
    assert alpha["copy_group_sizes"] == {"1": 1, "2": 1}
    assert report["cross_dataset_near_duplicates"] == 0
    coverage = json.loads((tmp_path / "reports" / "class_coverage.json").read_text())
    assert list(coverage["alpha"]["splits"]) == ["train", "valid"]


def test_main_stops_when_two_resplit_datasets_share_an_image(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # beta/f.png is a byte copy of alpha/b.png.
    splits = {**GROUPED, "beta": {**GROUPED["alpha"], "dedupe_copies": False}}
    with pytest.raises(SplitLeakError, match="1 near-duplicate pairs across"):
        run_cli(tmp_path, monkeypatch, CLASS_MAP, splits)
    assert not (tmp_path / "reports" / "splits.json").exists()
