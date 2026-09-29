import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from conftest import smooth_image
from PIL import Image, ImageOps
from pydantic import ValidationError

from frc_xdata import merge
from frc_xdata.config import ClassMapConfig, MergeConfig, ProjectConfig
from frc_xdata.errors import DataLeakError, SplitLeakError
from frc_xdata.harmonize import to_coco
from frc_xdata.inspect_datasets import ANNOTATIONS_NAME, Box, ImageRecord, ImageRef
from frc_xdata.merge import dihedral, dihedral_hashes, near_any, run_merge

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_MAP = ClassMapConfig(classes=["fuel", "robot"], datasets={})
MERGE = {
    "name": "merged",
    "report": "reports/merge.json",
    "sources": {
        "alpha": {"role": "keep_splits", "exclude_pattern": "^skip_"},
        "beta": {"role": "train_only", "dedupe_copies": True},
    },
    "protected": ["lock"],
}
FUEL = (Box("fuel", 4, 4, 10, 10),)
FRAMES = r"^(?P<recording>.+?)[_-](?P<frame>\d+)\.jpg$"


def put(key: str, split: str, images: dict[str, tuple[Image.Image, str, tuple[Box, ...]]]) -> None:
    """Write one harmonized split: file name -> (image, source name, boxes)."""
    split_dir = Path("data/harmonized") / key / split
    split_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (image, source, boxes) in images.items():
        image.save(split_dir / name)
        records.append(ImageRecord(ImageRef(key, split, name), source, 64, 64, boxes))
    (split_dir / ANNOTATIONS_NAME).write_text(
        json.dumps(to_coco(records, ["fuel", "robot"])), encoding="utf-8"
    )


@pytest.fixture
def project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProjectConfig:
    """Harmonized alpha (kept splits), beta (train only), and lock (protected).

    beta's train holds a clean image, a mirrored copy of alpha's test image,
    a rotated copy of the lockbox image, a new image sharing a source name
    with alpha's test image, and two copies of one photo. Two more copies
    of another photo differ in that only the one dropped as a copy matches
    alpha's valid image. Of three video frames, one comes before alpha's
    valid frame by more than the buffer, one within it, and one is from the
    lockbox's recording. beta's valid image joins train.
    """
    monkeypatch.chdir(tmp_path)
    test_image = smooth_image(4)
    lock_image = smooth_image(6)
    valid_image = smooth_image(3)
    put(
        "alpha",
        "train",
        {
            "a1.png": (smooth_image(1), "a1.jpg", FUEL),
            "a2.png": (smooth_image(2), "skip_a2.jpg", ()),
        },
    )
    put("alpha", "valid", {"av.png": (valid_image, "run_0010.jpg", FUEL)})
    put("alpha", "test", {"at.png": (test_image, "shared.jpg", FUEL)})
    put(
        "beta",
        "train",
        {
            "b1.png": (smooth_image(5), "b1.jpg", FUEL),
            "b2.png": (ImageOps.mirror(test_image), "b2.jpg", FUEL),
            "b3.png": (lock_image.rotate(90), "b3.jpg", ()),
            "b4.png": (smooth_image(7), "shared.jpg", ()),
            # Two augmented copies of one photo. The rotated one has the larger box.
            "b6a.png": (smooth_image(9), "b6.jpg", (Box("fuel", 4, 4, 20, 20),)),
            "b6b.png": (smooth_image(10), "b6.jpg", FUEL),
            "b7a.png": (ImageOps.flip(valid_image), "b7.jpg", (Box("fuel", 4, 4, 20, 20),)),
            "b7b.png": (smooth_image(11), "b7.jpg", FUEL),
            "b8.png": (smooth_image(12), "run_0004.jpg", ()),
            "b9.png": (smooth_image(13), "run_0005.jpg", ()),
            "b10.png": (smooth_image(14), "clip_mov-0050.jpg", ()),
        },
    )
    put("beta", "valid", {"b5.png": (smooth_image(8), "b5.jpg", (Box("robot", 0, 0, 30, 30),))})
    put("lock", "test", {"l1.png": (lock_image, "clip_mov-0001.jpg", FUEL)})
    Path("reports").mkdir()
    Path("reports/class_coverage.json").write_text(
        json.dumps({"alpha": {"labeled": ["fuel"]}}), encoding="utf-8"
    )
    data = yaml.safe_load((REPO_ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    data["merge"] = MERGE
    data["splits"]["buffer_frames"] = 5
    data["splits"]["datasets"] = {
        "alpha": {"method": "temporal", "recording_pattern": FRAMES},
        "lock": {"method": "eval_only", "recording_pattern": r"^(?P<recording>.+_mov)-\d+\.jpg$"},
    }
    return ProjectConfig.model_validate(data)


def merged_names(split: str) -> list[str]:
    coco = json.loads(
        (Path("data/harmonized/merged") / split / ANNOTATIONS_NAME).read_text(encoding="utf-8")
    )
    return sorted(i["file_name"] for i in coco["images"])


def report() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path("reports/merge.json").read_text(encoding="utf-8"))
    return data


def test_dihedral_gives_eight_distinct_views_of_an_asymmetric_image() -> None:
    views = [np.asarray(v).tobytes() for v in dihedral(smooth_image(1))]
    assert len(views) == 8
    assert len(set(views)) == 8


def test_a_flipped_or_rotated_copy_is_near_its_original(tmp_path: Path) -> None:
    original = smooth_image(1)
    for n, view in enumerate(
        [ImageOps.mirror(original), original.rotate(270), ImageOps.flip(original)]
    ):
        view.save(tmp_path / f"{n}.png")
    candidates = np.array(
        [dihedral_hashes(tmp_path / f"{n}.png") for n in range(3)], dtype=np.uint64
    )
    original.save(tmp_path / "o.png")
    protected = np.array([dihedral_hashes(tmp_path / "o.png")[0]], dtype=np.uint64)
    assert near_any(candidates, protected, 4).tolist() == [True, True, True]
    assert not near_any(candidates[:, :1], protected, 4).any()


def test_near_any_with_nothing_protected_matches_nothing() -> None:
    assert near_any(
        np.zeros((2, 8), dtype=np.uint64), np.array([], dtype=np.uint64), 4
    ).tolist() == [
        False,
        False,
    ]


def test_merge_keeps_test_splits_and_drops_every_kind_of_match(project: ProjectConfig) -> None:
    run_merge(project, CLASS_MAP)
    assert merged_names("train") == [
        "alpha__a1.png",
        "beta__b1.png",
        "beta__b5.png",
        "beta__b6b.png",
        "beta__b8.png",
    ]
    assert merged_names("valid") == ["alpha__av.png"]
    assert merged_names("test") == ["alpha__at.png"]
    merged = report()["datasets"]["merged"]
    assert merged["dropped"] == {
        "alpha": {"excluded": 1},
        "beta": {"copy": 1, "near_protected": 4, "same_scene": 2, "source_name": 1},
    }
    assert merged["splits"] == {
        "train": {"images": 5},
        "valid": {"images": 1},
        "test": {"images": 1},
    }
    assert merged["from_source"]["beta"] == {"train": 4, "valid": 0, "test": 0}


def test_merged_test_images_are_the_harmonized_files(project: ProjectConfig) -> None:
    run_merge(project, CLASS_MAP)
    original = Path("data/harmonized/alpha/test/at.png").read_bytes()
    assert Path("data/harmonized/merged/test/alpha__at.png").read_bytes() == original
    coco = json.loads(Path("data/harmonized/merged/test", ANNOTATIONS_NAME).read_text("utf-8"))
    (entry,) = coco["images"]
    assert entry["extra"] == {
        "name": "shared.jpg",
        "source_dataset": "alpha",
        "source_split": "test",
    }
    assert len(coco["annotations"]) == 1


def test_merge_writes_a_coverage_entry_and_keeps_the_others(project: ProjectConfig) -> None:
    run_merge(project, CLASS_MAP)
    coverage = json.loads(Path("reports/class_coverage.json").read_text(encoding="utf-8"))
    assert coverage["alpha"] == {"labeled": ["fuel"]}
    assert coverage["merged"]["labeled"] == ["fuel", "robot"]
    assert coverage["merged"]["splits"]["train"]["classes"]["robot"]["instances"] == 1


def test_merge_is_the_same_on_a_second_run(project: ProjectConfig) -> None:
    run_merge(project, CLASS_MAP)
    first = {
        p: p.read_bytes() for p in sorted(Path("data/harmonized/merged").rglob("*")) if p.is_file()
    }
    first_report = report()
    run_merge(project, CLASS_MAP)
    second = {
        p: p.read_bytes() for p in sorted(Path("data/harmonized/merged").rglob("*")) if p.is_file()
    }
    assert first == second
    assert report() == first_report


@pytest.mark.parametrize("view", ["copy", "mirror"])
def test_a_field_test_image_in_the_merged_data_stops_the_run(
    project: ProjectConfig, view: str
) -> None:
    field = Path("data/field_test/event")
    field.mkdir(parents=True)
    if view == "copy":
        shutil.copy("data/harmonized/alpha/valid/av.png", field / "frame.png")
    else:
        ImageOps.mirror(Image.open("data/harmonized/beta/train/b1.png")).save(field / "frame.png")
    with pytest.raises(DataLeakError, match="1 merged images match the field test set"):
        run_merge(project, CLASS_MAP)
    assert not Path("data/harmonized/merged").exists()


def test_an_unrelated_field_test_image_passes(project: ProjectConfig) -> None:
    field = Path("data/field_test/event")
    field.mkdir(parents=True)
    smooth_image(99).save(field / "frame.png")
    run_merge(project, CLASS_MAP)
    assert report()["field_test_images"] == 1


def test_a_kept_train_image_matching_a_protected_one_stops_the_run(project: ProjectConfig) -> None:
    ImageOps.flip(Image.open("data/harmonized/lock/test/l1.png")).save(
        "data/harmonized/alpha/train/a1.png"
    )
    with pytest.raises(SplitLeakError, match="1 merged train images"):
        run_merge(project, CLASS_MAP)


def test_a_kept_train_image_close_to_its_own_valid_split_is_allowed(
    project: ProjectConfig,
) -> None:
    shutil.copy("data/harmonized/alpha/valid/av.png", "data/harmonized/alpha/train/a1.png")
    run_merge(project, CLASS_MAP)
    assert "alpha__a1.png" in merged_names("train")


def test_merge_settings_keep_sources_and_protected_datasets_apart() -> None:
    with pytest.raises(ValidationError, match="both merged and protected"):
        MergeConfig.model_validate({**MERGE, "protected": ["alpha"]})
    with pytest.raises(ValidationError, match="keep its splits"):
        MergeConfig.model_validate({**MERGE, "sources": {"beta": {"role": "train_only"}}})
    with pytest.raises(ValidationError, match="does not compile"):
        MergeConfig.model_validate(
            {**MERGE, "sources": {"alpha": {"role": "keep_splits", "exclude_pattern": "^(x"}}}
        )


def test_committed_merge_settings_load() -> None:
    data = yaml.safe_load((REPO_ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    cfg = ProjectConfig.model_validate(data).merge
    assert cfg is not None
    assert cfg.protected == ["pankratz"]
    assert {k: s.role for k, s in cfg.sources.items()} == {
        "marswars": "keep_splits",
        "robotzftp2": "keep_splits",
        "scorekeeper": "keep_splits",
        "testingfrfr": "train_only",
    }


def test_main_runs_the_merge(project: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    data = yaml.safe_load((REPO_ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    data["merge"] = MERGE
    Path("configs").mkdir()
    Path("configs/project.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    Path("configs/class_map.yaml").write_text(
        yaml.safe_dump(CLASS_MAP.model_dump()), encoding="utf-8"
    )
    assert (
        merge.main(
            ["--project-config", "configs/project.yaml", "--class-map", "configs/class_map.yaml"]
        )
        == 0
    )
    assert merged_names("test") == ["alpha__at.png"]
