import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
from conftest import coco, image_entry, smooth_image, write_split

from frc_xdata import upload
from frc_xdata.config import API_KEY_VAR, ProjectConfig, load_yaml
from frc_xdata.errors import ConfigError, DataLeakError, DirtyTreeError, UploadCheckError
from frc_xdata.harmonize import to_coco
from frc_xdata.inspect_datasets import Box, ImageRecord, ImageRef
from frc_xdata.upload import (
    VersionInfo,
    check_before_upload,
    compare_export,
    reported_sizes,
    split_sizes,
    upload_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-test-secret"
# Harmonized layout: a (train, one box), b (valid, no boxes), c (test, one box).
FUEL = Box("fuel", 4, 4, 8, 8)
LAYOUT = {"train": ("a.png", 1, 1), "valid": ("b.png", 2, 0), "test": ("c.png", 3, 1)}


@pytest.fixture
def project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProjectConfig:
    """Run from ``tmp_path`` with one harmonized dataset and its split report."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(REPO_ROOT / "configs" / "project.yaml", tmp_path / "project.yaml")
    project = load_yaml(tmp_path / "project.yaml", ProjectConfig)
    dataset = project.paths.harmonized_dir / "alpha"
    for split, (name, seed, boxes) in LAYOUT.items():
        record = ImageRecord(ImageRef("alpha", split, name), name, 64, 64, (FUEL,) * boxes)
        write_split(dataset / split, to_coco([record], ["fuel", "robot"]))
        smooth_image(seed).save(dataset / split / name)
    write_splits_report(project, {"train": 1, "valid": 1, "test": 1})
    return project


def write_splits_report(project: ProjectConfig, sizes: dict[str, int]) -> None:
    report = {"datasets": {"alpha": {"splits": {s: {"images": n} for s, n in sizes.items()}}}}
    project.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (project.paths.reports_dir / "splits.json").write_text(json.dumps(report), encoding="utf-8")


def write_export(root: Path, images: dict[str, list[tuple[str, int]]]) -> None:
    """Write the COCO files of a Roboflow-style export, renamed but keeping the upload name."""
    for split, entries in images.items():
        split_dir = root / split
        records, anns = [], []
        for image_id, (name, boxes) in enumerate(entries):
            exported = f"{Path(name).stem}_png.rf.{image_id}.png"
            records.append(image_entry(image_id, exported, source=name))
            anns += [(image_id, 1, [4.0, 4.0, 8.0, 8.0])] * boxes
        write_split(split_dir, coco(records, anns))


MATCHING = {"train": [("a.png", 1)], "valid": [("b.png", 0)], "test": [("c.png", 1)]}


def test_split_sizes_loads_every_split(project: ProjectConfig) -> None:
    assert split_sizes(project.paths.harmonized_dir / "alpha") == {
        "train": 1,
        "valid": 1,
        "test": 1,
    }


def test_split_sizes_refuses_a_missing_split(project: ProjectConfig) -> None:
    shutil.rmtree(project.paths.harmonized_dir / "alpha" / "test")
    with pytest.raises(UploadCheckError, match="test"):
        split_sizes(project.paths.harmonized_dir / "alpha")


def test_reported_sizes_refuses_a_dataset_that_was_not_resplit(project: ProjectConfig) -> None:
    with pytest.raises(UploadCheckError, match="beta"):
        reported_sizes(project.paths.reports_dir / "splits.json", "beta")


def test_check_before_upload_passes_and_warns_without_field_test(
    project: ProjectConfig, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        sizes = check_before_upload(project.paths.harmonized_dir / "alpha", "alpha", project)
    assert sizes == {"train": 1, "valid": 1, "test": 1}
    assert "no field-test images" in caplog.text


def test_check_before_upload_refuses_sizes_that_differ_from_the_report(
    project: ProjectConfig,
) -> None:
    write_splits_report(project, {"train": 1, "valid": 2, "test": 1})
    with pytest.raises(UploadCheckError, match="valid has 1, the report says 2"):
        check_before_upload(project.paths.harmonized_dir / "alpha", "alpha", project)


def test_check_before_upload_refuses_a_field_test_near_duplicate(project: ProjectConfig) -> None:
    field = project.paths.field_test_dir / "event"
    field.mkdir(parents=True)
    smooth_image(3).resize((48, 48)).save(field / "frame.png")
    smooth_image(99).save(field / "other.png")
    with pytest.raises(DataLeakError, match="1 image pairs"):
        check_before_upload(project.paths.harmonized_dir / "alpha", "alpha", project)


def test_upload_splits_names_every_split_in_its_own_call(tmp_path: Path) -> None:
    calls: list[tuple[Path, str]] = []
    upload_splits(tmp_path, lambda path, split: calls.append((path, split)))
    assert calls == [(tmp_path / s, s) for s in ("train", "valid", "test")]


def test_compare_export_accepts_a_matching_version(project: ProjectConfig, tmp_path: Path) -> None:
    write_export(tmp_path / "export", MATCHING)
    result = compare_export(tmp_path / "export", project.paths.harmonized_dir / "alpha", "alpha")
    assert result.ok
    assert result.exported == result.expected == {"train": 1, "valid": 1, "test": 1}


def test_compare_export_lists_every_kind_of_mismatch(
    project: ProjectConfig, tmp_path: Path
) -> None:
    write_export(
        tmp_path / "export",
        {
            "train": [("a.png", 1), ("b.png", 0), ("stranger.png", 0)],
            "valid": [("b.png", 0)],
            "test": [("c.png", 3)],
        },
    )
    result = compare_export(tmp_path / "export", project.paths.harmonized_dir / "alpha", "alpha")
    assert not result.ok
    assert result.missing == []
    assert result.extra == ["b.png", "stranger.png"]
    assert result.moved == [("b.png", "valid", "train")]
    assert result.box_count_changed == ["c.png"]
    assert result.exported == {"train": 3, "valid": 1, "test": 1}


def test_compare_export_lists_missing_images(project: ProjectConfig, tmp_path: Path) -> None:
    write_export(tmp_path / "export", {"train": [("a.png", 1)], "valid": [("b.png", 0)]})
    result = compare_export(tmp_path / "export", project.paths.harmonized_dir / "alpha", "alpha")
    assert result.missing == ["c.png"]
    assert result.exported["test"] == 0


def test_upload_main_dry_run_checks_without_the_network(
    project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> None:
        raise AssertionError("dry run must not reach Roboflow")

    monkeypatch.delenv(API_KEY_VAR, raising=False)
    monkeypatch.setattr(upload, "_roboflow_upload", fail)
    assert upload.upload_main(["alpha", "--project-config", "project.yaml", "--dry-run"]) == 0


def test_upload_main_uploads_each_split_to_the_configured_project(
    project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake(api_key: str, slug: str, key: str, retries: int) -> upload.Upload:
        assert (api_key, slug, key) == (SECRET, project.platform.project, "alpha")
        return lambda path, split: calls.append((path.name, split))

    monkeypatch.setenv(API_KEY_VAR, SECRET)
    monkeypatch.setattr(upload, "_roboflow_upload", fake)
    assert upload.upload_main(["alpha", "--project-config", "project.yaml"]) == 0
    assert calls == [("train", "train"), ("valid", "valid"), ("test", "test")]


def test_upload_main_uploads_nothing_when_a_check_fails(
    project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_VAR, SECRET)
    monkeypatch.setattr(upload, "_roboflow_upload", lambda *a: pytest.fail("uploaded"))
    write_splits_report(project, {"train": 2, "valid": 1, "test": 1})
    with pytest.raises(UploadCheckError):
        upload.upload_main(["alpha", "--project-config", "project.yaml"])


def run_verify(
    monkeypatch: pytest.MonkeyPatch,
    export: dict[str, list[tuple[str, int]]],
    *extra: str,
    dirty: bool = False,
) -> int:
    monkeypatch.setenv(API_KEY_VAR, SECRET)
    monkeypatch.setattr(upload, "git_is_dirty", lambda repo: dirty)
    monkeypatch.setattr(upload, "git_commit", lambda repo: "abc123")

    def fake(api_key: str, slug: str, version: int) -> upload.Export:
        def download(dest: Path) -> VersionInfo:
            write_export(dest, export)
            return VersionInfo(
                splits={"train": 1, "valid": 1, "test": 1},
                preprocessing={"auto-orient": True},
                augmentation={},
            )

        return download

    monkeypatch.setattr(upload, "_roboflow_export", fake)
    return upload.verify_main(
        ["alpha", "--project-config", "project.yaml", "--version", "4", *extra]
    )


def test_verify_main_writes_a_passing_report(
    project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_verify(monkeypatch, MATCHING) == 0
    report = json.loads((project.paths.reports_dir / "platform_upload_alpha.json").read_text())
    assert report["ok"] is True
    assert (report["project"], report["version"], report["git_commit"]) == (
        project.platform.project,
        4,
        "abc123",
    )
    assert report["preprocessing"] == {"auto-orient": True}
    assert report["missing"] == {"count": 0, "examples": []}
    assert SECRET not in json.dumps(report)


def test_verify_main_fails_and_reports_a_moved_image(
    project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved = {"train": [("a.png", 1), ("b.png", 0)], "test": [("c.png", 1)]}
    assert run_verify(monkeypatch, moved) == 1
    report = json.loads((project.paths.reports_dir / "platform_upload_alpha.json").read_text())
    assert report["ok"] is False
    assert report["moved"] == {"count": 1, "examples": [["b.png", "valid", "train"]]}


def test_verify_main_refuses_a_dirty_tree_unless_allowed(
    project: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(DirtyTreeError):
        run_verify(monkeypatch, MATCHING, dirty=True)
    assert run_verify(monkeypatch, MATCHING, "--allow-dirty", dirty=True) == 0
    report = json.loads((project.paths.reports_dir / "platform_upload_alpha.json").read_text())
    assert report["dirty"] is True


@pytest.mark.integration
def test_missing_project_is_a_config_error_not_a_new_project() -> None:
    from frc_xdata.config import roboflow_api_key

    with pytest.raises(ConfigError, match="Create it in the web app"):
        upload._roboflow_project(roboflow_api_key(), "frc-xdata-no-such-project-9f3a")


def test_compare_export_refuses_a_name_shared_by_two_splits(
    project: ProjectConfig, tmp_path: Path
) -> None:
    dataset = project.paths.harmonized_dir / "alpha"
    shutil.copy(dataset / "train" / "_annotations.coco.json", dataset / "test")
    write_export(tmp_path / "export", MATCHING)
    with pytest.raises(UploadCheckError, match="more than one harmonized split"):
        compare_export(tmp_path / "export", dataset, "alpha")
