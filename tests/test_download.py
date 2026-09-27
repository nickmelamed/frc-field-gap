import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from frc_xdata import download
from frc_xdata.config import API_KEY_VAR, DatasetSpec
from frc_xdata.download import (
    MANIFEST_NAME,
    Manifest,
    build_manifest,
    digest_lines,
    download_all,
    read_manifest,
    redact,
    resolve_all,
    summarize_project,
    verified_manifest,
    verify_manifest,
    workspace_projects,
)

SECRET = "sEcReT-kEy"

PINNED = DatasetSpec(workspace="team", project="rebuilt", version=2, license="CC BY 4.0")
UNPINNED = DatasetSpec(workspace="team", license="CC BY 4.0")


class FetchError(Exception):
    pass


class FakeFetch:
    """Writes a tiny COCO-shaped dataset and counts how often it ran."""

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_for = fail_for or set()

    def __call__(self, spec: DatasetSpec, dest: Path) -> None:
        assert spec.project is not None
        self.calls.append(spec.project)
        if spec.project in self.fail_for:
            dest.mkdir(parents=True)
            (dest / "half.jpg").write_bytes(b"partial")
            raise FetchError(f"GET https://api.example/{spec.project}?api_key={SECRET} failed")
        (dest / "train").mkdir(parents=True)
        (dest / "train" / "a.jpg").write_bytes(b"image a")
        (dest / "train" / "_annotations.coco.json").write_text("{}", encoding="utf-8")
        (dest / "README.txt").write_text(f"version {spec.version}", encoding="utf-8")


def run(tmp_path: Path, datasets: dict[str, DatasetSpec], fetch: FakeFetch, **kwargs: Any) -> Any:
    return download_all(
        datasets,
        raw_dir=tmp_path / "raw",
        manifests_dir=tmp_path / "manifests",
        fetch=fetch,
        errors=(FetchError,),
        secret=SECRET,
        **kwargs,
    )


def make_dataset(root: Path) -> None:
    (root / "valid").mkdir(parents=True)
    (root / "valid" / "b.jpg").write_bytes(b"bbb")
    (root / "a.txt").write_text("a", encoding="utf-8")


def test_redact_removes_every_occurrence() -> None:
    assert redact(f"{SECRET} and {SECRET}", SECRET) == "<redacted> and <redacted>"


def test_redact_with_empty_secret_is_a_no_op() -> None:
    assert redact("unchanged", "") == "unchanged"


def test_build_manifest_hashes_every_file_sorted(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text("old", encoding="utf-8")
    manifest = build_manifest(tmp_path, "key", PINNED)
    assert [e.path for e in manifest.files] == ["a.txt", "valid/b.jpg"]
    assert manifest.files[1].sha256 == hashlib.sha256(b"bbb").hexdigest()
    assert manifest.files[1].size == 3
    assert (manifest.project, manifest.version, manifest.format) == ("rebuilt", 2, "coco")


def test_build_manifest_refuses_unpinned_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not pinned"):
        build_manifest(tmp_path, "key", UNPINNED)


def test_verify_manifest_clean_copy_has_no_problems(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    assert verify_manifest(tmp_path, build_manifest(tmp_path, "key", PINNED)) == []


def test_verify_manifest_reports_changed_missing_and_unlisted(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    manifest = build_manifest(tmp_path, "key", PINNED)
    (tmp_path / "valid" / "b.jpg").write_bytes(b"BBB")
    (tmp_path / "a.txt").unlink()
    (tmp_path / "new.jpg").write_bytes(b"new")
    assert verify_manifest(tmp_path, manifest) == [
        "missing: a.txt",
        "changed: valid/b.jpg",
        "unlisted: new.jpg",
    ]


def test_digest_lines_list_other_files_and_one_line_per_image_directory(
    tmp_path: Path,
) -> None:
    make_dataset(tmp_path)
    (tmp_path / "valid" / "c.PNG").write_bytes(b"ccc")
    (tmp_path / "valid" / "_annotations.coco.json").write_text("{}", encoding="utf-8")
    lines = digest_lines(build_manifest(tmp_path, "key", PINNED)).splitlines()
    b_line = f"{hashlib.sha256(b'bbb').hexdigest()}  valid/b.jpg\n"
    c_line = f"{hashlib.sha256(b'ccc').hexdigest()}  valid/c.PNG\n"
    combined = hashlib.sha256((b_line + c_line).encode()).hexdigest()
    assert lines == [
        f"{hashlib.sha256(b'a').hexdigest()}  a.txt",
        f"{hashlib.sha256(b'{}').hexdigest()}  valid/_annotations.coco.json",
        f"# images valid/: 2 files, sha256 {combined}",
    ]


def test_digest_lines_change_when_one_image_changes(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    before = digest_lines(build_manifest(tmp_path, "key", PINNED))
    (tmp_path / "valid" / "b.jpg").write_bytes(b"BBB")
    after = digest_lines(build_manifest(tmp_path, "key", PINNED))
    assert before.splitlines()[0] == after.splitlines()[0]
    assert before.splitlines()[1] != after.splitlines()[1]


def test_digest_lines_images_at_the_root_group_under_dot(tmp_path: Path) -> None:
    (tmp_path / "top.jpg").write_bytes(b"t")
    lines = digest_lines(build_manifest(tmp_path, "key", PINNED)).splitlines()
    assert lines[0].startswith("# images ./: 1 files, sha256 ")


def test_read_manifest_missing_or_unreadable_is_none(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text("not json", encoding="utf-8")
    assert read_manifest(tmp_path) is None


def test_verified_manifest_rejects_a_different_version(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    manifest = build_manifest(tmp_path, "key", PINNED)
    (tmp_path / MANIFEST_NAME).write_text(manifest.model_dump_json(), encoding="utf-8")
    assert verified_manifest(tmp_path, PINNED) == manifest
    assert verified_manifest(tmp_path, PINNED.model_copy(update={"version": 3})) is None


def test_download_writes_dataset_manifest_and_digest(tmp_path: Path) -> None:
    fetch = FakeFetch()
    assert run(tmp_path, {"a": PINNED}, fetch) == []
    dest = tmp_path / "raw" / "a"
    manifest = Manifest.model_validate_json((dest / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert verify_manifest(dest, manifest) == []
    assert (tmp_path / "manifests" / "a.sha256").read_text(encoding="utf-8") == digest_lines(
        manifest
    )
    assert not (tmp_path / "raw" / ".a.partial").exists()


def test_second_run_skips_verified_dataset(tmp_path: Path) -> None:
    fetch = FakeFetch()
    run(tmp_path, {"a": PINNED}, fetch)
    (tmp_path / "manifests" / "a.sha256").unlink()
    assert run(tmp_path, {"a": PINNED}, fetch) == []
    assert fetch.calls == ["rebuilt"]
    assert (tmp_path / "manifests" / "a.sha256").is_file()


@pytest.mark.parametrize("change", ["force", "tampered", "new-version"])
def test_download_runs_again_when_needed(tmp_path: Path, change: str) -> None:
    fetch = FakeFetch()
    run(tmp_path, {"a": PINNED}, fetch)
    spec = PINNED
    if change == "tampered":
        (tmp_path / "raw" / "a" / "train" / "a.jpg").write_bytes(b"edited")
    elif change == "new-version":
        spec = PINNED.model_copy(update={"version": 3})
    assert run(tmp_path, {"a": spec}, fetch, force=change == "force") == []
    assert fetch.calls == ["rebuilt", "rebuilt"]
    assert verified_manifest(tmp_path / "raw" / "a", spec) is not None


def test_failure_is_recorded_redacted_and_others_continue(tmp_path: Path) -> None:
    fetch = FakeFetch(fail_for={"broken"})
    broken = PINNED.model_copy(update={"project": "broken"})
    failures = run(tmp_path, {"bad": broken, "good": PINNED, "later": UNPINNED}, fetch)
    assert [f.key for f in failures] == ["bad", "later"]
    assert SECRET not in failures[0].reason
    assert "<redacted>" in failures[0].reason
    assert failures[1].reason == "project or version not pinned"
    assert not (tmp_path / "raw" / "bad").exists()
    assert not (tmp_path / "raw" / ".bad.partial").exists()
    assert verified_manifest(tmp_path / "raw" / "good", PINNED) is not None


def test_failed_redownload_keeps_the_previous_copy(tmp_path: Path) -> None:
    run(tmp_path, {"a": PINNED}, FakeFetch())
    failures = run(tmp_path, {"a": PINNED}, FakeFetch(fail_for={"rebuilt"}), force=True)
    assert [f.key for f in failures] == ["a"]
    assert verified_manifest(tmp_path / "raw" / "a", PINNED) is not None


def test_unexpected_errors_stop_the_run(tmp_path: Path) -> None:
    def fetch(spec: DatasetSpec, dest: Path) -> None:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        download_all(
            {"a": PINNED},
            raw_dir=tmp_path / "raw",
            manifests_dir=tmp_path / "manifests",
            fetch=fetch,
            errors=(FetchError,),
        )


PROJECT_RESPONSE = {
    "project": {"images": 30, "classes": {"robot": 5, "fuel": 40}, "splits": {"train": 20}},
    "versions": [
        {"id": "team/rebuilt/10", "images": 30, "splits": {"train": 20, "valid": 10}},
        {"id": "team/rebuilt/2", "images": 12, "splits": {"train": 12}},
    ],
}

WORKSPACE_RESPONSE = {
    "workspace": {"projects": [{"id": "team/zeta", "name": "Z"}, {"id": "team/alpha"}]}
}


def test_summarize_project_sorts_versions_numerically() -> None:
    summary = summarize_project("team", "rebuilt", PROJECT_RESPONSE)
    assert [v.version for v in summary.versions] == [2, 10]
    assert summary.latest_version == 10
    assert list(summary.classes) == ["fuel", "robot"]
    assert summary.versions[1].splits == {"train": 20, "valid": 10}


def test_summarize_project_without_versions() -> None:
    summary = summarize_project("team", "rebuilt", {"project": {"images": 0}})
    assert summary.latest_version is None


def test_workspace_projects_returns_sorted_slugs() -> None:
    assert workspace_projects(WORKSPACE_RESPONSE) == ["alpha", "zeta"]


def test_resolve_all_lists_workspace_projects_when_project_unset() -> None:
    looked_up: list[tuple[str, str]] = []

    def get_project(ws: str, proj: str) -> Any:
        looked_up.append((ws, proj))
        return PROJECT_RESPONSE

    resolved, failures = resolve_all(
        {"one": PINNED, "all": UNPINNED},
        get_project=get_project,
        get_workspace=lambda ws: WORKSPACE_RESPONSE,
        errors=(FetchError,),
    )
    assert failures == []
    assert [r.key for r in resolved] == ["one", "all"]
    assert [p.project for p in resolved[1].projects] == ["alpha", "zeta"]
    assert looked_up == [("team", "rebuilt"), ("team", "alpha"), ("team", "zeta")]


@pytest.mark.parametrize(
    "response",
    [{"unexpected": True}, {"project": {}, "versions": [{"id": "team/rebuilt/latest"}]}],
    ids=["shape", "version-id"],
)
def test_resolve_all_records_bad_responses(response: Any) -> None:
    resolved, failures = resolve_all(
        {"one": PINNED},
        get_project=lambda ws, proj: response,
        get_workspace=lambda ws: WORKSPACE_RESPONSE,
        errors=(FetchError,),
    )
    assert resolved == []
    assert [f.key for f in failures] == ["one"]


def test_resolve_all_records_api_errors_redacted() -> None:
    def get_workspace(ws: str) -> Any:
        raise FetchError(f"403 for ?api_key={SECRET}")

    resolved, failures = resolve_all(
        {"all": UNPINNED},
        get_project=lambda ws, proj: PROJECT_RESPONSE,
        get_workspace=get_workspace,
        errors=(FetchError,),
        secret=SECRET,
    )
    assert resolved == []
    assert SECRET not in failures[0].reason


def write_configs(tmp_path: Path) -> list[str]:
    (tmp_path / "project.yaml").write_text(
        "seed: 1\npaths:\n"
        "  raw_dir: raw\n  reports_dir: reports\n  manifests_dir: reports/data_manifests\n"
        "  assets_dir: assets\n  field_test_dir: field_test\n",
        encoding="utf-8",
    )
    (tmp_path / "datasets.yaml").write_text(
        "datasets:\n"
        "  a:\n    workspace: team\n    project: rebuilt\n    version: 2\n"
        "    license: CC BY 4.0\n"
        "  b:\n    workspace: team\n    license: CC BY 4.0\n",
        encoding="utf-8",
    )
    return ["--project-config", "project.yaml", "--datasets-config", "datasets.yaml"]


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, SECRET)
    return write_configs(tmp_path)


def test_main_downloads_pinned_and_exits_nonzero_for_unpinned(
    cli: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetch = FakeFetch()
    monkeypatch.setattr(download, "_roboflow_fetch", lambda key: fetch)
    assert download.main(cli) == 1
    assert fetch.calls == ["rebuilt"]
    assert (tmp_path / "reports" / "data_manifests" / "a.sha256").is_file()
    failures = json.loads((tmp_path / "reports" / "download_failures.json").read_text())
    assert failures == [{"key": "b", "reason": "project or version not pinned"}]


def test_main_only_limits_the_run(
    cli: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(download, "_roboflow_fetch", lambda key: FakeFetch())
    assert download.main([*cli, "--only", "a"]) == 0


def test_main_rejects_unknown_keys(cli: list[str]) -> None:
    with pytest.raises(SystemExit, match="unknown dataset keys: nope"):
        download.main([*cli, "--only", "nope"])


def test_main_resolve_writes_resolution(
    cli: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from roboflow.adapters import rfapi

    monkeypatch.setattr(rfapi, "get_project", lambda key, ws, proj: PROJECT_RESPONSE)
    monkeypatch.setattr(rfapi, "get_workspace", lambda key, ws: WORKSPACE_RESPONSE)
    assert download.main([*cli, "--resolve"]) == 0
    resolved = json.loads((tmp_path / "reports" / "dataset_resolution.json").read_text())
    assert [r["key"] for r in resolved] == ["a", "b"]
    assert resolved[0]["projects"][0]["latest_version"] == 10


def test_main_redacts_sdk_errors_before_any_dataset(
    cli: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_login(key: str) -> Any:
        raise RuntimeError(f"bad key {key}")

    monkeypatch.setattr(download, "_roboflow_fetch", failing_login)
    assert download.main(cli) == 1
    # main replaces the root handlers, so caplog sees nothing. Read stderr.
    err = capsys.readouterr().err
    assert SECRET not in err
    assert "bad key <redacted>" in err


@pytest.mark.integration
def test_real_download_of_first_pinned_dataset(tmp_path: Path) -> None:
    from frc_xdata.config import DatasetsConfig, load_yaml, roboflow_api_key

    repo = Path(__file__).resolve().parents[1]
    specs = load_yaml(repo / "configs" / "datasets.yaml", DatasetsConfig).datasets
    pinned = {key: spec for key, spec in specs.items() if spec.pinned}
    assert pinned, "pin a dataset version in configs/datasets.yaml first"
    key, spec = next(iter(pinned.items()))
    api_key = roboflow_api_key()
    failures = download_all(
        {key: spec},
        raw_dir=tmp_path / "raw",
        manifests_dir=tmp_path / "manifests",
        fetch=download._roboflow_fetch(api_key),
        errors=download._sdk_errors(),
        secret=api_key,
    )
    assert failures == []
    assert verified_manifest(tmp_path / "raw" / key, spec) is not None
