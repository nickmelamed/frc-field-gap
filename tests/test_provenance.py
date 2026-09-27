import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from frc_xdata.provenance import (
    git_commit,
    git_is_dirty,
    package_versions,
    sha256_file,
    utc_timestamp,
)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    run_git(tmp_path, "init", "-q")
    run_git(tmp_path, "config", "user.email", "test@example.com")
    run_git(tmp_path, "config", "user.name", "Test")
    run_git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    run_git(tmp_path, "add", "a.txt")
    run_git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_git_commit_returns_head_sha(repo: Path) -> None:
    sha = git_commit(repo)
    assert sha == run_git(repo, "rev-parse", "HEAD")
    assert len(sha) == 40


def test_git_commit_outside_repo_raises(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        git_commit(tmp_path)


def test_git_is_dirty_tracks_changes(repo: Path) -> None:
    assert not git_is_dirty(repo)
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    assert git_is_dirty(repo)


def test_git_is_dirty_counts_untracked_files(repo: Path) -> None:
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    assert git_is_dirty(repo)


def test_package_versions_includes_dependencies() -> None:
    versions = package_versions()
    assert "pydantic" in versions
    assert list(versions) == sorted(versions)


def test_utc_timestamp_is_timezone_aware() -> None:
    parsed = datetime.fromisoformat(utc_timestamp())
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0
