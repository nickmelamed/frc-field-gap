"""Record where a result came from: git state, file hashes, and package versions."""

import hashlib
import subprocess
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

CHUNK_BYTES = 1 << 20


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def git_commit(repo: Path) -> str:
    """Return the full SHA of the commit checked out in ``repo``.

    Raises:
        subprocess.CalledProcessError: If ``repo`` is not a git repository or
            has no commits.
    """
    return _git(repo, "rev-parse", "HEAD")


def git_is_dirty(repo: Path) -> bool:
    """Return True if ``repo`` has uncommitted changes or untracked files."""
    return bool(_git(repo, "status", "--porcelain"))


def sha256_file(path: Path) -> str:
    """Return the hex SHA256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Return every installed distribution and its version, sorted by name."""
    found = {d.metadata["Name"].lower(): d.version for d in distributions()}
    return dict(sorted(found.items()))


def utc_timestamp() -> str:
    """Return the current UTC time in ISO 8601 with second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")
