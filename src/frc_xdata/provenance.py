"""Record where a result came from: git state, file hashes, and package versions."""

import hashlib
import logging
import subprocess
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

CHUNK_BYTES = 1 << 20

logger = logging.getLogger(__name__)


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


def git_tree(repo: Path) -> str:
    """Return the SHA of the tree checked out in ``repo``.

    A rebase merge changes the commit SHA but keeps the tree SHA when the
    content is unchanged, so the tree still identifies the code on main.
    """
    return _git(repo, "rev-parse", "HEAD^{tree}")


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
    """Return every installed distribution and its version, sorted by name.

    A distribution with no ``Name`` in its metadata, which a half-removed
    install can leave behind, is skipped with a warning.
    """
    found = {}
    for dist in distributions():
        names = dist.metadata.get_all("Name") or []
        name = names[0] if names else ""
        if not name:
            logger.warning("skipping an installed distribution with no Name metadata")
            continue
        found[name.lower()] = dist.version
    return dict(sorted(found.items()))


def utc_timestamp() -> str:
    """Return the current UTC time in ISO 8601 with second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")
