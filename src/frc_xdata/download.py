"""Download pinned Universe datasets in COCO format and record their hashes.

Each dataset lands in ``data/raw/<key>/`` with a ``MANIFEST.json`` listing
every file, its size, and its SHA256. A short ``<key>.sha256`` digest in
``reports/data_manifests/`` is committed and rewritten on every run, so drift
in the source data shows up in ``git diff`` down to the annotation file or
image directory that changed.

With ``--resolve`` nothing is downloaded. Each candidate is looked up on
Universe instead, and its versions, split sizes, and classes are written to
``reports/dataset_resolution.json`` so versions can be pinned by hand.
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, ValidationError

from frc_xdata.config import (
    DatasetsConfig,
    DatasetSpec,
    ProjectConfig,
    load_yaml,
    roboflow_api_key,
)
from frc_xdata.logging_utils import add_log_level_argument, setup_logging
from frc_xdata.provenance import sha256_file, utc_timestamp

MANIFEST_NAME = "MANIFEST.json"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
RESOLUTION_NAME = "dataset_resolution.json"
FAILURES_NAME = "download_failures.json"
PROJECT_CONFIG = Path("configs/project.yaml")
DATASETS_CONFIG = Path("configs/datasets.yaml")
REDACTED = "<redacted>"

logger = logging.getLogger(__name__)

Fetch = Callable[[DatasetSpec, Path], None]


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ManifestEntry(_Record):
    """One file in a downloaded dataset, relative to the dataset directory."""

    path: str
    size: int
    sha256: str


class Manifest(_Record):
    """What was downloaded for one dataset, and the hash of every file."""

    key: str
    workspace: str
    project: str
    version: int
    format: str
    created: str
    files: list[ManifestEntry]

    def matches(self, spec: DatasetSpec) -> bool:
        """Return True if this manifest was made from the dataset ``spec`` pins."""
        return (self.workspace, self.project, self.version, self.format) == (
            spec.workspace,
            spec.project,
            spec.version,
            spec.format,
        )


class Failure(_Record):
    """A dataset that could not be downloaded or resolved, and why."""

    key: str
    reason: str


def redact(text: str, secret: str) -> str:
    """Return ``text`` with every occurrence of ``secret`` replaced.

    The Roboflow SDK puts the API key in request URLs, so network errors can
    carry it in their message.
    """
    return text.replace(secret, REDACTED) if secret else text


class RedactFilter(logging.Filter):
    """Remove a secret from every log record that passes through a handler.

    Installed on the root handlers, so it also covers records from other
    libraries, such as urllib3 logging request URLs at DEBUG.
    """

    def __init__(self, secret: str) -> None:
        """Keep the secret to remove."""
        super().__init__()
        self.secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record's message without the secret, and keep it."""
        message = record.getMessage()
        if self.secret and self.secret in message:
            record.msg = redact(message, self.secret)
            record.args = None
        return True


def build_manifest(root: Path, key: str, spec: DatasetSpec) -> Manifest:
    """Hash every file under ``root`` except an existing manifest.

    Raises:
        ValueError: If ``spec`` is not pinned to a project and version.
    """
    if spec.project is None or spec.version is None:
        raise ValueError(f"{key} is not pinned to a project and version")
    files = [
        ManifestEntry(
            path=path.relative_to(root).as_posix(),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / MANIFEST_NAME
    ]
    return Manifest(
        key=key,
        workspace=spec.workspace,
        project=spec.project,
        version=spec.version,
        format=spec.format,
        created=utc_timestamp(),
        files=files,
    )


def verify_manifest(root: Path, manifest: Manifest) -> list[str]:
    """Return every difference between ``root`` and ``manifest``, or nothing.

    Missing, changed, and unlisted files are all reported, since any of them
    means the local copy is not the dataset the manifest describes.
    """
    problems = []
    listed = {entry.path for entry in manifest.files}
    for entry in manifest.files:
        path = root / entry.path
        if not path.is_file():
            problems.append(f"missing: {entry.path}")
        elif path.stat().st_size != entry.size or sha256_file(path) != entry.sha256:
            problems.append(f"changed: {entry.path}")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and rel != MANIFEST_NAME and rel not in listed:
            problems.append(f"unlisted: {rel}")
    return problems


def digest_lines(manifest: Manifest) -> str:
    """Return a short digest of the manifest, small enough to commit.

    Files other than images get one ``sha256sum`` line each, so annotation
    files can be checked with ``sha256sum -c``. The images in each directory
    collapse into one comment line holding their count and the SHA256 of
    their ``sha256sum`` lines, sorted by path. A per-image list for the
    larger datasets would run to megabytes.
    """
    lines = []
    image_lines: dict[str, list[str]] = {}
    for entry in manifest.files:
        line = f"{entry.sha256}  {entry.path}\n"
        path = PurePosixPath(entry.path)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image_lines.setdefault(f"{path.parent}/", []).append(line)
        else:
            lines.append(line)
    for directory, group in sorted(image_lines.items()):
        combined = hashlib.sha256("".join(group).encode()).hexdigest()
        lines.append(f"# images {directory}: {len(group)} files, sha256 {combined}\n")
    return "".join(lines)


def read_manifest(root: Path) -> Manifest | None:
    """Return the manifest in ``root``, or None if it is missing or unreadable."""
    path = root / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as e:
        logger.warning("ignoring unreadable manifest %s: %s", path, e)
        return None


def verified_manifest(root: Path, spec: DatasetSpec) -> Manifest | None:
    """Return the manifest if ``root`` holds exactly the pinned dataset, else None."""
    manifest = read_manifest(root)
    if manifest is None or not manifest.matches(spec):
        return None
    problems = verify_manifest(root, manifest)
    for problem in problems:
        logger.info("%s: %s", root.name, problem)
    return None if problems else manifest


def _write_digest(manifests_dir: Path, manifest: Manifest) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{manifest.key}.sha256").write_text(digest_lines(manifest), encoding="utf-8")


def download_all(
    datasets: dict[str, DatasetSpec],
    raw_dir: Path,
    manifests_dir: Path,
    fetch: Fetch,
    errors: tuple[type[Exception], ...],
    secret: str = "",
    force: bool = False,
) -> list[Failure]:
    """Download every pinned dataset that is not already complete on disk.

    Each download goes to a staging directory and is moved into place only
    after its manifest is written, so an interrupted run never leaves a
    directory that looks finished. A failure is recorded and the remaining
    datasets still run.

    Args:
        datasets: Specs by key.
        raw_dir: Parent of the per-dataset directories.
        manifests_dir: Where the committed ``<key>.sha256`` files go.
        fetch: Writes one dataset's files into the directory it is given.
        errors: Exceptions from ``fetch`` that mean this dataset failed.
            Anything else stops the run.
        secret: Removed from failure reasons before they are logged.
        force: Download again even if the local copy matches its manifest.

    Returns:
        The datasets that failed, in the order they were tried.
    """
    failures = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key, spec in datasets.items():
        dest = raw_dir / key
        if not spec.pinned:
            failures.append(Failure(key=key, reason="project or version not pinned"))
            logger.warning("%s: skipped, project or version not pinned", key)
            continue
        existing = None if force else verified_manifest(dest, spec)
        if existing is not None:
            logger.info("%s: already downloaded and verified", key)
            _write_digest(manifests_dir, existing)
            continue
        staging = raw_dir / f".{key}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        logger.info("%s: downloading %s version %s", key, spec.url, spec.version)
        try:
            fetch(spec, staging)
            manifest = build_manifest(staging, key, spec)
            (staging / MANIFEST_NAME).write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            shutil.rmtree(dest, ignore_errors=True)
            staging.rename(dest)
            _write_digest(manifests_dir, manifest)
        except errors as e:
            reason = redact(f"{type(e).__name__}: {e}", secret)
            failures.append(Failure(key=key, reason=reason))
            logger.error("%s: download failed: %s", key, reason)
            shutil.rmtree(staging, ignore_errors=True)
            continue
        logger.info("%s: %d files", key, len(manifest.files))
    return failures


class _ApiProject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    images: int = 0
    classes: dict[str, int] = {}
    splits: dict[str, int] = {}


class _ApiVersion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    images: int = 0
    splits: dict[str, int] = {}


class _ApiProjectResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project: _ApiProject
    versions: list[_ApiVersion] = []


class _ApiWorkspaceProject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str = ""
    images: int = 0
    classes: dict[str, int] = {}


class _ApiWorkspace(BaseModel):
    model_config = ConfigDict(extra="ignore")

    projects: list[_ApiWorkspaceProject] = []


class _ApiWorkspaceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workspace: _ApiWorkspace


class VersionSummary(_Record):
    """One published version of a project."""

    version: int
    images: int
    splits: dict[str, int]


class ProjectSummary(_Record):
    """What Universe reports for one project, used to pin a version."""

    workspace: str
    project: str
    images: int
    classes: dict[str, int]
    latest_version: int | None
    versions: list[VersionSummary]


class Resolution(_Record):
    """The lookup result for one dataset key."""

    key: str
    projects: list[ProjectSummary]


def summarize_project(workspace: str, project: str, response: Any) -> ProjectSummary:
    """Summarize a project lookup response from the Roboflow API.

    Raises:
        ValueError: If the response does not have the expected shape, which
            includes pydantic's ValidationError.
    """
    parsed = _ApiProjectResponse.model_validate(response)
    versions = sorted(
        (
            VersionSummary(
                version=int(v.id.rsplit("/", 1)[-1]),
                images=v.images,
                splits=v.splits,
            )
            for v in parsed.versions
        ),
        key=lambda v: v.version,
    )
    return ProjectSummary(
        workspace=workspace,
        project=project,
        images=parsed.project.images,
        classes=dict(sorted(parsed.project.classes.items())),
        latest_version=versions[-1].version if versions else None,
        versions=versions,
    )


def workspace_projects(response: Any) -> list[str]:
    """Return the project slugs in a workspace lookup response, sorted.

    Raises:
        ValidationError: If the response does not have the expected shape.
    """
    parsed = _ApiWorkspaceResponse.model_validate(response)
    return sorted(p.id.rsplit("/", 1)[-1] for p in parsed.workspace.projects)


def resolve_all(
    datasets: dict[str, DatasetSpec],
    get_project: Callable[[str, str], Any],
    get_workspace: Callable[[str], Any],
    errors: tuple[type[Exception], ...],
    secret: str = "",
) -> tuple[list[Resolution], list[Failure]]:
    """Look up every dataset, or every project in its workspace if unset.

    Args:
        datasets: Specs by key.
        get_project: Returns the API response for a workspace and project.
        get_workspace: Returns the API response for a workspace.
        errors: Exceptions from the lookups that mean this dataset failed.
        secret: Removed from failure reasons before they are logged.

    Returns:
        The resolved datasets and the ones that failed.
    """
    # ValueError also covers pydantic's ValidationError when the API response
    # has a shape this code does not expect.
    caught: tuple[type[Exception], ...] = (*errors, ValueError)
    resolved = []
    failures = []
    for key, spec in datasets.items():
        try:
            if spec.project is None:
                slugs = workspace_projects(get_workspace(spec.workspace))
                logger.info("%s: workspace %s has %s", key, spec.workspace, slugs)
            else:
                slugs = [spec.project]
            projects = [
                summarize_project(spec.workspace, slug, get_project(spec.workspace, slug))
                for slug in slugs
            ]
        except caught as e:
            reason = redact(f"{type(e).__name__}: {e}", secret)
            failures.append(Failure(key=key, reason=reason))
            logger.error("%s: lookup failed: %s", key, reason)
            continue
        for p in projects:
            logger.info(
                "%s: %s/%s latest version %s, %d images, classes %s",
                key,
                p.workspace,
                p.project,
                p.latest_version,
                p.images,
                sorted(p.classes),
            )
        resolved.append(Resolution(key=key, projects=projects))
    return resolved, failures


def _write_json(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump(mode="json") for r in records]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _sdk_errors() -> tuple[type[Exception], ...]:
    from roboflow.adapters.rfapi import RoboflowError

    return (RoboflowError, RuntimeError, requests.RequestException, OSError, zipfile.BadZipFile)


def _roboflow_fetch(api_key: str) -> Fetch:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    def fetch(spec: DatasetSpec, dest: Path) -> None:
        project = rf.project(f"{spec.workspace}/{spec.project}")
        version = project.version(spec.version)
        # overwrite=True because the SDK otherwise returns early whenever the
        # directory exists, even if an earlier download stopped halfway.
        version.download(spec.format, location=str(dest), overwrite=True)

    return fetch


def _select(datasets: DatasetsConfig, only: list[str] | None) -> dict[str, DatasetSpec]:
    if not only:
        return dict(datasets.datasets)
    unknown = sorted(set(only) - set(datasets.datasets))
    if unknown:
        raise SystemExit(f"unknown dataset keys: {', '.join(unknown)}")
    return {key: datasets.datasets[key] for key in only}


def main(argv: list[str] | None = None) -> int:
    """Run the download command line and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-config", type=Path, default=PROJECT_CONFIG)
    parser.add_argument("--datasets-config", type=Path, default=DATASETS_CONFIG)
    parser.add_argument(
        "--only", action="append", metavar="KEY", help="limit to this dataset key (repeatable)"
    )
    parser.add_argument("--resolve", action="store_true", help="look datasets up, download none")
    parser.add_argument("--force", action="store_true", help="download even if already verified")
    add_log_level_argument(parser)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    project = load_yaml(args.project_config, ProjectConfig)
    datasets = _select(load_yaml(args.datasets_config, DatasetsConfig), args.only)
    api_key = roboflow_api_key()
    for handler in logging.getLogger().handlers:
        handler.addFilter(RedactFilter(api_key))
    errors = _sdk_errors()
    reports = project.paths.reports_dir

    try:
        if args.resolve:
            from roboflow.adapters import rfapi

            resolved, failures = resolve_all(
                datasets,
                get_project=lambda ws, proj: rfapi.get_project(api_key, ws, proj),
                get_workspace=lambda ws: rfapi.get_workspace(api_key, ws),
                errors=errors,
                secret=api_key,
            )
            _write_json(reports / RESOLUTION_NAME, resolved)
            logger.info("wrote %s", reports / RESOLUTION_NAME)
        else:
            failures = download_all(
                datasets,
                raw_dir=project.paths.raw_dir,
                manifests_dir=project.paths.manifests_dir,
                fetch=_roboflow_fetch(api_key),
                errors=errors,
                secret=api_key,
                force=args.force,
            )
            _write_json(reports / FAILURES_NAME, failures)
    except errors as e:
        # Reached when the SDK fails before any dataset is tried, such as
        # while checking the key. Log it redacted instead of a traceback,
        # which could print the key inside a request URL.
        reason = redact(f"{type(e).__name__}: {e}", api_key)
        logger.error("roboflow failed: %s", reason)
        if not args.resolve:
            # Replace the last run's report, which would otherwise look current.
            _write_json(reports / FAILURES_NAME, [Failure(key="*", reason=reason)])
        return 1

    for failure in failures:
        logger.error("%s: %s", failure.key, failure.reason)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
