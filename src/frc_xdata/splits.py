"""Re-split datasets so no scene lands on both sides of a train/test split.

Two methods, chosen per dataset in ``configs/project.yaml`` (see D-013).
``temporal`` cuts each recording in frame order into train, valid, and test,
so valid sits between train and test in time, and drops frames within a
buffer of each cut. ``grouped`` joins images that share a recording, a
source name, or a near-duplicate perceptual hash, and assigns whole groups.
A final pass drops any train or valid image that is a near duplicate of a
test image. Everything here is pure, and the caller supplies the hashes.
"""

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from frc_xdata.config import SplitFractions, SplitMethod, SplitsConfig
from frc_xdata.errors import ConfigError, SplitLeakError
from frc_xdata.inspect_datasets import ImageRecord, ImageRef, near_duplicate_pairs

Split = Literal["train", "valid", "test"]
SPLITS: tuple[Split, ...] = ("train", "valid", "test")
DropReason = Literal["copy", "buffer", "near_test"]
DROP_REASONS: tuple[DropReason, ...] = ("copy", "buffer", "near_test")


@dataclass(frozen=True)
class SplitResult:
    """Where each image of one dataset went.

    ``unit`` names the recording (temporal) or group (grouped) an image
    belongs to, for counting how many independent scenes each split holds.
    """

    split: dict[ImageRef, Split]
    unit: dict[ImageRef, str]
    dropped: dict[ImageRef, DropReason]


def recording_key(name: str, pattern: str) -> str | None:
    """Return the recording a source name belongs to, or None if it names none."""
    match = re.match(pattern, name)
    return match["recording"] if match else None


def copy_groups(records: Sequence[ImageRecord]) -> dict[str, list[ImageRecord]]:
    """Group the augmented copies of each photo in the train split.

    Copies share a source name, but so do some different photos, and random
    rotation and exposure changes move copies too far apart for a perceptual
    hash to match them (D-014). Copies do keep the same number of boxes of
    each class, so the name and those counts together identify a photo.
    Only train is augmented, so valid and test images are left out.
    """
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.ref.split == "train":
            counts = sorted(Counter(b.label for b in r.boxes).items())
            groups[f"{r.source_name}|{counts}"].append(r)
    return dict(groups)


def pick_copy(copies: Sequence[ImageRecord]) -> ImageRecord:
    """Return the copy with the least total box area, then the first file name.

    Rotating a box and taking its axis-aligned hull only makes it larger,
    so the smallest total area is the least rotated copy.
    """
    return min(copies, key=lambda r: (sum(b.w * b.h for b in r.boxes), r.ref.file_name))


def cut_points(n: int, fractions: SplitFractions) -> tuple[int, int]:
    """Return indices ``(a, b)`` so ``[:a]`` is train, ``[a:b]`` valid, ``[b:]`` test.

    Valid and test get at least one image each, so ``n`` must be at least 3.
    """
    a = min(round(n * fractions.train), n - 2)
    b = min(max(a + round(n * fractions.valid), a + 1), n - 1)
    return a, b


def temporal_split(
    records: Sequence[ImageRecord], method: SplitMethod, cfg: SplitsConfig
) -> SplitResult:
    """Cut each recording in frame order and drop frames next to each cut.

    A recording with fewer than ``cfg.min_recording_images`` images, or whose
    name matches the method's ``train_only_pattern``, goes whole to train.
    Otherwise an image is dropped when the first frame of the next split is
    at most ``cfg.buffer_frames`` frames after it.

    Raises:
        ConfigError: If a source name does not match the recording pattern,
            since every image needs a place in frame order.
    """
    by_recording: dict[str, list[tuple[int, ImageRecord]]] = defaultdict(list)
    for r in records:
        match = re.match(method.recording_pattern, r.source_name)
        if match is None:
            raise ConfigError(f"{r.ref}: {r.source_name!r} does not match the recording pattern")
        by_recording[match["recording"]].append((int(match["frame"]), r))

    split: dict[ImageRef, Split] = {}
    unit: dict[ImageRef, str] = {}
    dropped: dict[ImageRef, DropReason] = {}
    for recording, frames in sorted(by_recording.items()):
        frames.sort(key=lambda fr: (fr[0], fr[1].ref.file_name))
        n = len(frames)
        train_only = method.train_only_pattern is not None and re.match(
            method.train_only_pattern, recording
        )
        whole = n < cfg.min_recording_images or train_only
        a, b = (n, n) if whole else cut_points(n, cfg.fractions)
        for i, (frame, r) in enumerate(frames):
            unit[r.ref] = recording
            if i < a:
                split[r.ref], next_start = "train", a
            elif i < b:
                split[r.ref], next_start = "valid", b
            else:
                split[r.ref], next_start = "test", n
            if next_start < n and frames[next_start][0] - frame <= cfg.buffer_frames:
                dropped[r.ref] = "buffer"
    return SplitResult(split, unit, dropped)


def leak_groups(
    records: Sequence[ImageRecord],
    hashes: Sequence[int],
    pattern: str,
    max_distance: int,
) -> list[list[int]]:
    """Return indices of ``records`` grouped so related images stay together.

    Two images are related when they share a recording, share a source
    name, or have perceptual hashes within ``max_distance`` bits. Joining
    too much only makes the split coarser, so shared names join even when
    they belong to different photos.
    """
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    first: dict[str, int] = {}
    for i, r in enumerate(records):
        keys = [f"name:{r.source_name}"]
        if (recording := recording_key(r.source_name, pattern)) is not None:
            keys.append(f"recording:{recording}")
        for key in keys:
            if key in first:
                join(i, first[key])
            else:
                first[key] = i
    for i, j in near_duplicate_pairs(np.array(hashes, dtype=np.uint64), max_distance):
        join(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    return sorted(groups.values())


def _tie_key(seed: str, names: Sequence[str]) -> str:
    return hashlib.sha256(f"{seed}:{min(names)}".encode()).hexdigest()


def assign_groups(
    groups: Sequence[Sequence[str]], fractions: SplitFractions, seed: str
) -> list[Split]:
    """Assign each group of file names to a split, largest groups first.

    Each group goes to the split furthest below its target share of images.
    Groups of equal size are ordered by a hash of the seed and their first
    file name, so the result depends only on the groups, not their order.
    """
    targets = {s: getattr(fractions, s) * sum(len(g) for g in groups) for s in SPLITS}
    filled: Counter[Split] = Counter()
    order = sorted(range(len(groups)), key=lambda g: (-len(groups[g]), _tie_key(seed, groups[g])))
    assigned: dict[int, Split] = {}
    for g in order:
        best = max(SPLITS, key=lambda s: ((targets[s] - filled[s]) / targets[s], -SPLITS.index(s)))
        assigned[g] = best
        filled[best] += len(groups[g])
    return [assigned[g] for g in range(len(groups))]


def grouped_split(
    records: Sequence[ImageRecord],
    hashes: Mapping[ImageRef, int],
    method: SplitMethod,
    cfg: SplitsConfig,
    max_distance: int,
    seed: str,
) -> SplitResult:
    """Dedupe copies if configured, then assign leak groups to splits."""
    dropped: dict[ImageRef, DropReason] = {}
    if method.dedupe_copies:
        for copies in copy_groups(records).values():
            keep = pick_copy(copies)
            dropped.update({c.ref: "copy" for c in copies if c is not keep})
    kept = [r for r in records if r.ref not in dropped]
    kept_hashes = [hashes[r.ref] for r in kept]
    groups = leak_groups(kept, kept_hashes, method.recording_pattern, max_distance)
    names = [[kept[i].ref.file_name for i in g] for g in groups]
    split: dict[ImageRef, Split] = {}
    unit: dict[ImageRef, str] = {}
    for g, s in zip(groups, assign_groups(names, cfg.fractions, seed), strict=True):
        label = min(kept[i].ref.file_name for i in g)
        for i in g:
            split[kept[i].ref] = s
            unit[kept[i].ref] = label
    return SplitResult(split, unit, dropped)


def near_test_images(
    split: Mapping[ImageRef, str], hashes: Mapping[ImageRef, int], max_distance: int
) -> list[ImageRef]:
    """Return train and valid images within ``max_distance`` bits of a test image."""
    test = np.array([hashes[ref] for ref, s in split.items() if s == "test"], dtype=np.uint64)
    if test.size == 0:
        return []
    return [
        ref
        for ref, s in split.items()
        if s != "test" and np.bitwise_count(np.uint64(hashes[ref]) ^ test).min() <= max_distance
    ]


def check_no_leak(
    split: Mapping[ImageRef, str], hashes: Mapping[ImageRef, int], max_distance: int
) -> None:
    """Raise if any train or valid image is a near duplicate of a test image.

    Raises:
        SplitLeakError: If one is found. The message gives the count and the
            first leaking image.
    """
    if leaks := near_test_images(split, hashes, max_distance):
        raise SplitLeakError(f"{len(leaks)} images near a test image, first {leaks[0]}")


def split_dataset(
    records: Sequence[ImageRecord],
    hashes: Mapping[ImageRef, int],
    method: SplitMethod,
    cfg: SplitsConfig,
    max_distance: int,
    seed: str,
) -> SplitResult:
    """Split one dataset with its configured method, then drop images near test."""
    if method.method == "temporal":
        result = temporal_split(records, method, cfg)
    else:
        result = grouped_split(records, hashes, method, cfg, max_distance, seed)
    split = {ref: s for ref, s in result.split.items() if ref not in result.dropped}
    dropped = dict(result.dropped)
    for ref in near_test_images(split, hashes, max_distance):
        dropped[ref] = "near_test"
        del split[ref]
    return SplitResult(split, result.unit, dropped)
