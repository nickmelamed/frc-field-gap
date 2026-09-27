import random
import re
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from frc_xdata.config import ProjectConfig, SplitFractions, SplitMethod, SplitsConfig, load_yaml
from frc_xdata.errors import ConfigError, SplitLeakError
from frc_xdata.inspect_datasets import Box, ImageRecord, ImageRef
from frc_xdata.splits import (
    assign_groups,
    check_no_leak,
    copy_groups,
    cut_points,
    grouped_split,
    leak_groups,
    near_test_images,
    pick_copy,
    recording_key,
    split_dataset,
    temporal_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = SplitFractions(train=0.7, valid=0.15, test=0.15)
TEMPORAL = SplitMethod(
    method="temporal", recording_pattern=r"^(?P<recording>.+)_(?P<frame>\d+)\.jpg$"
)
GROUPED = SplitMethod(
    method="grouped", recording_pattern=r"^(?P<recording>.+_mp4)-\d+\.jpg$", dedupe_copies=True
)


def cfg(buffer_frames: int = 2, min_images: int = 5) -> SplitsConfig:
    return SplitsConfig(
        fractions=FRACTIONS,
        buffer_frames=buffer_frames,
        min_recording_images=min_images,
        datasets={},
    )


def record(
    file_name: str,
    source: str | None = None,
    labels: tuple[str, ...] = (),
    split: str = "train",
    size: float = 2,
) -> ImageRecord:
    boxes = tuple(Box(label, 0, 0, size, size) for label in labels)
    return ImageRecord(ImageRef("ds", split, file_name), source or file_name, 64, 64, boxes)


def frames(recording: str, numbers: range | list[int]) -> list[ImageRecord]:
    return [record(f"{recording}_{n}.rf.x.jpg", f"{recording}_{n}.jpg") for n in numbers]


def distinct_hashes(records: list[ImageRecord]) -> dict[ImageRef, int]:
    # Random 64-bit hashes differ in about 32 bits, far above any threshold
    # used here, so images only match where a test plants a match.
    return {r.ref: random.Random(r.ref.file_name).getrandbits(64) for r in records}


def frame_of(ref: ImageRef) -> int:
    return int(ref.file_name.split("_")[1].split(".")[0])


def by_split(result_split: dict[ImageRef, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for ref, s in sorted(result_split.items(), key=lambda kv: kv[0].file_name):
        out[s].append(ref.file_name)
    return out


def test_recording_key_matches_or_returns_none() -> None:
    assert recording_key("FRC_2_mp4-57.jpg", GROUPED.recording_pattern) == "FRC_2_mp4"
    assert recording_key("frame_0324.jpg", GROUPED.recording_pattern) is None


def test_copy_groups_split_photos_that_share_a_name_by_box_counts() -> None:
    records = [
        record("a1", "p.jpg", ("fuel",)),
        record("a2", "p.jpg", ("fuel",)),
        record("b1", "p.jpg", ("robot", "robot")),
        record("v1", "p.jpg", ("fuel",), split="valid"),
    ]
    groups = copy_groups(records)
    assert sorted(sorted(r.ref.file_name for r in g) for g in groups.values()) == [
        ["a1", "a2"],
        ["b1"],
    ]


def test_pick_copy_prefers_the_smallest_boxes_then_the_first_name() -> None:
    rotated = record("a", "p.jpg", ("fuel",), size=4)
    straight = record("b", "p.jpg", ("fuel",), size=2)
    assert pick_copy([rotated, straight]) is straight
    assert pick_copy([record("d", size=2), record("c", size=2)]).ref.file_name == "c"


@pytest.mark.parametrize(("n", "cuts"), [(20, (14, 17)), (3, (1, 2)), (100, (70, 85))])
def test_cut_points_leave_valid_and_test_non_empty(n: int, cuts: tuple[int, int]) -> None:
    assert cut_points(n, FRACTIONS) == cuts


def test_temporal_split_orders_train_valid_test_and_drops_the_buffer() -> None:
    result = temporal_split(frames("run", range(20)), TEMPORAL, cfg(buffer_frames=2))
    got = {
        s: sorted(frame_of(ref) for ref, x in result.split.items() if x == s)
        for s in ("train", "valid", "test")
    }
    assert got == {"train": list(range(14)), "valid": [14, 15, 16], "test": [17, 18, 19]}
    assert sorted(frame_of(ref) for ref in result.dropped) == [12, 13, 15, 16]
    assert set(result.dropped.values()) == {"buffer"}
    assert set(result.unit.values()) == {"run"}


def test_temporal_split_measures_the_buffer_in_frames() -> None:
    result = temporal_split(frames("run", range(0, 200, 10)), TEMPORAL, cfg(buffer_frames=9))
    assert result.dropped == {}


def test_temporal_split_sorts_frames_as_numbers() -> None:
    result = temporal_split(frames("run", [8, 9, 10, 11, 100]), TEMPORAL, cfg(buffer_frames=0))
    assert by_split(result.split)["test"] == ["run_100.rf.x.jpg"]


def test_temporal_split_sends_short_recordings_to_train() -> None:
    records = frames("clip", range(4)) + frames("run", range(20))
    result = temporal_split(records, TEMPORAL, cfg())
    clip = {s for ref, s in result.split.items() if ref.file_name.startswith("clip")}
    assert clip == {"train"}
    assert not any(ref.file_name.startswith("clip") for ref in result.dropped)


def test_temporal_split_rejects_a_name_without_a_frame_number() -> None:
    with pytest.raises(ConfigError, match=r"odd\.jpg"):
        temporal_split([record("odd.jpg")], TEMPORAL, cfg())


def test_leak_groups_join_by_name_recording_and_hash() -> None:
    records = [
        record("a", "same.jpg"),
        record("b", "same.jpg"),
        record("c", "FRC_2_mp4-1.jpg"),
        record("d", "FRC_2_mp4-9.jpg"),
        record("e"),
        record("f"),
        record("g"),
    ]
    hashes = [0xFFFF, 0xFFFF0000, 0xFFFF00000000, 0xFFFF << 48, 0b1111, 0b0111, 0xF0F0F0F0]
    groups = leak_groups(records, hashes, GROUPED.recording_pattern, max_distance=1)
    assert groups == [[0, 1], [2, 3], [4, 5], [6]]


def test_assign_groups_hits_the_fractions_and_ignores_input_order() -> None:
    groups = [[f"img{i:03d}"] for i in range(100)]
    splits = assign_groups(groups, FRACTIONS, "seed")
    assert Counter(splits) == {"train": 70, "valid": 15, "test": 15}
    shuffled = groups[:]
    random.Random(0).shuffle(shuffled)
    again = dict(zip(map(tuple, shuffled), assign_groups(shuffled, FRACTIONS, "seed"), strict=True))
    assert again == dict(zip(map(tuple, groups), splits, strict=True))


def test_assign_groups_seed_changes_ties_but_not_totals() -> None:
    groups = [[f"img{i:03d}"] for i in range(100)]
    a = assign_groups(groups, FRACTIONS, "one")
    b = assign_groups(groups, FRACTIONS, "two")
    assert a != b
    assert Counter(a) == Counter(b)


def test_assign_groups_places_the_largest_group_first_in_train() -> None:
    groups = [[f"big{i}" for i in range(50)], *([f"s{i}"] for i in range(50))]
    assert assign_groups(groups, FRACTIONS, "seed")[0] == "train"


def test_grouped_split_keeps_one_copy_and_never_splits_a_group() -> None:
    records = [
        record("p1", "p.jpg", ("fuel",), size=3),
        record("p2", "p.jpg", ("fuel",), size=2),
        record("p3", "p.jpg", ("fuel",), size=4),
        *(record(f"v{i}", f"FRC_2_mp4-{i}.jpg", split="valid") for i in range(5)),
        *(record(f"x{i}") for i in range(14)),
    ]
    hashes = distinct_hashes(records)
    result = grouped_split(records, hashes, GROUPED, cfg(), max_distance=4, seed="s")
    assert result.dropped == {records[0].ref: "copy", records[2].ref: "copy"}
    assert records[1].ref in result.split
    video = {result.split[r.ref] for r in records if r.ref.file_name.startswith("v")}
    assert len(video) == 1
    assert len({result.unit[r.ref] for r in records if r.ref.file_name.startswith("v")}) == 1


def test_grouped_split_is_the_same_for_shuffled_input() -> None:
    records = [record(f"x{i:02d}", labels=("fuel",) * (i % 3)) for i in range(40)]
    hashes = distinct_hashes(records)
    first = grouped_split(records, hashes, GROUPED, cfg(), max_distance=4, seed="s")
    shuffled = records[:]
    random.Random(1).shuffle(shuffled)
    second = grouped_split(shuffled, hashes, GROUPED, cfg(), max_distance=4, seed="s")
    assert first == second


def test_split_dataset_drops_train_images_near_a_test_image() -> None:
    records = frames("run", range(20))
    hashes = distinct_hashes(records)
    first_test = next(r for r in records if frame_of(r.ref) == 17)
    early = next(r for r in records if frame_of(r.ref) == 0)
    hashes[early.ref] = hashes[first_test.ref] ^ 0b11
    result = split_dataset(records, hashes, TEMPORAL, cfg(buffer_frames=0), 4, "s")
    assert result.dropped == {early.ref: "near_test"}
    assert early.ref not in result.split
    check_no_leak(result.split, hashes, 4)


def test_split_dataset_puts_every_image_in_one_place() -> None:
    records = frames("run", range(30)) + frames("clip", range(3))
    result = split_dataset(records, distinct_hashes(records), TEMPORAL, cfg(), 4, "s")
    assert set(result.split) | set(result.dropped) == {r.ref for r in records}
    assert not set(result.split) & set(result.dropped)


def test_check_no_leak_raises_on_a_planted_leak() -> None:
    train, test = ImageRef("ds", "train", "a"), ImageRef("ds", "test", "b")
    split = {train: "train", test: "test"}
    assert near_test_images(split, {train: 0b1, test: 0b0}, 1) == [train]
    with pytest.raises(SplitLeakError, match="1 images"):
        check_no_leak(split, {train: 0b1, test: 0b0}, 1)
    check_no_leak(split, {train: 0xFF, test: 0}, 1)


def test_committed_split_settings_load() -> None:
    project = load_yaml(REPO_ROOT / "configs" / "project.yaml", ProjectConfig)
    methods = {k: m.method for k, m in project.splits.datasets.items()}
    assert methods == {"marswars": "temporal", "robotzftp2": "temporal", "scorekeeper": "grouped"}


def test_split_settings_need_three_images_to_cut_a_recording() -> None:
    with pytest.raises(ValidationError):
        cfg(min_images=2)


def test_temporal_split_sends_train_only_recordings_to_train() -> None:
    method = TEMPORAL.model_copy(update={"train_only_pattern": "^official"})
    records = frames("official-video", range(30)) + frames("run", range(30))
    result = temporal_split(records, method, cfg())
    official = {s for ref, s in result.split.items() if ref.file_name.startswith("official")}
    assert official == {"train"}
    assert not any(ref.file_name.startswith("official") for ref in result.dropped)
    assert "test" in result.split.values()


def test_committed_train_only_pattern_covers_the_official_videos() -> None:
    project = load_yaml(REPO_ROOT / "configs" / "project.yaml", ProjectConfig)
    pattern = project.splits.datasets["marswars"].train_only_pattern
    assert pattern is not None
    for recording, official in [
        ("2026-FIRST-Robotics-Competition-Additional-Field-Interactions_mp4", True),
        ("2026-FIRST-Robotics-Competition-Field-Tour_-Hub_mp4", True),
        ("Basler_daA1280-54uc__24770352__20260112_181311364", False),
        ("IMG_8069_MOV", False),
    ]:
        assert bool(re.match(pattern, recording)) is official
