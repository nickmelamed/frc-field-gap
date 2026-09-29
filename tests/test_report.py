import json
import logging
import re
import shutil
from pathlib import Path

import pytest

from frc_xdata.config import AreaBuckets
from frc_xdata.diagnose import (
    ClassDiagnosis,
    Diagnosis,
    FeatureSummary,
    GalleryTile,
    ReviewCount,
    SizeRow,
    SliceScore,
    Slicing,
    SplitDiagnosis,
)
from frc_xdata.errors import ConfigError
from frc_xdata.evaluate import (
    BootstrapResult,
    ClassInterval,
    ClassMetrics,
    ConfusionTable,
    EvalMetrics,
    Interval,
    PerImageLimit,
    PRPoint,
    RunResult,
    UnitCount,
)
from frc_xdata.report import (
    DIAGNOSIS_MARKERS,
    EVALUATION_MARKERS,
    RESULTS_MARKERS,
    PublishedRun,
    diagnosis_section,
    diagnosis_sections,
    evaluation_sections,
    load_runs,
    main,
    replace_between,
    results_table,
    run_section,
    select_runs,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# The same pattern and exemption scripts/agent/check_numbers.py uses to find
# reported numbers.
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?%|\d+\.\d+)(?![\w.]*\d)")
NOT_A_RESULT = "numbers: ok"


def result(
    run_id: str,
    *,
    model: str = "m",
    dataset: str = "alpha",
    limit: int | None = None,
    recall: float = 0.875,
    units: int = 2,
) -> RunResult:
    fuel = ClassMetrics(
        name="fuel",
        instances=8,
        map50=0.912,
        map50_95=0.604,
        precision=None if recall == 0 else 0.933,
        recall=recall,
        true_positives=7,
        false_positives=1,
        false_negatives=1,
    )
    return RunResult(
        run_id=run_id,
        model=model,
        dataset=dataset,
        split="test",
        limit=limit,
        predictions_from="this run",
        scored_classes=["fuel"],
        metrics=EvalMetrics(
            images=5,
            confidence=0.5,
            iou=0.5,
            map50=0.912,
            map50_95=0.604,
            small_map50_95=0.311,
            medium_map50_95=0.702,
            large_map50_95=None,
            classes=[fuel],
            confusion=ConfusionTable(labels=["fuel", "background"], matrix=[[7, 1], [1, 0]]),
            pr_curve={
                "fuel": [
                    PRPoint(threshold=0.25, precision=0.857, recall=0.875),
                    PRPoint(threshold=0.95, precision=None, recall=0.0),
                ]
            },
        ),
        units=[
            UnitCount(unit=f"rec{i}", images=5 - i if i < 5 else 1, boxes={"fuel": 20 - i})
            for i in range(units)
        ],
        bootstrap=BootstrapResult(
            units=units,
            resamples=100,
            level=0.95,
            seed=1,
            classes=[
                ClassInterval(
                    name="fuel",
                    map50=Interval(low=0.801, high=0.977),
                    recall=Interval(low=0.714, high=1.0),
                )
            ],
        ),
    )


def write_run(
    runs_dir: Path, r: RunResult, created: str, dirty: bool = False, source_dirty: bool = False
) -> None:
    run_dir = runs_dir / r.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
    meta = {
        "created": created,
        "git": {"commit": "abc123", "dirty": dirty},
        "source_dirty": source_dirty,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_select_runs_keeps_the_newest_clean_whole_split_run(tmp_path: Path) -> None:
    write_run(tmp_path, result("old", recall=0.5), "2026-09-27T10:00:00+00:00")
    write_run(tmp_path, result("new"), "2026-09-27T11:00:00+00:00")
    write_run(tmp_path, result("dirty"), "2026-09-27T12:00:00+00:00", dirty=True)
    write_run(tmp_path, result("slice", limit=3), "2026-09-27T13:00:00+00:00")
    write_run(
        tmp_path, result("rescored", limit=None), "2026-09-27T14:00:00+00:00", source_dirty=True
    )
    write_run(tmp_path, result("other", dataset="beta"), "2026-09-27T09:00:00+00:00")
    chosen = select_runs(load_runs(tmp_path))
    assert [r.result.run_id for r in chosen] == ["new", "other"]


def test_load_runs_skips_a_run_without_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_run(tmp_path, result("good"), "2026-09-27T10:00:00+00:00")
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "metrics.json").write_text(result("half").model_dump_json(), "utf-8")
    with caplog.at_level(logging.WARNING):
        runs = load_runs(tmp_path)
    assert [r.result.run_id for r in runs] == ["good"]
    assert "half" in caplog.text


def test_load_runs_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "none") == []


def clean(r: RunResult) -> PublishedRun:
    return PublishedRun(result=r, created="2026-09-27T10:00:00+00:00", dirty=False)


def published(r: RunResult) -> list[PublishedRun]:
    return select_runs([clean(r)])


def test_results_table_has_one_row_per_scored_class() -> None:
    table = results_table(published(result("r1")))
    header, rule, row, _, note = table.splitlines()
    assert header.count("|") == rule.count("|") == row.count("|")
    assert row == (
        "| m | alpha | test | 5 | fuel | 0.912 | 0.801 to 0.977 | 0.604 | 0.5 | 0.933 | "
        "0.875 | 0.714 to 1.0 |"
    )
    assert note.startswith("Intervals hold the middle 95% of scores")


def test_undefined_precision_is_shown_as_not_available() -> None:
    row = results_table(published(result("r1", recall=0.0))).splitlines()[2]
    assert "| n/a |" in row


def test_tables_are_placeholders_without_runs() -> None:
    assert results_table([]).startswith("TBD")
    assert evaluation_sections([]).startswith("No model")


def test_every_reported_number_is_stored_in_the_run() -> None:
    r = result("r1")
    text = results_table(published(r)) + "\n" + evaluation_sections(published(r))
    reported = {
        n for line in text.splitlines() if NOT_A_RESULT not in line for n in NUMBER.findall(line)
    }
    assert reported
    assert reported <= set(NUMBER.findall(r.model_dump_json()))


def test_only_the_interval_level_is_exempt_from_the_numbers_check() -> None:
    r = result("r1")
    text = results_table(published(r)) + "\n" + evaluation_sections(published(r))
    exempt = [line for line in text.splitlines() if NOT_A_RESULT in line]
    assert len(exempt) == 2
    assert all(set(NUMBER.findall(line)) == {"95%"} for line in exempt)


def test_run_section_lists_the_largest_units_and_sums_the_rest() -> None:
    section = run_section(clean(result("r1", units=12)))
    assert "| `rec9` |" in section
    assert "`rec10`" not in section
    assert "The other 2 hold 2 images and 19 fuel boxes." in section
    assert "| 0.95 | n/a | 0.0 |" in section
    assert "| Small | Medium | Large |\n|---|---|---|\n| 0.311 | 0.702 | n/a |" in section


def test_replace_between_rewrites_only_the_marked_text() -> None:
    start, end = RESULTS_MARKERS
    text = f"intro\n{start}\nold\n{end}\noutro\n"
    assert replace_between(text, RESULTS_MARKERS, "new") == f"intro\n{start}\nnew\n{end}\noutro\n"


@pytest.mark.parametrize(
    "text",
    [
        "no markers",
        f"{RESULTS_MARKERS[0]} only the start",
        f"{RESULTS_MARKERS[1]} end first {RESULTS_MARKERS[0]}",
        f"{RESULTS_MARKERS[0]}{RESULTS_MARKERS[0]}{RESULTS_MARKERS[1]}",
    ],
)
def test_replace_between_refuses_bad_markers(text: str) -> None:
    with pytest.raises(ConfigError):
        replace_between(text, RESULTS_MARKERS, "new")


def test_write_report_is_idempotent_and_keeps_hand_written_text(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_run(runs, result("r1"), "2026-09-27T10:00:00+00:00")
    readme, evaluation = tmp_path / "README.md", tmp_path / "EVALUATION.md"
    start, end = RESULTS_MARKERS
    readme.write_text(f"# Title\n{start}\nTBD\n{end}\nAttribution\n", "utf-8")
    start, end = EVALUATION_MARKERS
    evaluation.write_text(f"Intro\n{start}\n{end}\n", "utf-8")

    assert write_report(runs, readme, evaluation) == [readme, evaluation]
    first = readme.read_text(encoding="utf-8"), evaluation.read_text(encoding="utf-8")
    assert write_report(runs, readme, evaluation) == []
    assert (readme.read_text(encoding="utf-8"), evaluation.read_text(encoding="utf-8")) == first
    assert first[0].startswith("# Title\n")
    assert first[0].endswith("Attribution\n")
    assert "### m on alpha test" in first[1]
    # Rebase merges rewrite commit SHAs, so docs name the run, not the commit.
    assert "abc123" not in first[1]


def test_main_rewrites_both_documents_from_the_configured_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO_ROOT / "configs" / "project.yaml", tmp_path / "configs" / "project.yaml")
    write_run(tmp_path / "reports" / "runs", result("r1"), "2026-09-27T10:00:00+00:00")
    start, end = RESULTS_MARKERS
    Path("README.md").write_text(f"{start}\n{end}\n", "utf-8")
    Path("docs").mkdir()
    start, end = EVALUATION_MARKERS
    Path("docs/EVALUATION.md").write_text(f"{start}\n{end}\n", "utf-8")
    assert main(["--project-config", "configs/project.yaml"]) == 0
    assert "| m | alpha | test |" in Path("README.md").read_text(encoding="utf-8")
    assert "### m on alpha test" in Path("docs/EVALUATION.md").read_text(encoding="utf-8")


def test_run_section_names_classes_the_model_was_not_trained_on() -> None:
    assert "not scored" not in run_section(clean(result("r1")))
    r = result("r1").model_copy(update={"unscored_classes": ["robot"]})
    assert "The robot boxes in alpha are not scored, since m was never trained to find them." in (
        run_section(clean(r))
    )


def with_limit(at_limit: int, above: int) -> RunResult:
    limit = PerImageLimit(per_image=25, images_at_limit=at_limit, images_above_threshold=above)
    return result("r1").model_copy(update={"per_image_limit": limit})


def test_run_section_says_when_the_per_image_limit_was_reached() -> None:
    assert "reached the limit" not in run_section(clean(result("r1")))
    assert "reached the limit" not in run_section(clean(with_limit(0, 0)))
    reached = run_section(clean(with_limit(3, 0)))
    assert "3 of 5 images reached the limit of 25 predictions per image and class" in reached
    assert "lower bound" not in reached
    assert "In 2 of them every kept box" in run_section(clean(with_limit(3, 2)))


def test_limit_note_numbers_are_stored_in_the_run() -> None:
    r = with_limit(3, 2)
    reported = {n for line in run_section(clean(r)).splitlines() for n in NUMBER.findall(line)}
    assert reported <= set(NUMBER.findall(r.model_dump_json())) | {"95%"}


def score(name: str, images: int, labeled: int, hits: int, fps: int) -> SliceScore:
    return SliceScore(
        name=name,
        images=images,
        labeled=labeled,
        hits=hits,
        false_positives=fps,
        misses=labeled - hits,
        map50=0.812 if labeled else None,
        precision=round(hits / (hits + fps), 3) if hits + fps else None,
        recall=round(hits / labeled, 3) if labeled else None,
        false_positives_per_image=round(fps / images, 3),
        false_positive_kinds={
            "duplicate": 0,
            "localization": 1,
            "inside_unscored": 0,
            "background": fps - 1,
        },
    )


def diagnosis(review: list[ReviewCount] | None = None) -> Diagnosis:
    yes, no = score("yes", 3, 5, 4, 2), score("no", 2, 0, 0, 3)
    fuel = ClassDiagnosis(
        name="fuel",
        overall=score("all", 5, 5, 4, 5),
        slicings=[
            Slicing(
                by="brightness",
                slices=[score("below 101.5", 2, 1, 1, 2), score("101.5 and above", 3, 4, 3, 3)],
            ),
            Slicing(by="has labeled fuel", slices=[yes, no]),
        ],
        sizes=[SizeRow(bucket="small", labeled=2, hits=1, recall=0.5, false_positives=4)],
    )
    return Diagnosis(
        model="m",
        confidence=0.5,
        iou=0.5,
        feature_px=384,
        localization_floor=0.1,
        area_buckets=AreaBuckets(small_max=0.0025, medium_max=0.0225),
        quantiles=[0.1, 0.5, 0.9],
        brightness_edges=[101.5],
        sharpness_edges=[220.25],
        splits=[
            SplitDiagnosis(
                run_id="r1",
                predictions_from="this run",
                dataset="alpha",
                split="test",
                max_predictions_per_image=25,
                classes=[fuel],
            )
        ],
        domain=[
            FeatureSummary(
                dataset="alpha",
                split="train",
                role="training",
                images=9,
                boxes=12,
                brightness=[80.25, 110.5, 140.75],
                sharpness=[90.125, 150.5, 400.875],
                box_side=[0.021, 0.043, 0.087],
                boxes_per_image=[1.0, 2.0, 6.0],
            )
        ],
        gallery=[GalleryTile(dataset="alpha", file_name="a.jpg", kind="miss", source="rec1")],
        review=review
        if review is not None
        else [
            ReviewCount(
                dataset="alpha",
                kind="miss",
                total=7,
                sampled=4,
                verdicts={"ball cut off at the image edge": 3, "clear fuel": 1},
                by_automatic_kind={},
            )
        ],
    )


def write_diagnosis_file(root: Path, d: Diagnosis) -> Path:
    out = root / "diagnosis" / d.model
    out.mkdir(parents=True)
    (out / "diagnosis.json").write_text(d.model_dump_json(indent=2), encoding="utf-8")
    return root / "diagnosis"


def test_every_number_in_the_diagnosis_section_is_in_its_json() -> None:
    d = diagnosis()
    text = diagnosis_sections([d])
    reported = {n for line in text.splitlines() for n in NUMBER.findall(line)}
    assert "101.5" in reported
    assert reported <= set(NUMBER.findall(d.model_dump_json()))


def test_diagnosis_section_names_slices_sizes_review_and_gallery() -> None:
    text = diagnosis_section(diagnosis())
    assert "Has labeled fuel:" in text
    assert "| alpha | no | 2 | 0 | 0 | 3 | 0 | 0.0 | n/a | n/a | 1.5 |" in text
    assert "| alpha | small | 2 | 1 | 0.5 | 4 |" in text
    assert "| Verdict | alpha misses |" in text
    assert "| ball cut off at the image edge | 3 |" in text
    assert "| 1 | alpha | miss | `rec1` |" in text


def test_a_diagnosis_without_review_says_so() -> None:
    assert "No errors have been reviewed by eye yet." in diagnosis_section(diagnosis(review=[]))


def test_write_report_fills_the_diagnosis_markers_once_a_diagnosis_exists(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_run(runs, result("r1"), "2026-09-27T10:00:00+00:00")
    readme, evaluation = tmp_path / "README.md", tmp_path / "EVALUATION.md"
    readme.write_text("{}\n{}\n".format(*RESULTS_MARKERS), "utf-8")
    start, end = EVALUATION_MARKERS
    d_start, d_end = DIAGNOSIS_MARKERS
    evaluation.write_text(f"{start}\n{end}\n{d_start}\nold\n{d_end}\n", "utf-8")
    # Without a diagnosis the marked text is left alone.
    write_report(runs, readme, evaluation, tmp_path / "missing")
    assert "\nold\n" in evaluation.read_text(encoding="utf-8")
    assert write_report(runs, readme, evaluation, write_diagnosis_file(tmp_path, diagnosis())) == [
        evaluation
    ]
    assert "### Where m's errors fall" in evaluation.read_text(encoding="utf-8")


def test_write_report_needs_diagnosis_markers_when_a_diagnosis_exists(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_run(runs, result("r1"), "2026-09-27T10:00:00+00:00")
    readme, evaluation = tmp_path / "README.md", tmp_path / "EVALUATION.md"
    readme.write_text("{}\n{}\n".format(*RESULTS_MARKERS), "utf-8")
    evaluation.write_text("{}\n{}\n".format(*EVALUATION_MARKERS), "utf-8")
    with pytest.raises(ConfigError):
        write_report(runs, readme, evaluation, write_diagnosis_file(tmp_path, diagnosis()))


def test_false_positive_kinds_are_split_by_whether_images_have_labels() -> None:
    text = diagnosis_section(diagnosis())
    assert "| alpha | all | 0 | 1 | 0 | 4 |" in text
    assert "| alpha | with labeled fuel | 0 | 1 | 0 | 1 |" in text
    assert "| alpha | without labeled fuel | 0 | 1 | 0 | 2 |" in text


def test_main_draws_figures_for_each_diagnosis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO_ROOT / "configs" / "project.yaml", tmp_path / "configs" / "project.yaml")
    write_run(tmp_path / "reports" / "runs", result("r1"), "2026-09-27T10:00:00+00:00")
    write_diagnosis_file(tmp_path / "reports", diagnosis().model_copy(update={"gallery": []}))
    Path("README.md").write_text("{}\n{}\n".format(*RESULTS_MARKERS), "utf-8")
    Path("docs").mkdir()
    markers = [*EVALUATION_MARKERS, *DIAGNOSIS_MARKERS]
    Path("docs/EVALUATION.md").write_text("\n".join(markers) + "\n", "utf-8")
    assert main(["--project-config", "configs/project.yaml"]) == 0
    assert Path("docs/assets/diagnosis_m_size.png").is_file()
    text = Path("docs/EVALUATION.md").read_text(encoding="utf-8")
    assert "### Where m's errors fall" in text
    assert "The failure gallery" not in text


def test_no_diagnosis_says_so() -> None:
    assert diagnosis_sections([]) == "No model has been diagnosed yet."
