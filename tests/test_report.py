import json
import logging
import re
import shutil
from pathlib import Path

import pytest

from frc_xdata.errors import ConfigError
from frc_xdata.evaluate import (
    BootstrapResult,
    ClassInterval,
    ClassMetrics,
    ConfusionTable,
    EvalMetrics,
    Interval,
    PRPoint,
    RunResult,
    UnitCount,
)
from frc_xdata.report import (
    EVALUATION_MARKERS,
    RESULTS_MARKERS,
    PublishedRun,
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
