from pathlib import Path

import pytest
from matplotlib.figure import Figure
from PIL import Image
from test_report import choice, diagnosis

from frc_xdata import figures
from frc_xdata.diagnose import Slicing
from frc_xdata.figures import (
    domain_figure,
    size_figure,
    source_figure,
    threshold_figure,
    write_figures,
)

# The large-file hook rejects anything over 500 KB.
MAX_BYTES = 500_000


def test_write_figures_draws_size_and_domain(tmp_path: Path) -> None:
    # The test diagnosis has no source slicing, so there is no source figure.
    paths = write_figures(diagnosis(), tmp_path)
    assert [p.name for p in paths] == ["diagnosis_m_size.png", "diagnosis_m_domain.png"]
    for p in paths:
        assert 0 < p.stat().st_size < MAX_BYTES
        with Image.open(p) as image:
            assert image.format == "PNG"


def test_figures_are_the_same_bytes_every_time(tmp_path: Path) -> None:
    first = [p.read_bytes() for p in write_figures(diagnosis(), tmp_path / "a")]
    second = [p.read_bytes() for p in write_figures(diagnosis(), tmp_path / "b")]
    assert first == second


def test_no_source_figure_without_a_source_slicing(tmp_path: Path) -> None:
    d = diagnosis()
    assert source_figure(d, "alpha", tmp_path / "s.png") is None
    split = d.splits[0]
    fuel = split.classes[0]
    by_source = Slicing(by="source", slices=fuel.slicings[1].slices)
    fuel = fuel.model_copy(update={"slicings": [*fuel.slicings, by_source]})
    d = d.model_copy(update={"splits": [split.model_copy(update={"classes": [fuel]})]})
    assert source_figure(d, "alpha", tmp_path / "s.png") == tmp_path / "s.png"
    assert source_figure(d, "beta", tmp_path / "t.png") is None


def test_the_domain_figure_needs_three_quantiles(tmp_path: Path) -> None:
    d = diagnosis().model_copy(update={"quantiles": [0.5]})
    assert domain_figure(d, tmp_path / "d.png") is None


def test_a_dataset_with_no_small_boxes_is_still_in_the_size_legend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = diagnosis()
    split = d.splits[0]
    fuel = split.classes[0]
    # Its only boxes are medium, as in a split with no small boxes.
    medium = [s.model_copy(update={"bucket": "medium"}) for s in fuel.sizes]
    fuel = fuel.model_copy(update={"sizes": medium})
    d = d.model_copy(update={"splits": [split.model_copy(update={"classes": [fuel]})]})
    drawn: list[Figure] = []
    monkeypatch.setattr(figures, "_save", lambda fig, path: drawn.append(fig) or path)
    size_figure(d, tmp_path / "s.png")
    legend = drawn[0].axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == [split.dataset]


def test_threshold_figure_draws_one_repeatable_png(tmp_path: Path) -> None:
    choices = [choice("m"), choice("n")]
    first = threshold_figure(choices, tmp_path / "a.png")
    assert 0 < first.stat().st_size < MAX_BYTES
    with Image.open(first) as image:
        assert image.format == "PNG"
    assert threshold_figure(choices, tmp_path / "b.png").read_bytes() == first.read_bytes()
