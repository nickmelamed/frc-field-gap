"""Draw the diagnosis figures in docs/assets/ from a model's diagnosis.

Each figure shows numbers from the tables ``frc-report`` writes into
docs/EVALUATION.md. Each dataset has one color from a colorblind-safe
palette, and every bar is labeled with its value.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from frc_xdata.diagnose import SIZE_BUCKETS, Diagnosis

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
# Categorical slots 1 to 6 in fixed order. Slots 3 to 5 fall below 3:1
# against the surface, which is why every bar carries its value.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
DPI = 100
# The PNG's software tag names the matplotlib version, which would change
# every figure on an upgrade without changing what it shows.
METADATA: dict[str, str | None] = {"Software": None}


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width: float, height: float, cols: int = 1, rows: int = 1) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        _style(ax)
    return fig, flat


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, metadata=METADATA)
    plt.close(fig)
    return path


def size_figure(d: Diagnosis, path: Path) -> Path:
    """Draw recall by relative box size, one bar per dataset in each size."""
    splits = [(s.dataset, s.classes[0]) for s in d.splits]
    fig, (ax,) = _figure(8, 3.6)
    slot = 0.8 / len(splits)
    for n, (dataset, c) in enumerate(splits):
        rows = {r.bucket: r for r in c.sizes}
        # The first bar drawn names the dataset, since a size may have no boxes.
        label: str | None = dataset
        for x, bucket in enumerate(SIZE_BUCKETS):
            r = rows.get(bucket)
            left = x - 0.4 + n * slot
            if r is None or r.recall is None:
                ax.text(
                    left + slot / 2,
                    0.02,
                    "no boxes",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=MUTED,
                )
                continue
            ax.bar(
                left + slot / 2,
                r.recall,
                width=slot * 0.9,
                color=SERIES[n],
                label=label,
            )
            label = None
            ax.text(
                left + slot / 2,
                r.recall + 0.02,
                f"{r.recall}\n({r.labeled})",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT,
            )
    ax.set_xticks(range(len(SIZE_BUCKETS)), [f"{b} boxes" for b in SIZE_BUCKETS])
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(f"Recall at confidence {d.confidence}", color=MUTED, fontsize=9)
    ax.set_title(
        f"{d.model}: recall by box size relative to the image (labeled boxes in brackets)",
        loc="left",
        fontsize=10,
        color=TEXT,
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        ncols=len(splits),
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        labelcolor=TEXT,
    )
    fig.tight_layout()
    return _save(fig, path)


def source_figure(d: Diagnosis, dataset: str, path: Path) -> Path | None:
    """Draw false positives per image by source for one dataset, or return None."""
    split = next((s for s in d.splits if s.dataset == dataset), None)
    if split is None:
        return None
    c = split.classes[0]
    slicing = next((x for x in c.slicings if x.by == "source"), None)
    if slicing is None:
        return None
    slices = list(reversed(slicing.slices))
    fig, (ax,) = _figure(8, 0.5 * len(slices) + 1.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    labels = [f"{x.name} ({x.images} images, {x.labeled} labeled)" for x in slices]
    values = [x.false_positives_per_image for x in slices]
    ax.barh(range(len(slices)), values, height=0.7, color=SERIES[0])
    for y, v in enumerate(values):
        ax.text(v + 0.05, y, str(v), va="center", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(slices)), labels)
    top = max(values, default=0.0)
    ax.set_xlim(0, top * 1.15 if top else 1.0)
    ax.set_xlabel(
        f"False positives per image at confidence {d.confidence}", color=MUTED, fontsize=9
    )
    ax.set_title(
        f"{d.model} on {dataset}: false positives per image by source",
        loc="left",
        fontsize=10,
        color=TEXT,
    )
    fig.tight_layout()
    return _save(fig, path)


def _range_panel(
    ax: Axes, title: str, names: Sequence[str], values: Sequence[Sequence[float]]
) -> None:
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for y, v in enumerate(values):
        if len(v) != 3:
            continue
        low, mid, high = v
        ax.plot([low, high], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot(
            [mid],
            [y],
            marker="o",
            markersize=8,
            color=SERIES[0],
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.text(mid, y + 0.28, str(mid), ha="center", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10, color=TEXT)


def domain_figure(d: Diagnosis, path: Path) -> Path | None:
    """Draw each split's feature quantiles, training split first, or return None.

    Needs the quantiles to be three values, drawn as low, middle, and high.
    """
    if len(d.quantiles) != 3:
        return None
    names = [f"{f.dataset} {f.split}" for f in d.domain]
    low, mid, high = d.quantiles
    fig, axes = _figure(10, 5.2, cols=2, rows=2)
    panels = [
        ("Box side, as a share of the image side", [f.box_side for f in d.domain]),
        ("Labeled boxes per image", [f.boxes_per_image for f in d.domain]),
        ("Brightness (mean gray level, 0 to 255)", [f.brightness for f in d.domain]),
        ("Sharpness (variance of the Laplacian)", [f.sharpness for f in d.domain]),
    ]
    for ax, (title, values) in zip(axes, panels, strict=True):
        _range_panel(ax, title, names, values)
    fig.suptitle(
        f"{d.model}: training split against each test split (line from quantile {low} to "
        f"{high}, dot and value at {mid})",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    fig.tight_layout()
    return _save(fig, path)


def write_figures(d: Diagnosis, assets_dir: Path) -> list[Path]:
    """Write every figure for one model and return the paths written.

    False positives by source are drawn for the split with the most false
    positives per image.
    """
    stem = f"diagnosis_{d.model}"
    written = [size_figure(d, assets_dir / f"{stem}_size.png")]
    worst = max(d.splits, key=lambda s: s.classes[0].overall.false_positives_per_image)
    found = source_figure(d, worst.dataset, assets_dir / f"{stem}_{worst.dataset}_sources.png")
    written += [] if found is None else [found]
    domain = domain_figure(d, assets_dir / f"{stem}_domain.png")
    return written + ([] if domain is None else [domain])
