import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from frc_xdata.config import DatasetSpec
from frc_xdata.download import MANIFEST_NAME, build_manifest

SIDE = 64


def smooth_image(seed: int, side: int = SIDE) -> Image.Image:
    """Return a blurry random image, so its perceptual hash is stable under resizing."""
    small = np.random.default_rng(seed).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((side, side), Image.Resampling.BILINEAR)


def coco(images: list[dict[str, Any]], anns: list[tuple[int, int, list[float]]]) -> dict[str, Any]:
    """Build a Roboflow-style COCO file, including its unused placeholder category."""
    return {
        "categories": [
            {"id": 0, "name": "workspace", "supercategory": "none"},
            {"id": 1, "name": "fuel", "supercategory": "workspace"},
            {"id": 2, "name": "robot", "supercategory": "workspace"},
        ],
        "images": images,
        "annotations": [
            {"id": n, "image_id": image_id, "category_id": cat, "bbox": bbox}
            for n, (image_id, cat, bbox) in enumerate(anns)
        ],
    }


def image_entry(
    image_id: int, name: str, side: int = SIDE, source: str | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": image_id, "file_name": name, "width": side, "height": side}
    if source is not None:
        entry["extra"] = {"name": source}
    return entry


def write_split(root: Path, data: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")


def finish(dataset_dir: Path) -> None:
    spec = DatasetSpec(workspace="team", project="p", version=1, license="CC BY 4.0")
    manifest = build_manifest(dataset_dir, dataset_dir.name, spec)
    (dataset_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(), encoding="utf-8")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Two tiny datasets with known duplicates, leaks, and label problems.

    alpha/train: a (fuel medium, robot large), b (no labels), and c, a byte
    copy of a whose one box runs past the image edge. alpha/valid: d, a
    resized copy of a sharing a's source name. beta/train: e (robot) and f,
    a byte copy of b.
    """
    raw = tmp_path / "data" / "raw"
    alpha, beta = raw / "alpha", raw / "beta"
    (alpha / "train").mkdir(parents=True)
    (alpha / "valid").mkdir(parents=True)
    (beta / "train").mkdir(parents=True)

    smooth_image(1).save(alpha / "train" / "a.png")
    smooth_image(2).save(alpha / "train" / "b.png")
    shutil.copy(alpha / "train" / "a.png", alpha / "train" / "c.png")
    smooth_image(1).resize((48, 48)).save(alpha / "valid" / "d.png")
    smooth_image(3).save(beta / "train" / "e.png")
    shutil.copy(alpha / "train" / "b.png", beta / "train" / "f.png")

    write_split(
        alpha / "train",
        coco(
            [
                image_entry(0, "a.png", source="a.png"),
                image_entry(1, "b.png"),
                image_entry(2, "c.png", source="a.png"),
            ],
            [(0, 1, [4, 4, 4, 4]), (0, 2, [0, 0, 40, 40]), (2, 1, [60, 60, 10, 10])],
        ),
    )
    write_split(
        alpha / "valid",
        coco([image_entry(0, "d.png", side=48, source="a.png")], [(0, 1, [1, 1, 1, 1])]),
    )
    write_split(
        beta / "train",
        coco(
            [image_entry(0, "e.png"), image_entry(1, "f.png")],
            [(0, 2, [0, 0, 10, 10])],
        ),
    )
    finish(alpha)
    finish(beta)
    return raw
