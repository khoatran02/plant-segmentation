import numpy as np
import pytest
from PIL import Image

from src.dataset_statistics import measure_dataset, summarize


def test_counts_foreground_pixels_instead_of_summing_class_ids(tiny_dataset):
    rows = measure_dataset(tiny_dataset, ["train", "val", "test"])
    assert len(rows) == 9
    assert all(r["foreground_pixels"] == 12 for r in rows)
    assert all(r["mask_ratio_percent"] == 12.5 for r in rows)
    summary = summarize(rows)
    assert summary["histogram_counts"][12] == 9
    assert sum(summary["histogram_counts"]) == 9
    assert summary["label_disagreements"] == []


def test_exif_and_full_mask_are_included(tiny_dataset):
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (8, 12)).save(tiny_dataset / "images/train/train_0.jpg", exif=exif)
    Image.fromarray(np.ones((8, 12), dtype=np.uint8)).save(
        tiny_dataset / "annotations/train/train_0.png"
    )
    rows = measure_dataset(tiny_dataset, ["train"])
    assert (rows[0]["width"], rows[0]["height"]) == (12, 8)
    assert rows[0]["mask_ratio_percent"] == 100
    summary = summarize(rows)
    assert summary["exif_axis_swaps"] == 1
    assert summary["histogram_counts"][-1] == 1
    assert sum(summary["histogram_counts"]) == 3


def test_mismatched_mask_shape_fails(tiny_dataset):
    Image.new("L", (4, 4)).save(tiny_dataset / "annotations/train/train_0.png")
    with pytest.raises(ValueError, match="shape mismatch"):
        measure_dataset(tiny_dataset, ["train"])
