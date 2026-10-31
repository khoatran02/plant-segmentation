import json

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import pytest

from src.data import build_coco_split, load_metadata, load_pair, split_samples


def test_png_rle_roundtrip_preserves_all_pixels(tiny_dataset, tmp_path):
    path = build_coco_split(tiny_dataset, "train", tmp_path / "cache")
    payload = json.loads(path.read_text())
    assert len(payload["images"]) == 3
    assert len(payload["annotations"]) == 6
    for info, sample in zip(payload["images"], split_samples(tiny_dataset, "train")):
        _, mask = load_pair(sample)
        restored = np.zeros_like(mask, dtype=bool)
        for ann in payload["annotations"]:
            if ann["image_id"] == info["id"]:
                restored |= mask_utils.decode(ann["segmentation"]).astype(bool)
                assert ann["category_id"] == sample["index"]
        np.testing.assert_array_equal(restored, mask > 0)


def test_cache_invalidation_and_dataset_is_not_modified(tiny_dataset, tmp_path):
    sample = split_samples(tiny_dataset, "train")[0]
    original = sample["image_path"].read_bytes()
    path = build_coco_split(tiny_dataset, "train", tmp_path / "cache")
    before = json.loads(path.read_text())["plantseg"]["fingerprint"]
    mtime = path.stat().st_mtime_ns
    assert build_coco_split(tiny_dataset, "train", tmp_path / "cache").stat().st_mtime_ns == mtime
    mask = np.ones((8, 12), np.uint8)
    Image.fromarray(mask).save(sample["mask_path"])
    build_coco_split(tiny_dataset, "train", tmp_path / "cache")
    assert json.loads(path.read_text())["plantseg"]["fingerprint"] != before
    assert sample["image_path"].read_bytes() == original


def test_more_than_200_components_merged_without_pixel_loss(tiny_dataset, tmp_path):
    sample = split_samples(tiny_dataset, "train")[0]
    Image.new("RGB", (50, 50)).save(sample["image_path"])
    mask = np.zeros((50, 50), np.uint8)
    mask[::3, ::3] = 1  # 289 disconnected pseudo-instances
    Image.fromarray(mask).save(sample["mask_path"])
    payload = json.loads(build_coco_split(tiny_dataset, "train", tmp_path / "cache").read_text())
    anns = [a for a in payload["annotations"] if a["image_id"] == 1]
    assert len(anns) == 200
    restored = np.logical_or.reduce([mask_utils.decode(a["segmentation"]) for a in anns])
    np.testing.assert_array_equal(restored, mask > 0)


def test_exif_applied_to_rgb_only(tiny_dataset):
    sample = split_samples(tiny_dataset, "train")[0]
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (8, 12)).save(sample["image_path"], exif=exif)
    image, mask = load_pair(sample)
    assert image.size == (12, 8)
    assert mask.shape == (8, 12)


def test_metadata_label_conflict_and_missing_mask_fail(tiny_dataset):
    sample = split_samples(tiny_dataset, "train")[0]
    Image.fromarray(np.full((8, 12), 9, np.uint8)).save(sample["mask_path"])
    with pytest.raises(ValueError, match="disagree"):
        load_pair(sample)
    sample["mask_path"].unlink()
    with pytest.raises(FileNotFoundError):
        split_samples(tiny_dataset, "train")
    assert len(load_metadata(tiny_dataset)) == 9


def test_label_policies_are_explicit_and_recorded(tiny_dataset, tmp_path):
    sample = split_samples(tiny_dataset, "train")[0]
    mask = np.zeros((8, 12), np.uint8)
    mask[1:3, 1:3] = 2  # metadata says label 1
    Image.fromarray(mask).save(sample["mask_path"])
    original = sample["mask_path"].read_bytes()
    report = []
    _, unchanged = load_pair(sample, "mask", report)
    assert set(np.unique(unchanged)) == {0, 2}
    _, corrected = load_pair(sample, "metadata", report)
    assert set(np.unique(corrected)) == {0, 1}
    assert len(report) == 2
    assert sample["mask_path"].read_bytes() == original
    payload = json.loads(
        build_coco_split(tiny_dataset, "train", tmp_path / "cache", label_policy="mask").read_text()
    )
    assert payload["annotations"][0]["category_id"] == 1
    assert payload["plantseg"]["label_disagreements"][0]["png_label"] == 2
    with pytest.raises(ValueError, match="disagree"):
        load_pair(sample, "strict")
