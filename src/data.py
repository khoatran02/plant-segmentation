"""PlantSeg metadata, aligned PNG pairs, and reproducible SAM3 COCO caches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "dataset" / "plantseg"
if not DEFAULT_DATASET_DIR.exists() and (ROOT / "datasets" / "plantseg").exists():
    DEFAULT_DATASET_DIR = ROOT / "datasets" / "plantseg"


def load_metadata(dataset_dir: Path) -> dict:
    with (dataset_dir / "Metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metadata = {}
    for row in rows:
        name = row["Name"]
        if name in metadata:
            raise ValueError(f"Duplicate metadata image: {name}")
        metadata[name] = {
            "index": int(row["Index"]),
            "plant": row["Plant"].strip().lower(),
            "disease": row["Disease"].strip().lower(),
            "mask": row.get("Label file") or Path(name).with_suffix(".png").name,
        }
    return metadata


def categories_from_metadata(metadata: dict) -> list[dict]:
    categories = {}
    for row in metadata.values():
        entry = {
            "id": row["index"],
            "name": row["disease"],
            "supercategory": row["plant"],
        }
        if entry["id"] in categories and categories[entry["id"]] != entry:
            raise ValueError(f"Conflicting class metadata: {entry}")
        categories[entry["id"]] = entry
    ids = sorted(categories)
    if ids != list(range(len(ids))):
        raise ValueError("Metadata class IDs must be contiguous starting at zero")
    return [categories[i] for i in ids]


def split_samples(dataset_dir: Path, split: str, metadata: dict | None = None) -> list[dict]:
    metadata = metadata if metadata is not None else load_metadata(dataset_dir)
    image_dir = dataset_dir / "images" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if image_path.name not in metadata:
            raise ValueError(f"Image absent from metadata: {image_path}")
        row = metadata[image_path.name]
        mask_path = dataset_dir / "annotations" / split / row["mask"]
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        samples.append({"image_path": image_path, "mask_path": mask_path, **row})
    if not samples:
        raise ValueError(f"Empty split: {image_dir}")
    return samples


def load_pair(
    sample: dict, label_policy: str = "strict", label_report: list | None = None
) -> tuple[Image.Image, np.ndarray]:
    """Apply EXIF to RGB only; masks already use the displayed coordinate frame."""
    with Image.open(sample["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(sample["mask_path"]) as source:
        mask = np.array(source)
    if mask.ndim != 2 or mask.shape != (image.height, image.width):
        raise ValueError(
            f"RGB/mask shape mismatch: {sample['image_path']} {image.size}, {mask.shape}"
        )
    labels = set(np.unique(mask).tolist()) - {0}
    if len(labels) > 1:
        raise ValueError(
            f"Expected one foreground disease per PNG: {sample['mask_path']}, labels={labels}"
        )
    if label_policy not in {"strict", "mask", "metadata"}:
        raise ValueError(f"Unknown label policy: {label_policy}")
    if labels - {sample["index"] + 1}:
        if label_policy == "strict":
            raise ValueError(
                f"PNG labels {labels} disagree with metadata for {sample['image_path']}"
            )
        if label_report is not None:
            label_report.append(
                {
                    "name": sample["image_path"].name,
                    "metadata_label": sample["index"] + 1,
                    "png_label": next(iter(labels)),
                    "policy": label_policy,
                }
            )
        if label_policy == "metadata":
            mask = np.where(mask > 0, sample["index"] + 1, 0).astype(np.uint8)
    return image, mask


def _annotation_file(dataset_dir: Path, split: str) -> Path:
    for candidate in (
        dataset_dir / f"annotation_{split}.json",
        dataset_dir / "json" / f"annotation_{split}.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"annotation_{split}.json under {dataset_dir}")


def dataset_fingerprint(dataset_dir: Path, samples: list[dict], source: str, split: str) -> str:
    """Paths + file stats invalidate caches on dataset replacement (not a content audit)."""
    digest = hashlib.sha256((str(dataset_dir.resolve()) + source + "png-rle-v1").encode())
    digest.update((dataset_dir / "Metadata.csv").read_bytes())
    paths = [p for s in samples for p in (s["image_path"], s["mask_path"])]
    if source == "coco":
        paths.append(_annotation_file(dataset_dir, split))
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def build_coco_split(
    dataset_dir: Path,
    split: str,
    cache_dir: Path,
    rebuild: bool = False,
    source: str = "png",
    label_policy: str = "mask",
) -> Path:
    """PNG connected components are pseudo-instances; all foreground pixels are retained.

    With >200 components, small components are merged into one final pseudo-instance.
    This bounds SAM3 matching memory without silently dropping images or mask pixels.
    """
    from pycocotools import mask as mask_utils
    from scipy import ndimage

    metadata = load_metadata(dataset_dir)
    samples = split_samples(dataset_dir, split, metadata)
    fingerprint = hashlib.sha256(
        (dataset_fingerprint(dataset_dir, samples, source, split) + label_policy).encode()
    ).hexdigest()
    out_path = cache_dir / "coco" / f"{split}-{source}-{label_policy}.json"
    if out_path.exists() and not rebuild:
        with out_path.open() as stream:
            cached = json.load(stream)
        if cached.get("plantseg", {}).get("fingerprint") == fingerprint:
            return out_path
    images, annotations, stats = [], [], Counter()
    label_report = []
    categories = categories_from_metadata(metadata)
    raw_by_name, raw_anns = {}, {}
    if source == "coco":
        with _annotation_file(dataset_dir, split).open() as stream:
            raw = json.load(stream)
        raw_by_name = {i["file_name"]: i for i in raw["images"]}
        for ann in raw["annotations"]:
            raw_anns.setdefault(str(ann["image_id"]), []).append(ann)
    elif source != "png":
        raise ValueError(f"Unknown annotation source: {source}")
    for image_id, sample in enumerate(samples, 1):
        image, mask = load_pair(sample, label_policy, label_report)
        labels = np.unique(mask[mask > 0])
        category_id = int(labels[0]) - 1 if labels.size else sample["index"]
        if not 0 <= category_id < len(categories):
            raise ValueError(f"Out-of-range PNG label for {sample['mask_path']}")
        image_info = {
            "id": image_id,
            "disease_id": category_id,
            "file_name": sample["image_path"].name,
            "height": image.height,
            "width": image.width,
        }
        regions = []
        if source == "png":
            components, count = ndimage.label(mask > 0, structure=np.ones((3, 3)))
            sizes = np.bincount(components.ravel())[1:]
            order = np.argsort(-sizes, kind="stable") + 1
            if count > 200:
                stats["images_with_merged_components"] += 1
                merge_ids = order[199:]
                components[np.isin(components, merge_ids)] = order[199]
                order = order[:200]
            for component in order:
                region = np.asfortranarray(components == component, dtype=np.uint8)
                rle = mask_utils.encode(region)
                area = int(mask_utils.area(rle))
                bbox = mask_utils.toBbox(rle).tolist()
                rle["counts"] = rle["counts"].decode("ascii")
                regions.append({"segmentation": rle, "area": area, "bbox": bbox})
        else:
            raw_image = raw_by_name.get(sample["image_path"].name)
            if raw_image is not None:
                for ann in raw_anns.get(str(raw_image["id"]), []):
                    polygons = [p for p in ann["segmentation"] if len(p) >= 6 and len(p) % 2 == 0]
                    if polygons and ann["bbox"][2] >= 1 and ann["bbox"][3] >= 1:
                        regions.append(
                            {
                                "segmentation": polygons,
                                "bbox": ann["bbox"],
                                "area": ann["area"],
                            }
                        )
            if not regions:
                stats["images_without_valid_coco_regions"] += 1
                continue
        images.append(image_info)
        for region in regions:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "iscrowd": 0,
                    **region,
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "plantseg": {
            "fingerprint": fingerprint,
            "source": source,
            "stats": dict(stats),
            "label_policy": label_policy,
            "label_disagreements": label_report,
            "dataset_dir": str(dataset_dir.resolve()),
            "split": split,
        },
    }
    temporary = out_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(out_path)
    print(
        f"[data] {split}/{source}: {len(images)} images, {len(annotations)} regions; {dict(stats)}",
        flush=True,
    )
    return out_path
