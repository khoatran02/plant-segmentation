"""Measure native PlantSeg image sizes and ground-truth foreground coverage."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data import DEFAULT_DATASET_DIR, categories_from_metadata, load_metadata, split_samples


def measure_dataset(dataset_dir: Path, splits: list[str]) -> list[dict]:
    metadata = load_metadata(dataset_dir)
    num_classes = len(categories_from_metadata(metadata))
    rows = []
    for split in splits:
        samples = split_samples(dataset_dir, split, metadata)
        for sample in tqdm(samples, desc=split):
            with Image.open(sample["image_path"]) as image:
                stored_width, stored_height = image.size
                orientation = image.getexif().get(274, 1)
            width, height = stored_width, stored_height
            # Same displayed frame as src.data.load_pair, without decoding RGB pixels.
            if orientation in {5, 6, 7, 8}:
                width, height = height, width
            with Image.open(sample["mask_path"]) as image:
                mask = np.array(image)
            if mask.ndim != 2 or mask.shape != (height, width):
                raise ValueError(f"Image/mask shape mismatch: {sample['image_path']}")
            counts = np.bincount(mask.ravel())
            labels = np.flatnonzero(counts[1:]) + 1
            if len(labels) > 1 or (len(labels) and labels[0] > num_classes):
                raise ValueError(f"Invalid foreground labels: {sample['mask_path']}: {labels}")
            foreground = int(np.count_nonzero(mask))
            rows.append(
                {
                    "name": sample["image_path"].name,
                    "split": split,
                    "stored_width": stored_width,
                    "stored_height": stored_height,
                    "exif_orientation": orientation,
                    "width": width,
                    "height": height,
                    "foreground_pixels": foreground,
                    "total_pixels": int(mask.size),
                    "mask_ratio_percent": 100 * foreground / mask.size,
                    "metadata_label": sample["index"] + 1,
                    "png_label": int(labels[0]) if len(labels) else 0,
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    ratios = np.array([row["mask_ratio_percent"] for row in rows])
    histogram, edges = np.histogram(ratios, bins=np.arange(101))
    return {
        "images": len(rows),
        "splits": {
            split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")
        },
        "width_median_px": float(np.median([r["width"] for r in rows])),
        "height_median_px": float(np.median([r["height"] for r in rows])),
        "mask_ratio_mean_percent": float(ratios.mean()),
        "mask_ratio_median_percent": float(np.median(ratios)),
        "mask_ratio_min_percent": float(ratios.min()),
        "mask_ratio_max_percent": float(ratios.max()),
        "pixel_weighted_mask_ratio_percent": (
            100 * sum(r["foreground_pixels"] for r in rows) / sum(r["total_pixels"] for r in rows)
        ),
        "empty_masks": sum(r["foreground_pixels"] == 0 for r in rows),
        "exif_axis_swaps": sum(r["exif_orientation"] in {5, 6, 7, 8} for r in rows),
        "label_disagreements": [
            r["name"] for r in rows if r["png_label"] and r["png_label"] != r["metadata_label"]
        ],
        "histogram_bin_edges_percent": edges.tolist(),
        "histogram_counts": histogram.tolist(),
        "method": {
            "resolution": "Native image width/height after EXIF orientation; no resizing.",
            "mask_ratio": "100 * count(PNG pixels > 0) / number of PNG pixels, per image.",
            "histogram": "100 bins of width 1%; [left, right), final bin includes 100%.",
            "mask_source": "Ground-truth PNG; all nonzero disease labels are foreground.",
        },
        "comparison_datasets": [],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_statistics(rows: list[dict], summary: dict, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16, "svg.fonttype": "none"})

    def resolutions(ax):
        ax.scatter(
            [r["width"] for r in rows],
            [r["height"] for r in rows],
            s=30,
            c="#83b915",
            edgecolors="white",
            linewidths=0.5,
            alpha=0.65,
            label=f"PlantSeg (In-the-wild), n={len(rows):,}",
        )
        ax.set(xscale="log", yscale="log", xlabel="Width (pixels)", ylabel="Height (pixels)")
        ax.set_title("Image Resolutions")
        ax.grid(True, which="major", color="0.65", alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left")

    def mask_ratios(ax):
        ax.bar(
            np.arange(100) + 0.5,
            summary["histogram_counts"],
            width=1,
            color="#609fbc",
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set(xlim=(0, 100), xlabel="Mask ratios (%)", ylabel="Number of Images")
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_title(f"Ground-truth Mask Ratios (n={len(rows):,})")
        ax.grid(axis="y", color="0.65", alpha=0.8)
        ax.set_axisbelow(True)

    for filename, plot in (("image-resolutions", resolutions), ("mask-ratios", mask_ratios)):
        fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
        plot(ax)
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"{filename}.{extension}", dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    resolutions(axes[0])
    mask_ratios(axes[1])
    fig.savefig(output_dir / "distributions.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/dataset-statistics"))
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"]
    )
    args = parser.parse_args()
    if len(set(args.splits)) != len(args.splits):
        parser.error("Each split must be specified only once")
    rows = measure_dataset(args.dataset_dir, args.splits)
    summary = summarize(rows)
    summary["dataset_dir"] = str(args.dataset_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per-image.csv", rows)
    write_csv(
        args.output_dir / "mask-ratio-histogram.csv",
        [
            {"left_percent": i, "right_percent": i + 1, "images": count}
            for i, count in enumerate(summary["histogram_counts"])
        ],
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_statistics(rows, summary, args.output_dir)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("histogram")}, indent=2))
    print(f"Saved plots and measurements to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
