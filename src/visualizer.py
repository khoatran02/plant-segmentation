"""Render PlantSeg COCO annotations over their source images."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets" / "plantseg"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "visualization"
SPLITS = ("train", "val", "test")


def _color_for_category(category_id: int) -> tuple[int, int, int]:
    """Return a stable, reasonably distinct color for a category."""
    red = (category_id * 67 + 53) % 256
    green = (category_id * 149 + 97) % 256
    blue = (category_id * 211 + 193) % 256
    return red, green, blue


def _polygon_points(segmentation: Any) -> Iterable[list[tuple[float, float]]]:
    """Convert a COCO polygon segmentation into lists of point pairs."""
    if not isinstance(segmentation, list):
        return

    polygons = (
        segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
    )
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            continue
        yield list(zip(polygon[::2], polygon[1::2]))


def render_image(
    image_path: Path,
    output_path: Path,
    annotations: Iterable[dict[str, Any]],
    point_radius: int = 3,
) -> None:
    """Render polygon outlines and their vertices without filling interiors."""
    with Image.open(image_path) as source:
        image = source.convert("RGBA")

    line_draw = ImageDraw.Draw(image)

    for annotation in annotations:
        color = _color_for_category(int(annotation.get("category_id", 0)))
        for points in _polygon_points(annotation.get("segmentation")):
            pixel_points = [(round(x), round(y)) for x, y in points]
            line_draw.line(
                pixel_points + [pixel_points[0]],
                fill=(*color, 255),
                width=2,
                joint="curve",
            )
            for x, y in pixel_points:
                line_draw.ellipse(
                    (
                        x - point_radius,
                        y - point_radius,
                        x + point_radius,
                        y + point_radius,
                    ),
                    fill=(*color, 255),
                    outline=(255, 255, 255, 255),
                    width=1,
                )

    image = image.convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def render_split(dataset_dir: Path, output_dir: Path, split: str) -> tuple[int, int]:
    """Render every annotated image in one dataset split."""
    annotation_path = dataset_dir / f"annotation_{split}.json"
    image_dir = dataset_dir / "images" / split
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open(encoding="utf-8") as file:
        dataset = json.load(file)

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    rendered = 0
    skipped = 0
    for image_info in dataset.get("images", []):
        image_path = image_dir / image_info["file_name"]
        output_path = output_dir / split / image_info["file_name"]
        if not image_path.is_file():
            skipped += 1
            print(f"Skipping missing image: {image_path}")
            continue
        render_image(image_path, output_path, annotations_by_image[int(image_info["id"])])
        rendered += 1

    return rendered, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else (args.split,)
    total_rendered = 0
    total_skipped = 0
    for split in splits:
        rendered, skipped = render_split(args.dataset_dir, args.output_dir, split)
        total_rendered += rendered
        total_skipped += skipped
        print(f"{split}: rendered {rendered} image(s), skipped {skipped}")
    print(f"Done: rendered {total_rendered} image(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
