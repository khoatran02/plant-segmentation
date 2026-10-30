"""Refresh report evidence without importing PyTorch or changing the dataset.

Uses the actual annotation-preparation functions extracted from training.py.
Does not execute training, inference, or a pixel-level dataset re-audit.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import statistics
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    source = ROOT / "src/training.py"
    tree = ast.parse(source.read_text())
    wanted = {"load_metadata", "_find_annotation_file", "build_coco_split"}
    extracted = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
    namespace = dict(Path=Path, Any=Any, csv=csv, json=json, Counter=Counter)
    exec(compile(extracted, str(source), "exec"), namespace)
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        try:
                            defaults[node.args[0].value] = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):
                            pass
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "metadata, file presence, raw JSON structure and actual cache builder; no pixel re-audit or model run",
              "sha256": {str(p.relative_to(ROOT)): digest(p) for p in [source, ROOT / "docs/plantseg-v6-v7-statistics.md", ROOT / "docs/plantseg-stats/flagged-files.csv"]},
              "sam3_commit": subprocess.check_output(["git", "-C", str(ROOT / "src/sam3"), "rev-parse", "HEAD"], text=True).strip(),
              "training_cli_defaults": defaults, "datasets": {}}
    for version in ("v6", "v7"):
        dataset = ROOT / "data" / f"plantseg-{version}"
        with (dataset / "Metadata.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        entry = {"metadata_rows": len(rows), "metadata_class_ids": len({r["Index"] for r in rows}),
                 "host_plants": len({r["Plant"] for r in rows}), "splits": {},
                 "metadata_sha256": digest(dataset / "Metadata.csv"),
                 "coverage_mean": statistics.mean(float(r["Mask ratio"]) for r in rows),
                 "coverage_median": statistics.median(float(r["Mask ratio"]) for r in rows)}
        for split in ("train", "val", "test"):
            path = namespace["_find_annotation_file"](dataset, split)
            raw = json.loads(path.read_text())
            ids = {int(i["id"]) for i in raw["images"]}
            item = {"images": len(list((dataset / "images" / split).glob("*.jpg"))),
                    "masks": len(list((dataset / "annotations" / split).glob("*.png"))),
                    "json_images": len(raw["images"]), "json_categories": len(raw.get("categories", [])),
                    "annotations_listed": len(raw["annotations"]),
                    "orphan_annotations": sum(int(a["image_id"]) not in ids for a in raw["annotations"]),
                    "json_sha256": digest(path)}
            with tempfile.TemporaryDirectory(prefix="plantseg-report-") as tmp:
                cached = json.loads(namespace["build_coco_split"](dataset, split, Path(tmp)).read_text())
                item.update(cache_images=len(cached["images"]), cache_annotations=len(cached["annotations"]), cache_categories=len(cached["categories"]))
            entry["splits"][split] = item
        result["datasets"][version] = entry
    (evidence / "snapshot.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {evidence / 'snapshot.json'}")


if __name__ == "__main__":
    main()
