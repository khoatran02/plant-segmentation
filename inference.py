"""Save a reproducible sample of test predictions, overlays, and metrics."""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.data import (
    DEFAULT_DATASET_DIR,
    categories_from_metadata,
    load_metadata,
    load_pair,
    split_samples,
)
from src.evaluation import Sam3SemanticPredictor
from src.metrics import SegmentationMetrics
from src.runtime import file_sha256, read_checkpoint, write_json
from src.sam3_adapter import (
    build_model,
    cache_text_features,
    configure_semantic_only,
    configure_trainable,
)


def overlay(image, mask, color):
    pixels = np.asarray(image).copy()
    foreground = mask > 0
    pixels[foreground] = (pixels[foreground] * 0.5 + np.array(color, dtype=float) * 0.5).astype(
        np.uint8
    )
    return Image.fromarray(pixels)


def comparison(image, target, prediction, title, score):
    width, height = 360, 300
    canvas = Image.new("RGB", (width * 3, height + 105), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    draw.text((12, 8), title, fill="#0f172a", font=font)
    labels = ("Original", "Ground truth (green)", "Prediction (red)")
    panels = (
        image,
        overlay(image, target, (34, 197, 94)),
        overlay(image, prediction, (239, 68, 68)),
    )
    for index, (label, panel) in enumerate(zip(labels, panels)):
        draw.text((index * width + 12, 34), label, fill="#334155", font=font)
        panel = ImageOps.contain(panel, (width - 16, height - 8))
        canvas.paste(
            panel, (index * width + (width - panel.width) // 2, 60 + (height - panel.height) // 2)
        )
    draw.text(
        (12, height + 60),
        f"Foreground IoU: {score['iou']:.3f}   Dice: {score['dice']:.3f}",
        fill="#0f172a",
        font=font,
    )
    diseases = score.get("predicted_diseases", [])
    text = ", ".join(diseases[:3]) if diseases else "background only"
    if len(diseases) > 3:
        text += f" (+{len(diseases) - 3} classes)"
    draw.text((12, height + 83), f"Predicted: {text}"[:135], fill="#0f172a", font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/last.pt"))
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/last-test-samples"))
    parser.add_argument("--cache_dir", type=Path, default=Path("outputs/hf-cache"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--eval_mode", choices=("all_classes", "oracle"))
    parser.add_argument(
        "--known_disease",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default to the checkpoint's protocol; supplied disease and direct semantic mask",
    )
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.prompt_batch_size < 1:
        parser.error("count and prompt_batch_size must be positive")
    started = time.monotonic()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    state = read_checkpoint(args.checkpoint)
    config, identity = state["config"], state["model_identity"]
    if args.known_disease is None:
        args.known_disease = config.get("known_disease", False)
    if args.known_disease and args.eval_mode not in (None, "oracle"):
        parser.error("--known_disease requires --eval_mode oracle")
    args.eval_mode = args.eval_mode or ("oracle" if args.known_disease else "all_classes")
    checkpoint_info = {
        key: state.get(key)
        for key in ("epoch", "step", "best_metric", "input_image_size", "model_identity")
    }
    adaptation = state["trainable"]
    del state
    gc.collect()
    metadata = load_metadata(args.dataset_dir)
    categories = categories_from_metadata(metadata)
    samples = split_samples(args.dataset_dir, "test", metadata)
    samples = random.Random(args.seed).sample(samples, min(args.count, len(samples)))
    model_args = argparse.Namespace(
        model_id=identity["model_id"],
        model_revision=identity["resolved_revision"],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        attn_implementation=config.get("attn_implementation", "sdpa"),
        image_size=checkpoint_info["input_image_size"][0],
        no_pretrained=False,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        loss_mode=config.get("loss_mode", "combined"),
        known_disease=args.known_disease,
    )
    print("Loading pretrained SAM3 and adaptation weights...", flush=True)
    model = build_model(model_args)
    current = dict(model._plantseg_identity)
    cached_ref = (
        args.cache_dir
        / f"models--{model_args.model_id.replace('/', '--')}"
        / "refs"
        / model_args.model_revision
    )
    cached_commit = cached_ref.read_text().strip() if cached_ref.is_file() else None
    comparable = dict(current)
    unresolved_revision = identity["resolved_revision"] == "main"
    if unresolved_revision:
        # This legacy checkpoint recorded a branch rather than an immutable commit.
        comparable["resolved_revision"] = "main"
    saved_identity = dict(identity)
    comparable.pop("loss_version", None)
    saved_identity.pop("loss_version", None)
    if comparable != saved_identity:
        raise ValueError(f"Checkpoint/base identity mismatch: {identity} vs {current}")
    configure_trainable(model, config["scope"])
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if expected != set(adaptation):
        raise ValueError("Checkpoint parameter set differs from the pretrained model")
    incompatible = model.load_state_dict(adaptation, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(incompatible.unexpected_keys)
    del adaptation
    if config["scope"] == "heads":
        cache_text_features(model, categories)
    mask_source = "semantic" if args.known_disease else config.get("eval_mask_source", "instance")
    if mask_source == "semantic":
        configure_semantic_only(model)
    gc.collect()
    if args.device == "auto":
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(args.device)
    model = model.to(device).eval()
    dtype = torch.float16 if device.type == "cuda" else None
    predictor = Sam3SemanticPredictor(
        model,
        device,
        dtype,
        config.get("score_threshold", 0.5),
        config.get("mask_threshold", 0.5),
        args.prompt_batch_size,
        mask_source,
        args.known_disease,
    )
    for subdir in ("predictions", "comparisons", "overlays"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    metrics = SegmentationMetrics(categories)
    result = {
        "status": "running",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_info": checkpoint_info,
        "base_model_identity": current,
        "base_model_cached_commit": cached_commit,
        "unresolved_training_revision": unresolved_revision,
        "device": str(device),
        "precision": "fp16" if dtype else "fp32",
        "split": "test",
        "eval_mode": args.eval_mode,
        "known_disease": args.known_disease,
        "confidence_filtering": not args.known_disease,
        "eval_mask_source": predictor.mask_source,
        "eval_prompt_batch_size": args.prompt_batch_size,
        "seed": args.seed,
        "score_threshold": None if args.known_disease else predictor.score_threshold,
        "mask_threshold": predictor.mask_threshold,
        "label_policy": config.get("label_policy", "mask"),
        "sample_metrics": [],
        "selected_images": [sample["image_path"].name for sample in samples],
        "note": "Small random sample; not a full test-set benchmark.",
    }
    write_json(args.output_dir / "results.json", result)
    panels = []
    for index, sample in enumerate(samples, 1):
        image, target = load_pair(sample, result["label_policy"])
        labels = np.unique(target[target > 0])
        disease_id = int(labels[0]) - 1 if len(labels) else sample["index"]
        selected = [categories[disease_id]] if args.eval_mode == "oracle" else categories
        name = sample["image_path"].name
        print(f"[{index}/{len(samples)}] {name}: {len(selected)} prompts on {device}", flush=True)
        image_started = time.monotonic()
        prediction = predictor(image, selected)
        score = metrics.update(prediction, target, disease_id, name)
        score["seconds"] = time.monotonic() - image_started
        score["predicted_labels"] = np.unique(prediction[prediction > 0]).tolist()
        score["predicted_diseases"] = [
            categories[label - 1]["name"] for label in score["predicted_labels"]
        ]
        score["ground_truth_disease"] = categories[disease_id]["name"]
        Image.fromarray(prediction).save(args.output_dir / "predictions" / sample["mask_path"].name)
        overlay(image, prediction, (239, 68, 68)).save(
            args.output_dir / "overlays" / f"{Path(name).stem}.jpg", quality=92
        )
        panel = comparison(
            image, target, prediction, f"{name} | {categories[disease_id]['name']}", score
        )
        panel.save(args.output_dir / "comparisons" / f"{Path(name).stem}.jpg", quality=92)
        panels.append(panel)
        result["sample_metrics"].append(score)
        result["metrics"] = metrics.compute()
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "results.json", result)
        print(
            f"  IoU={score['iou']:.4f}, Dice={score['dice']:.4f}, {score['seconds']:.1f}s",
            flush=True,
        )
    sheet = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(args.output_dir / "contact-sheet.jpg", quality=94)
    result["status"] = "complete"
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output_dir / "results.json", result)
    print(f"Saved {len(samples)} predictions to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
