"""Native PNG evaluation, oracle disease prompts or all-class prediction."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.data import (
    DEFAULT_DATASET_DIR,
    categories_from_metadata,
    load_metadata,
    load_pair,
    split_samples,
    dataset_fingerprint,
)
from src.metrics import SegmentationMetrics
from src.runtime import (
    barrier,
    cleanup,
    distributed,
    gather_objects,
    rank,
    setup,
    world_size,
    write_json,
)


@torch.inference_mode()
def class_evidence(masks, scores, shape, mask_threshold=0.5, chunk_size=8):
    """Interpolate logits before thresholding; bound native-resolution GPU memory."""
    evidence = torch.zeros(shape, dtype=torch.float32, device=masks.device)
    for start in range(0, len(masks), chunk_size):
        logits = masks[start : start + chunk_size].float().unsqueeze(1)
        probabilities = (
            F.interpolate(logits, size=shape, mode="bilinear", align_corners=False)
            .sigmoid()
            .squeeze(1)
        )
        values = probabilities * scores[start : start + chunk_size, None, None]
        values.masked_fill_(probabilities <= mask_threshold, 0)
        evidence = torch.maximum(evidence, values.amax(0))
    return evidence


class Sam3SemanticPredictor:
    def __init__(self, model, device, dtype, score_threshold, mask_threshold, prompt_batch_size=1):
        if prompt_batch_size < 1:
            raise ValueError("prompt_batch_size must be positive")
        self.model, self.device, self.dtype = model, device, dtype
        self.score_threshold, self.mask_threshold = score_threshold, mask_threshold
        self.processor = model._plantseg_processor
        self.prompt_batch_size = prompt_batch_size
        # A predictor belongs to one evaluation: projections must refresh after training.
        self.text_batches = {}

    def _text_inputs(self, categories):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from src.sam3_adapter import prepare_inputs

        key = tuple((c["id"], c["name"]) for c in categories)
        if key not in self.text_batches:
            if hasattr(self.model, "_plantseg_text_hidden"):
                ids = torch.tensor([c["id"] for c in categories], device=self.device)
                features = self.model.text_projection(self.model._plantseg_text_hidden[ids])
                attention_mask = self.model._plantseg_text_mask[ids]
            else:
                text = prepare_inputs(
                    self.processor,
                    text=[c["name"] for c in categories],
                    max_text_length=self.model.config.text_config.max_position_embeddings,
                )
                text = {k: v.to(self.device) for k, v in text.items()}
                features = self.model.get_text_features(**text).pooler_output
                attention_mask = text["attention_mask"]
            self.text_batches[key] = {
                "text_embeds": BaseModelOutputWithPooling(pooler_output=features),
                "attention_mask": attention_mask,
            }
        return self.text_batches[key]

    @staticmethod
    def _expand_vision(vision, count):
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        # Broadcast the single image's features; never run the backbone per prompt.
        return Sam3VisionEncoderOutput(
            fpn_hidden_states=tuple(t.expand(count, *t.shape[1:]) for t in vision.fpn_hidden_states),
            fpn_position_encoding=tuple(
                t.expand(count, *t.shape[1:]) for t in vision.fpn_position_encoding
            ),
        )

    @torch.inference_mode()
    def __call__(self, image, categories):
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
            image_inputs = self.processor(images=image, return_tensors="pt")
            vision = self.model.get_vision_features(
                pixel_values=image_inputs.pixel_values.to(self.device)
            )
            best = torch.zeros((image.height, image.width), device=self.device)
            prediction = torch.zeros(best.shape, dtype=torch.int64, device=self.device)
            for start in range(0, len(categories), self.prompt_batch_size):
                selected = categories[start : start + self.prompt_batch_size]
                output = self.model(
                    vision_embeds=self._expand_vision(vision, len(selected)),
                    **self._text_inputs(selected),
                )
                scores = (
                    output.pred_logits.float().sigmoid() * output.presence_logits.float().sigmoid()
                )
                for index, category in enumerate(selected):
                    keep = scores[index] > self.score_threshold
                    evidence = class_evidence(
                        output.pred_masks[index][keep],
                        scores[index][keep],
                        best.shape,
                        self.mask_threshold,
                    )
                    replace = evidence > best  # ties retain the first class in input order
                    prediction = torch.where(replace, category["id"] + 1, prediction)
                    best = torch.maximum(best, evidence)
                del output
        return prediction.cpu().numpy().astype(np.uint8)


def evaluate(model, args, split, device, dtype, output_prefix=None):
    """No padded sampler and no DDP forward: every eval image is counted exactly once."""
    started = time.monotonic()
    metadata = load_metadata(args.dataset_dir)
    categories = categories_from_metadata(metadata)
    samples = split_samples(args.dataset_dir, split, metadata)
    limit = (
        getattr(args, "limit_val", None) if split == "val" else getattr(args, "limit_test", None)
    )
    if limit is not None:
        samples = samples[:limit]
    metrics = SegmentationMetrics(categories)
    label_report = []
    if model is not None:
        model.eval()
        predictor = Sam3SemanticPredictor(
            model, device, dtype, args.score_threshold, args.mask_threshold,
            getattr(args, "eval_prompt_batch_size", 1),
        )
    prediction_dir = getattr(args, "prediction_dir", None)
    save_dir = getattr(args, "save_predictions", None)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for sample in tqdm(
        samples[rank() :: world_size()],
        desc=f"{split}/{args.eval_mode}",
        disable=rank() != 0,
    ):
        image, target = load_pair(sample, args.label_policy, label_report)
        labels = np.unique(target[target > 0])
        disease_id = int(labels[0]) - 1 if labels.size else sample["index"]
        if prediction_dir is not None:
            path = prediction_dir / sample["mask_path"].name
            with Image.open(path) as source:
                prediction = np.array(source)
        else:
            selected = [categories[disease_id]] if args.eval_mode == "oracle" else categories
            prediction = predictor(image, selected)
        metrics.update(prediction, target, disease_id, sample["image_path"].name)
        if save_dir is not None:
            Image.fromarray(prediction.astype(np.uint8)).save(save_dir / sample["mask_path"].name)
    if distributed():
        # NCCL requires CUDA tensors; use CPU for Gloo smoke tests.
        for array in (metrics.confusion, metrics.binary):
            tensor = torch.as_tensor(array, device=device)
            dist.all_reduce(tensor)
            array[:] = tensor.cpu().numpy()
        metrics.image_scores = [
            row for part in gather_objects(metrics.image_scores) for row in part
        ]
    label_report = [row for part in gather_objects(label_report) for row in part]
    metrics.image_scores.sort(key=lambda row: row["name"])
    result = metrics.compute()
    result["dataset_fingerprint"] = dataset_fingerprint(args.dataset_dir, samples, "png", split)
    if model is not None:
        result["model_identity"] = getattr(model, "_plantseg_identity", None)
        result["adaptation_checkpoint"] = str(args.resume) if args.resume else None
    if prediction_dir is not None:
        result["prediction_dir"] = str(prediction_dir.resolve())
    result.update(
        {
            "label_policy": args.label_policy,
            "label_disagreements": sorted(label_report, key=lambda r: r["name"]),
        }
    )
    result.update(
        {
            "split": split,
            "eval_mode": args.eval_mode,
            "dataset_dir": str(args.dataset_dir.resolve()),
            "score_threshold": args.score_threshold,
            "mask_threshold": args.mask_threshold,
            "limited": limit is not None,
            "world_size": world_size(),
            "eval_prompt_batch_size": getattr(args, "eval_prompt_batch_size", 1),
            "evaluation_seconds": time.monotonic() - started,
        }
    )
    if output_prefix is not None and rank() == 0:
        prefix = Path(output_prefix)
        write_json(prefix.with_suffix(".json"), result)
        np.save(prefix.with_name(prefix.name + "-confusion.npy"), metrics.confusion)
        for suffix, records in (
            ("per-class", result["per_class"]),
            ("per-image", metrics.image_scores),
        ):
            with prefix.with_name(prefix.name + f"-{suffix}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                if records:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--label_policy", choices=("mask", "metadata", "strict"), default="mask")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    from src.sam3_adapter import add_model_arguments

    add_model_arguments(parser)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Fine-tuned adaptation best.pt / last.pt; omit for zero-shot",
    )
    parser.add_argument(
        "--prediction_dir",
        type=Path,
        help="External baseline class-index PNGs (0 background, 1..115 disease)",
    )
    parser.add_argument("--scope", choices=("heads", "full"), default="heads")
    parser.add_argument(
        "--eval_mode",
        choices=("oracle", "all_classes", "external"),
        default="all_classes",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--save_predictions", type=Path)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--limit_val", type=int)
    parser.add_argument("--limit_test", type=int)
    parser.add_argument("--eval_prompt_batch_size", type=int, default=1)
    args = parser.parse_args()
    if args.eval_prompt_batch_size < 1:
        parser.error("--eval_prompt_batch_size must be positive")
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.mask_threshold <= 1:
        parser.error("thresholds must be in [0, 1]")
    if any(v is not None and v < 1 for v in (args.limit_val, args.limit_test)):
        parser.error("sample limits must be positive")
    if args.prediction_dir is None and args.eval_mode == "external":
        parser.error("--eval_mode external requires --prediction_dir")
    if args.prediction_dir is not None and args.resume is not None:
        parser.error("--prediction_dir and --resume are mutually exclusive")
    device = setup("cpu" if args.prediction_dir else "cuda")
    try:
        model, dtype = None, None
        if args.prediction_dir is not None:
            args.eval_mode = "external"
        else:
            from src.runtime import load_adaptation
            from src.training import prepare_model, resolve_dtype

            args.no_pretrained = False
            args.frozen_dtype = "fp16" if args.amp != "off" else "fp32"
            args.gradient_checkpointing = False
            dtype = resolve_dtype(args.amp, device)
            model, _ = prepare_model(args, device)
            if args.resume:
                load_adaptation(model, args.resume)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = evaluate(model, args, args.split, device, dtype, args.output_dir / args.split)
        if rank() == 0:
            print(
                {
                    k: result[k]
                    for k in (
                        "eval_mode",
                        "num_images",
                        "miou",
                        "macc",
                        "foreground_iou",
                        "image_mean_dice",
                    )
                }
            )
        barrier()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
