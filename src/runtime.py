"""Shared torchrun lifecycle and epoch-boundary checkpoint helpers."""

from __future__ import annotations

import json
import hashlib
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if distributed() else 1


def setup(device_name: str = "cuda") -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Training SAM3 requires CUDA; enable Kaggle GPU T4 x2.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(minutes=60),
            device_id=device if device.type == "cuda" else None,
        )
    return device


def cleanup():
    if distributed():
        dist.destroy_process_group()


def barrier():
    if distributed():
        dist.barrier()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state(generator=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "loader": generator.get_state() if generator is not None else None,
    }


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])
    if generator is not None and state.get("loader") is not None:
        generator.set_state(state["loader"])


def gather_objects(value):
    if not distributed():
        return [value]
    result = [None] * world_size()
    dist.all_gather_object(result, value)
    return result


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def accumulation_window(index: int, batches: int, accumulation: int) -> tuple[int, bool]:
    """Normalize a short final accumulation group by its actual number of batches."""
    start = index // accumulation * accumulation
    length = min(accumulation, batches - start)
    return length, index + 1 == start + length


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay. step is the next successful update (1-based)."""
    import math

    if total <= 1:
        return 1.0
    warmup = min(warmup, total)
    if warmup and step <= warmup:
        return step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return 0.5 * (1 + math.cos(math.pi * progress))


def load_adaptation(model, path: Path, expected_args=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "trainable" not in state:
        raise ValueError(
            f"{path} is not an adaptation checkpoint; use --model_id for pretrained SAM3"
        )
    if (
        hasattr(model, "_plantseg_identity")
        and state.get("model_identity") != model._plantseg_identity
    ):
        raise ValueError(
            "Base checkpoint/model revision, Transformers backend or prompt vocabulary differs from adaptation checkpoint"
        )
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if expected != set(state["trainable"]):
        raise ValueError(
            "Checkpoint parameter set differs: use the same --scope and model revision"
        )
    if expected_args is not None:
        if state.get("format_version") != 3:
            raise ValueError(
                "Exact resume requires Transformers checkpoint format 3; the old upstream backend is incompatible"
            )
        saved = state.get("config", {})
        for key in (
            "scope",
            "annotation_source",
            "label_policy",
            "eval_mode",
            "selection_metric",
            "batch_size",
            "grad_accum",
            "seed",
            "score_threshold",
            "mask_threshold",
            "epochs",
            "warmup_ratio",
            "warmup_steps",
            "lr",
            "lr_vision",
            "lr_language",
            "weight_decay",
            "clip_norm",
            "amp",
            "amp_init_scale",
            "prompts_per_image",
            "pos_prob",
            "gradient_checkpointing",
            "frozen_dtype",
            "loss_mask_size",
            "attn_implementation",
            "limit_train",
            "limit_val",
            "workers",
            "val_every",
            "min_delta",
            "early_stopping_patience",
        ):
            if key in saved and saved[key] != getattr(expected_args, key):
                raise ValueError(
                    f"Resume mismatch for {key}: checkpoint={saved[key]}, requested={getattr(expected_args, key)}"
                )
        if state.get("world_size", world_size()) != world_size():
            raise ValueError("Resume requires the original world size to restore rank RNG states")
    unexpected = model.load_state_dict(state["trainable"], strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"Unexpected adaptation keys: {unexpected[:5]}")
    return state
