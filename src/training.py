"""Fine-tune SAM3 on PlantSeg PNG masks with single-GPU or torchrun DDP."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.data import (
    DEFAULT_DATASET_DIR,
    ROOT,
    build_coco_split,
    categories_from_metadata,
    load_metadata,
)
from src.runtime import (
    accumulation_window,
    barrier,
    cleanup,
    distributed,
    gather_objects,
    load_adaptation,
    lr_factor,
    rank,
    restore_rng,
    rng_state,
    seed_everything,
    seed_worker,
    setup,
    world_size,
    write_json,
)
from src.sam3_adapter import add_model_arguments


def resolve_dtype(name, device):
    if name == "auto":
        name = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp16"
    if name == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a supporting CUDA GPU; use --amp fp16 on T4")
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[name]


def prepare_model(args, device):
    from src import sam3_adapter as adapter

    model = adapter.build_model(args)
    frozen = adapter.configure_trainable(model, args.scope)
    if args.scope == "heads":
        categories = categories_from_metadata(load_metadata(args.dataset_dir))
        frozen = adapter.cache_text_features(model, categories)
    half_frozen = args.frozen_dtype == "fp16" or (args.frozen_dtype == "auto" and args.amp != "off")
    if half_frozen and args.amp == "off":
        raise ValueError("Frozen fp16 weights require AMP; use --frozen_dtype fp32 with --amp off")
    if half_frozen:
        for module in frozen:
            module.half()
    adapter.configure_activation_checkpointing(model, args.gradient_checkpointing)
    return model.to(device), frozen


class TrainingLoss(torch.nn.Module):
    """Return loss tensors from DDP.forward so unused-parameter detection sees the graph."""

    def __init__(self, network, criterion):
        super().__init__()
        self.network, self.criterion = network, criterion

    def forward(self, batch):
        from src.sam3_adapter import forward_model

        outputs = forward_model(self.network, batch["inputs"], batch.get("prompt_ids"))
        return self.criterion(outputs, batch["targets"])


def make_train_loader(dataset, args, generator, model):
    from src.sam3_data import PlantSegCollator

    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True)
        if distributed()
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=next(model.parameters()).is_cuda,
        collate_fn=PlantSegCollator(
            model._plantseg_processor,
            dataset.categories,
            args.prompts_per_image,
            args.pos_prob,
            model.config.text_config.max_position_embeddings,
        ),
        worker_init_fn=seed_worker,
        generator=generator,
        # Restart workers each epoch so checkpointed generator state reproduces worker RNG.
        persistent_workers=False,
        multiprocessing_context="spawn" if args.workers else None,
    )
    return loader, sampler


def checkpoint_payload(
    model, optimizer, scaler, args, epoch, step, best, bad_epochs, rng_states, manifest
):
    return {
        "format_version": 3,
        "model_identity": model._plantseg_identity,
        "input_image_size": getattr(model, "_plantseg_image_size", None),
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metric": best,
        "bad_epochs": bad_epochs,
        "config": vars(args),
        "rng_states": rng_states,
        "world_size": world_size(),
        "manifest": manifest,
    }


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(args, device):
    from src import sam3_adapter as adapter
    from src.evaluation import evaluate
    from src.losses import CORE_LOSS_KEY
    from src.sam3_data import PlantSegDataset, to_device

    # Identical initial parameters on every rank, then independent sampling RNG streams.
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank() == 0:
        print(
            "[data] Preparing train/val annotations (first run can take several minutes).",
            flush=True,
        )
        for split in ("train", "val"):
            build_coco_split(
                args.dataset_dir,
                split,
                args.output_dir,
                args.rebuild_annotations,
                args.annotation_source,
                args.label_policy,
            )
    barrier()
    dtype = resolve_dtype(args.amp, device)
    if rank() == 0:
        print(f"[model] Loading SAM3: {args.model_id}", flush=True)
    model, frozen = prepare_model(args, device)
    train_set = PlantSegDataset(args, "train", training=True)
    generator = torch.Generator().manual_seed(args.seed + rank())
    loader, sampler = make_train_loader(train_set, args, generator, model)
    if len(loader) == 0:
        raise ValueError("No batches: increase --limit_train or reduce world size")
    optimizer = adapter.build_optimizer(model, args)
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=args.amp_init_scale, enabled=dtype == torch.float16
    )
    loss_module = TrainingLoss(model, adapter.build_criterion().to(device))
    if distributed():
        loss_module = DDP(
            loss_module,
            device_ids=[device.index],
            find_unused_parameters=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    warmup = (
        args.warmup_steps
        if args.warmup_steps is not None
        else int(total_updates * args.warmup_ratio)
    )
    manifest = {}
    for split in ("train", "val"):
        path = (
            args.output_dir / "coco" / f"{split}-{args.annotation_source}-{args.label_policy}.json"
        )
        with path.open() as stream:
            manifest[split] = json.load(stream)["plantseg"]
    config = {
        "arguments": vars(args),
        "source_revision": revision(ROOT),
        "backend": "transformers",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "world_size": world_size(),
        "effective_batch_size": args.batch_size * world_size() * args.grad_accum,
        "input_image_size": model._plantseg_image_size,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "warmup_updates": warmup,
        "dataset": manifest,
        "prompt_vocabulary": categories_from_metadata(load_metadata(args.dataset_dir)),
        "model_identity": model._plantseg_identity,
    }
    start, step, best, bad_epochs = 0, 0, -1.0, 0
    seed_everything(args.seed + rank())
    if args.resume:
        state = load_adaptation(model, args.resume, args)
        if state.get("manifest") != manifest:
            raise ValueError(
                "Dataset fingerprint changed since checkpoint; resume with the original dataset"
            )
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, step = state["epoch"] + 1, state["step"]
        best, bad_epochs = state.get("best_metric", -1.0), state.get("bad_epochs", 0)
        previous_val_every = state.get("config", {}).get("val_every", args.val_every)
        if previous_val_every != args.val_every:
            # Scores remain comparable; patience restarts under the new observation schedule.
            bad_epochs = 0
            config["resume_validation_schedule"] = {
                "checkpoint": str(args.resume),
                "previous_val_every": previous_val_every,
                "val_every": args.val_every,
                "patience_reset": True,
            }
            if rank() == 0:
                print(
                    f"[resume] val_every {previous_val_every} -> {args.val_every}; "
                    "optimizer, scheduler and best metric restored; patience reset.",
                    flush=True,
                )
        restore_rng(state["rng_states"][rank()], generator)
    elif (args.output_dir / "last.pt").exists():
        raise ValueError("Output contains last.pt; use --resume or a new --output_dir")
    elif args.init_from:
        load_adaptation(model, args.init_from, allow_resolution_change=True)
        if rank() == 0:
            print(
                f"[init] Loaded adaptation weights from {args.init_from}; "
                "starting a new optimizer/LR schedule and checkpoint selection.",
                flush=True,
            )
    if rank() == 0:
        write_json(args.output_dir / "run-config.json", config)
        model._plantseg_processor.save_pretrained(args.output_dir / "processor")
        print(
            f"[run] {world_size()} GPU, image_size={model._plantseg_image_size}, effective batch={config['effective_batch_size']}, trainable={sum(p.numel() for p in trainable) / 1e6:.1f}M, eval={args.eval_mode}",
            flush=True,
        )
    log_path = args.output_dir / "log.jsonl"

    def log(record):
        if rank() == 0:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    for epoch in range(start, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if hasattr(train_set, "set_curr_epoch"):
            train_set.set_curr_epoch(epoch)
        adapter.set_train_mode(model, frozen)
        optimizer.zero_grad(set_to_none=True)
        running, seen, overflows = 0.0, 0, 0
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", disable=rank() != 0)
        for index, batch_dict in enumerate(progress):
            batch = to_device(batch_dict, device)
            group_size, update_now = accumulation_window(index, len(loader), args.grad_accum)
            sync = nullcontext() if update_now or not distributed() else loss_module.no_sync()
            with sync:
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    losses = loss_module(batch)
                    loss = losses[CORE_LOSS_KEY]
                finite = torch.isfinite(loss.detach()).to(torch.int32)
                if distributed():
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(
                        f"Nonfinite loss at epoch {epoch + 1}, batch {index}; retry with --amp off --frozen_dtype fp32"
                    )
                scaler.scale(loss / group_size).backward()
            running += loss.detach().item()
            seen += 1
            if not update_now:
                continue
            factor = lr_factor(step + 1, total_updates, warmup)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_norm)
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients with AMP disabled")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            did_update = scaler.get_scale() >= previous_scale
            step += int(did_update)
            overflows += int(not did_update)
            optimizer.zero_grad(set_to_none=True)
            if did_update and step % args.log_every == 0:
                log(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "rank0_loss": running / seen,
                        "lr": optimizer.param_groups[0]["lr"],
                        "amp_scale": scaler.get_scale(),
                    }
                )
                progress.set_postfix(loss=f"{running / seen:.3f}", step=step)
        totals = torch.tensor([running, seen], dtype=torch.float64, device=device)
        if distributed():
            dist.all_reduce(totals)
        record = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": (totals[0] / totals[1]).item(),
            "seconds": time.monotonic() - started,
            "amp_overflows": overflows,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        improved = False
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            metrics = evaluate(model, args, "val", device, dtype, args.output_dir / "val-latest")
            record.update(
                {
                    f"val_{k}": v
                    for k, v in metrics.items()
                    if v is None or isinstance(v, (int, float, str))
                }
            )
            selected = metrics[args.selection_metric]
            if selected is None or not math.isfinite(selected):
                raise ValueError(f"Selection metric {args.selection_metric} is undefined")
            improved = selected > best + args.min_delta
            if improved:
                best, bad_epochs = selected, 0
            else:
                bad_epochs += 1
        # Every rank participates once; only rank 0 writes checkpoints/logs.
        rng_states = gather_objects(rng_state(generator))
        if rank() == 0:
            checkpoint_started = time.monotonic()
            state = checkpoint_payload(
                model,
                optimizer,
                scaler,
                args,
                epoch,
                step,
                best,
                bad_epochs,
                rng_states,
                manifest,
            )
            save_checkpoint(args.output_dir / "last.pt", state)
            if improved:
                save_checkpoint(args.output_dir / "best.pt", state)
                write_json(args.output_dir / "best-metrics.json", metrics)
            record["checkpoint_seconds"] = time.monotonic() - checkpoint_started
            record["epoch_seconds"] = time.monotonic() - started
            log(record)
            best_display = f"{best:.5f}" if best >= 0 else "N/A (not validated yet)"
            print(
                f"[epoch {epoch + 1}] loss={record['train_loss']:.4f}, step={step}, "
                f"best {args.selection_metric}={best_display}, train={record['seconds']:.1f}s, "
                f"val={record.get('val_evaluation_seconds', 0):.1f}s, "
                f"save={record['checkpoint_seconds']:.1f}s",
                flush=True,
            )
        barrier()
        if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            if rank() == 0:
                print("[stop] early-stopping patience reached", flush=True)
            break


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--output_dir",
        "--model_path",
        dest="output_dir",
        type=Path,
        default=ROOT / "outputs/sam3-plantseg",
    )
    parser.add_argument("--annotation_source", choices=("png", "coco"), default="png")
    parser.add_argument("--rebuild_annotations", action="store_true")
    parser.add_argument("--label_policy", choices=("mask", "metadata", "strict"), default="mask")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Build and validate caches without CUDA/model",
    )
    parser.add_argument("--prompts_per_image", type=int, default=1)
    parser.add_argument(
        "--loss_mask_size",
        type=int,
        default=144,
        help="Mask supervision resolution; does not change eval resolution",
    )
    parser.add_argument("--pos_prob", type=float, default=0.7)
    parser.add_argument("--limit_train", type=int)
    parser.add_argument("--limit_val", type=int)
    parser.add_argument("--workers", "--num_workers", dest="workers", type=int, default=2)
    add_model_arguments(parser)
    parser.add_argument("--no_pretrained", action="store_true")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument(
        "--init_from",
        type=Path,
        help="Initialize adaptation weights for a new run, allowing a new image size; reset optimizer/LR/metrics",
    )
    parser.add_argument(
        "--resume_allow_val_schedule_change",
        action="store_true",
        help="Allow changing only val_every on resume; retain best metric, reset patience",
    )
    parser.add_argument("--scope", choices=("heads", "full"), default="heads")
    parser.add_argument("--frozen_dtype", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--amp_init_scale", type=float, default=1024)
    parser.add_argument(
        "--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU image batch")
    parser.add_argument(
        "--grad_accum",
        "--gradient_accumulation_steps",
        dest="grad_accum",
        type=int,
        default=8,
    )
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--lr_vision", type=float, default=2.5e-5)
    parser.add_argument("--lr_language", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--clip_norm", "--gradient_clip", dest="clip_norm", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=4,
        help="0 disables; counts validation events",
    )
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument(
        "--eval_prompt_batch_size",
        type=int,
        default=1,
        help="Prompts per inference batch sharing one image; increase after checking GPU memory",
    )
    parser.add_argument("--eval_mode", choices=("oracle", "all_classes"), default="oracle")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument(
        "--selection_metric",
        choices=("miou", "macc", "foreground_iou", "image_mean_iou", "image_mean_dice"),
        default="miou",
    )
    parser.add_argument(
        "--log_every", "--log_every_updates", dest="log_every", type=int, default=25
    )
    parser.add_argument("--seed", type=int, default=1105)
    args = parser.parse_args(argv)
    for name in (
        "epochs",
        "batch_size",
        "grad_accum",
        "val_every",
        "log_every",
        "prompts_per_image",
        "loss_mask_size",
        "eval_prompt_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    for name in ("limit_train", "limit_val", "image_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    for name in ("pos_prob", "warmup_ratio", "score_threshold", "mask_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must be in [0, 1]")
    if args.workers < 0 or args.early_stopping_patience < 0 or args.min_delta < 0:
        parser.error("workers, patience, min_delta cannot be negative")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("--warmup_steps cannot be negative")
    if args.no_pretrained and args.scope != "full":
        parser.error(
            "--no_pretrained requires --scope full; frozen random encoders are not supported"
        )
    if (
        args.amp_init_scale <= 0
        or min(args.lr, args.lr_vision, args.lr_language) <= 0
        or args.clip_norm <= 0
    ):
        parser.error("learning rates, AMP scale and gradient clip must be positive")
    if args.weight_decay < 0:
        parser.error("--weight_decay cannot be negative")
    if args.amp == "off" and args.frozen_dtype == "fp16":
        parser.error("--amp off requires --frozen_dtype fp32 or auto")
    return args


def main():
    args = parse_args()
    if args.prepare_only:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            raise ValueError("Run --prepare_only with python, before torchrun")
        for split in ("train", "val", "test"):
            build_coco_split(
                args.dataset_dir,
                split,
                args.output_dir,
                args.rebuild_annotations,
                args.annotation_source,
                args.label_policy,
            )
        return
    from src.sam3_adapter import check_model_access, load_sam3_api

    load_sam3_api()
    if int(os.environ.get("RANK", 0)) == 0:
        print("[setup] Transformers SAM3 imports OK.", flush=True)
    check_model_access(args)
    if int(os.environ.get("RANK", 0)) == 0:
        print("[setup] Model access check completed (skipped in local/offline mode).", flush=True)
    device = setup(args.device)
    try:
        train(args, device)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
