"""Fine-tune SAM3 (text-prompted instance segmentation) on PlantSeg lesions.

Every image is paired with its disease name as the text prompt ("apple black rot")
plus a few negative disease prompts that must produce no mask. The data pipeline,
model, matchers, losses and LR schedule are all taken from the vendored sam3 repo;
this file only wires them together without the Hydra trainer.

Pretrained weights: pass --checkpoint /path/to/sam3.pt, or log in to Hugging Face
(`huggingface-cli login`, after accepting the licence of facebook/sam3) and the
checkpoint is downloaded by sam3 itself. Use --no_pretrained to train from scratch.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
SAM3_REPO = ROOT / "src" / "sam3"
if str(SAM3_REPO) not in sys.path:
	sys.path.insert(0, str(SAM3_REPO))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam3.model import vitdet
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.loss.loss_fns import (
	CORE_LOSS_KEY,
	Boxes,
	IABCEMdetr,
	Masks,
	SemanticSegCriterion,
	instance_masks_to_semantic_masks,
)
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.optim.schedulers import InverseSquareRootParamScheduler
from sam3.train.transforms import basic as basic_tf
from sam3.train.transforms import basic_for_api as api_tf
from sam3.train.transforms.filter_query_transforms import (
	FilterCrowds,
	FilterEmptyTargets,
	FilterFindQueriesWithTooManyOut,
	FlexibleFilterFindGetQueries,
)
from sam3.train.transforms.segmentation import DecodeRle


DEFAULT_DATASET_DIR = ROOT / "data" / "plantseg-v7"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "sam3-plantseg"
DEFAULT_BPE = SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

RESOLUTION = 1008  # fixed by the SAM3 ViT (RoPE / absolute position embedding)
MAX_ANN_PER_IMG = 200
NORM_MEAN = NORM_STD = [0.5, 0.5, 0.5]

# Parameter prefixes of the two big pretrained encoders.
VISION_TRUNK = "backbone.vision_backbone.trunk."
LANGUAGE_BACKBONE = "backbone.language_backbone."


# --------------------------------------------------------------------------- data


def load_metadata(dataset_dir: Path) -> dict[str, dict[str, Any]]:
	"""Map image file name -> {index, plant, disease} from Metadata.csv."""
	with (dataset_dir / "Metadata.csv").open(encoding="utf-8-sig", newline="") as file:
		return {
			row["Name"]: {
				"index": int(row["Index"]),
				"plant": row["Plant"].strip().lower(),
				"disease": row["Disease"].strip().lower(),
			}
			for row in csv.DictReader(file)
		}


def _find_annotation_file(dataset_dir: Path, split: str) -> Path:
	for candidate in (
		dataset_dir / f"annotation_{split}.json",
		dataset_dir / "json" / f"annotation_{split}.json",
	):
		if candidate.is_file():
			return candidate
	raise FileNotFoundError(f"No annotation_{split}.json under {dataset_dir}")


def build_coco_split(dataset_dir: Path, split: str, cache_dir: Path, rebuild: bool = False) -> Path:
	"""Write a clean COCO json for one split and return its path.

	The raw PlantSeg json has no categories, stores every id as a string, and (in v7)
	still lists the annotations of images that were removed from the split. The class
	of an image comes from Metadata.csv, which is the image-level label.
	"""
	out_path = cache_dir / "coco" / f"{split}.json"
	if out_path.is_file() and not rebuild:
		return out_path

	metadata = load_metadata(dataset_dir)
	image_dir = dataset_dir / "images" / split
	with _find_annotation_file(dataset_dir, split).open(encoding="utf-8") as file:
		raw = json.load(file)

	images = {}
	for info in raw["images"]:
		if (image_dir / info["file_name"]).is_file() and info["file_name"] in metadata:
			images[int(info["id"])] = info

	annotations = []
	stats = Counter()
	for ann in raw["annotations"]:
		image_id = int(ann["image_id"])
		if image_id not in images:
			stats["annotation of image outside split"] += 1
			continue
		polygons = [p for p in ann["segmentation"] if len(p) >= 6 and len(p) % 2 == 0]
		x, y, w, h = (float(v) for v in ann["bbox"])
		if not polygons or w < 1 or h < 1:
			stats["degenerate annotation"] += 1
			continue
		category_id = metadata[images[image_id]["file_name"]]["index"]
		if int(ann["category_id"]) != category_id:
			stats["category differs from Metadata.csv"] += 1
		annotations.append({
			"id": len(annotations) + 1,
			"image_id": image_id,
			"category_id": category_id,
			"segmentation": polygons,
			"bbox": [x, y, w, h],
			"area": float(ann["area"]),
			"iscrowd": 0,
		})

	annotated = {ann["image_id"] for ann in annotations}
	stats["image without annotation"] = len(images) - len(annotated & set(images))
	images = {i: info for i, info in images.items() if i in annotated}

	categories = {}
	for entry in metadata.values():
		categories[entry["index"]] = {
			"id": entry["index"],
			"name": entry["disease"],
			"supercategory": entry["plant"],
		}

	per_image = Counter(ann["image_id"] for ann in annotations)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8") as file:
		json.dump({
			"images": [
				{"id": i, "file_name": info["file_name"], "height": info["height"], "width": info["width"]}
				for i, info in sorted(images.items())
			],
			"annotations": annotations,
			"categories": [categories[k] for k in sorted(categories)],
		}, file)

	print(
		f"[data] {split}: {len(images)} images, {len(annotations)} lesions, "
		f"max {max(per_image.values())} per image, {len(categories)} classes"
	)
	for reason, count in stats.items():
		print(f"[data]   dropped/flagged {count} x {reason}")
	return out_path


class PlantSegLoader(COCO_FROM_JSON):
	"""One datapoint per image with `num_prompts` text prompts instead of all 115 classes.

	Training: the image's own disease is a prompt with probability `pos_prob`; the rest
	are negative diseases (half from the same plant when possible, the hard ones), which
	must produce no mask. Evaluation: always the positive plus fixed negatives.
	"""

	def __init__(self, annotation_file: str, num_prompts: int, pos_prob: float, training: bool):
		super().__init__(annotation_file, include_negatives=True)
		with open(annotation_file, encoding="utf-8") as file:
			categories = json.load(file)["categories"]
		self.num_prompts = num_prompts
		self.pos_prob = pos_prob
		self.training = training
		self._plant = {c["id"]: c["supercategory"] for c in categories}
		self._same_plant = defaultdict(list)
		for cat_id, plant in self._plant.items():
			self._same_plant[plant].append(cat_id)

	def _choose_categories(self, idx: int, present: set[int]) -> list[int]:
		rng = random if self.training else random.Random(idx)
		positives = sorted(present) if (not self.training or rng.random() < self.pos_prob) else []
		count = max(0, self.num_prompts - len(positives))
		hard = [c for p in {self._plant[c] for c in present} for c in self._same_plant[p] if c not in present]
		easy = [c for c in self._sorted_cat_ids if c not in present and c not in hard]
		negatives = rng.sample(hard, min(len(hard), (count + 1) // 2))
		negatives += rng.sample(easy, min(len(easy), count - len(negatives)))
		return positives + negatives

	def loadQueriesAndAnnotationsFromDatapoint(self, idx: int):
		present = {ann["category_id"] for ann in self._raw_data[idx]["annotations"]}
		# A single chunk makes the parent treat `idx` as the image index.
		self.category_chunks = [self._choose_categories(idx, present)]
		return super().loadQueriesAndAnnotationsFromDatapoint(idx)


def build_transforms(training: bool) -> list:
	"""Training / eval transforms of the sam3 Roboflow recipe, plus a horizontal flip."""
	normalize = api_tf.NormalizeAPI(mean=NORM_MEAN, std=NORM_STD)
	max_size = basic_tf.get_random_resize_max_size(RESOLUTION)
	if not training:
		return [api_tf.ComposeAPI([
			DecodeRle(),
			api_tf.RandomResizeAPI(
				sizes=RESOLUTION, max_size=max_size, square=True, consistent_transform=False
			),
			api_tf.ToTensorAPI(),
			normalize,
		])]

	return [
		api_tf.ComposeAPI([
			FlexibleFilterFindGetQueries(query_filter=FilterCrowds()),
			DecodeRle(),
			api_tf.RandomHorizontalFlip(consistent_transform=False),
			api_tf.RandomResizeAPI(
				sizes=basic_tf.get_random_resize_scales(RESOLUTION, min_size=480, rounded=False),
				max_size=max_size,
				square=True,
				consistent_transform=False,
			),
			api_tf.PadToSizeAPI(size=RESOLUTION, consistent_transform=False),
			api_tf.ToTensorAPI(),
			FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
			normalize,
			FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
		]),
		FlexibleFilterFindGetQueries(
			query_filter=FilterFindQueriesWithTooManyOut(max_num_objects=MAX_ANN_PER_IMG)
		),
	]


def build_dataset(args: argparse.Namespace, split: str, training: bool) -> Sam3ImageDataset:
	coco_path = build_coco_split(args.dataset_dir, split, args.output_dir, args.rebuild_annotations)
	return Sam3ImageDataset(
		img_folder=str(args.dataset_dir / "images" / split) + "/",
		ann_file=str(coco_path),
		transforms=build_transforms(training),
		max_ann_per_img=MAX_ANN_PER_IMG if training else 100_000,
		multiplier=1,
		training=training,
		load_segmentation=True,
		max_train_queries=50_000,
		max_val_queries=50_000,
		use_caching=False,
		coco_json_loader=functools.partial(
			PlantSegLoader,
			num_prompts=args.prompts_per_image if training else 1 + args.val_negatives,
			pos_prob=args.pos_prob,
			training=training,
		),
		limit_ids=args.limit_train if training else args.limit_val,
	)


def make_loader(dataset: Sam3ImageDataset, args: argparse.Namespace, training: bool) -> DataLoader:
	return DataLoader(
		dataset,
		batch_size=args.batch_size if training else 1,
		shuffle=training,
		drop_last=training,
		num_workers=args.workers,
		pin_memory=True,
		collate_fn=functools.partial(collate_fn_api, dict_key="plantseg", with_seg_masks=True),
		multiprocessing_context="forkserver" if args.workers else None,
		persistent_workers=bool(args.workers),
	)


# -------------------------------------------------------------------------- model


def patch_vit_mlp() -> None:
	"""Make the ViT MLP usable for training and on pre-Ampere GPUs.

	The vendored vitdet.Mlp calls a fused bf16 kernel that raises when autograd is on
	and needs a bf16-capable GPU. Keep it for inference on Ampere+, otherwise run the
	plain Linear + activation (same weights, differentiable, honours autocast).
	"""
	fused = vitdet.addmm_act

	def addmm_act(activation, linear, x):
		if torch.is_grad_enabled() or torch.cuda.get_device_capability()[0] < 8:
			return activation()(linear(x))
		return fused(activation, linear, x)

	vitdet.addmm_act = addmm_act


def build_model(args: argparse.Namespace) -> torch.nn.Module:
	"""Build SAM3 on the CPU and load the pretrained weights."""
	patch_vit_mlp()
	use_hf = args.checkpoint is None and not args.no_pretrained
	try:
		model = build_sam3_image_model(
			bpe_path=str(DEFAULT_BPE),
			device="cpu",
			eval_mode=False,
			checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
			load_from_HF=use_hf,
			enable_segmentation=True,
		)
	except Exception as error:
		if not use_hf:
			raise
		raise SystemExit(
			f"Could not download the pretrained SAM3 checkpoint ({type(error).__name__}: {error}).\n"
			"Accept the licence of https://huggingface.co/facebook/sam3 and run `huggingface-cli login`, "
			"or pass --checkpoint /path/to/sam3.pt (or --no_pretrained to train from scratch)."
		)
	if args.no_pretrained:
		# The builder leaves some tensors uninitialised (they are expected from the checkpoint).
		with torch.no_grad():
			for name, param in model.named_parameters():
				if not torch.isfinite(param).all():
					print(f"[init] {name}: non-finite values, re-initialised")
					param.normal_(0, 0.02)
	return model


def configure_trainable(model: torch.nn.Module, scope: str) -> list[torch.nn.Module]:
	"""Freeze parameters according to `scope`; return the modules that stay frozen."""
	frozen_modules = []
	if scope == "heads":
		frozen_modules = [model.backbone.vision_backbone.trunk, model.backbone.language_backbone]
	frozen_ids = {id(p) for m in frozen_modules for p in m.parameters()}
	for param in model.parameters():
		param.requires_grad = id(param) not in frozen_ids
	return frozen_modules


@torch.no_grad()
def cache_text_features(model: torch.nn.Module, prompts: list[str]) -> None:
	"""Encode the fixed class prompts once and drop the (frozen) text encoder.

	forward_text() is then answered from the cache, which frees ~0.7 GB of GPU memory.
	Call before moving the model to the GPU; only prompts in `prompts` can be queried.
	"""
	backbone = model.backbone
	backbone.language_backbone.eval()
	encoded = backbone.forward_text(prompts, device="cpu")
	row_of = {prompt: i for i, prompt in enumerate(prompts)}
	buffers = {}
	for key, tensor in encoded.items():
		model.register_buffer(f"_text_cache_{key}", tensor.detach(), persistent=False)
		buffers[key] = f"_text_cache_{key}"

	def forward_text(captions, input_boxes=None, additional_text=None, device="cuda"):
		rows = torch.tensor([row_of[c] for c in captions])
		features, mask, embeds = (getattr(model, buffers[k]) for k in ("language_features", "language_mask", "language_embeds"))
		rows = rows.to(mask.device)
		return {
			"language_features": features[:, rows],
			"language_mask": mask[rows],
			"language_embeds": embeds[:, rows],
		}

	backbone.forward_text = forward_text
	backbone.language_backbone = None


def compute_trunk_in_fp32(trunk: torch.nn.Module) -> None:
	"""Keep the frozen ViT weights in fp16 but run each block in fp32.

	GTX 16xx cards have no tensor cores: their fp16 GEMMs are ~4x slower than fp32
	(0.3 vs 1.4 TFLOPS measured), so the 446M-parameter trunk would dominate the step.
	Blocks are widened right before they run and narrowed right after, so only one block
	is ever fp32 in memory.
	"""
	def widen(module, args):
		module.float()

	def narrow(module, args, output):
		module.half()

	for module in [trunk.patch_embed, trunk.ln_pre, *trunk.blocks]:
		module.register_forward_pre_hook(widen)
		module.register_forward_hook(narrow)

	forward = trunk.forward

	def fp32_forward(*args, **kwargs):
		with torch.autocast("cuda", enabled=False):
			return forward(*(a.float() if torch.is_tensor(a) else a for a in args), **kwargs)

	trunk.forward = fp32_forward


def force_contiguous_convs(model: torch.nn.Module) -> None:
	"""Feed every convolution NCHW-contiguous input.

	GTX 16xx cards (Turing without tensor cores) return NaN from cuDNN fp16 convolutions
	on channels_last tensors, which the SAM3 neck and pixel decoder otherwise produce.
	"""
	for module in model.modules():
		if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
			module.register_forward_pre_hook(lambda _, args: (args[0].contiguous(), *args[1:]))


def set_train_mode(model: torch.nn.Module, frozen_modules: list[torch.nn.Module]) -> None:
	"""Train mode, except frozen encoders (no stochastic depth / dropout noise there)."""
	model.train()
	for module in frozen_modules:
		module.eval()


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
	"""AdamW with the sam3 recipe's per-component LRs; returns (optimizer, group base LRs)."""
	groups = {
		"vision": (args.lr_vision, [], []),
		"language": (args.lr_language, [], []),
		"transformer": (args.lr, [], []),
	}
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if name.startswith("backbone.vision_backbone."):
			key = "vision"
		elif name.startswith(LANGUAGE_BACKBONE):
			key = "language"
		else:
			key = "transformer"
		decay, no_decay = groups[key][1], groups[key][2]
		(no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)

	param_groups = []
	for lr, decay, no_decay in groups.values():
		if decay:
			param_groups.append({"params": decay, "weight_decay": args.weight_decay, "base_lr": lr})
		if no_decay:
			param_groups.append({"params": no_decay, "weight_decay": 0.0, "base_lr": lr})
	return torch.optim.AdamW(param_groups, lr=args.lr)


def build_criterion() -> Sam3LossWrapper:
	"""Box + presence + mask + semantic-seg losses (the sam3 segmentation recipe)."""
	return Sam3LossWrapper(
		matcher=BinaryHungarianMatcherV2(
			focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2, stable=False
		),
		o2m_weight=2.0,
		o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
		use_o2m_matcher_on_o2m_aux=False,
		loss_fns_find=[
			Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
			IABCEMdetr(
				weak_loss=False,
				weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
				pos_weight=10.0,
				alpha=0.25,
				gamma=2,
				use_presence=True,
				pos_focal=False,
				pad_n_queries=200,
				pad_scale_pos=1.0,
			),
			Masks(focal_alpha=0.25, focal_gamma=2.0, weight_dict={"loss_mask": 200.0, "loss_dice": 10.0}, compute_aux=False),
		],
		loss_fn_semantic_seg=SemanticSegCriterion(
			weight_dict={"loss_semantic_seg": 20.0, "loss_semantic_presence": 1.0, "loss_semantic_dice": 30.0},
			presence_head=True,
			presence_loss=False,
			focal=True,
			focal_alpha=0.6,
			focal_gamma=2.0,
			downsample=False,
		),
		# "global" normalisation all-reduces over torch.distributed; we run single process.
		normalization="local",
		scale_by_find_batch_size=True,
	)


def amp_dtype(name: str) -> torch.dtype | None:
	if name == "auto":
		name = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
	return {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[name]


# ---------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(model, loader, device, dtype, threshold: float) -> dict[str, float]:
	"""Lesion-union IoU on positive prompts, detection recall, false-positive rate on negatives."""
	model.eval()
	ious, hits, false_alarms = [], [], []
	for batch_dict in tqdm(loader, desc="val", leave=False):
		_, batch = batch_dict.popitem()
		batch = copy_data_to_device(batch, device, non_blocking=True)
		with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
			out = list(model(batch))[0]
		targets = model.back_convert(batch.find_targets[0])

		scores = out["pred_logits"].float().sigmoid().squeeze(-1)  # (prompts, queries)
		scores = scores * out["presence_logit_dec"].float().sigmoid()
		keep = scores > threshold
		pred = out["pred_masks"].float() > 0  # (prompts, queries, h, w)
		pred_union = (pred & keep[:, :, None, None]).any(1)

		gt = F.interpolate(targets["masks"].float().unsqueeze(1), size=pred.shape[-2:], mode="area").squeeze(1) > 0.5
		gt_union = instance_masks_to_semantic_masks(gt, targets["num_boxes"])
		positive = targets["num_boxes"] > 0

		inter = (pred_union & gt_union).flatten(1).sum(1).float()
		union = (pred_union | gt_union).flatten(1).sum(1).float()
		ious += (inter / union.clamp(min=1))[positive].tolist()
		hits += keep.any(1)[positive].tolist()
		false_alarms += keep.any(1)[~positive].tolist()

	def mean(values: list) -> float:
		return sum(values) / len(values) if values else float("nan")

	return {"val_iou": mean(ious), "val_recall": mean(hits), "val_false_positive": mean(false_alarms)}


# ----------------------------------------------------------------------- training


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, step: int, args, best: float) -> None:
	"""Save only the trainable weights; the frozen ones come from the base checkpoint."""
	trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
	torch.save({
		"trainable": trainable,
		"optimizer": optimizer.state_dict(),
		"scaler": scaler.state_dict(),
		"epoch": epoch,
		"step": step,
		"best_iou": best,
		"args": {k: str(v) for k, v in vars(args).items()},
	}, path)


def train(args: argparse.Namespace) -> None:
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	device = torch.device("cuda")
	dtype = amp_dtype(args.amp)
	gpu_name = torch.cuda.get_device_name(0)
	args.output_dir.mkdir(parents=True, exist_ok=True)

	train_set = build_dataset(args, "train", training=True)
	val_set = build_dataset(args, "val", training=False)
	train_loader = make_loader(train_set, args, training=True)
	val_loader = make_loader(val_set, args, training=False)

	model = build_model(args)
	frozen_modules = configure_trainable(model, args.scope)
	if args.scope == "heads":
		with open(args.output_dir / "coco" / "train.json", encoding="utf-8") as file:
			cache_text_features(model, [c["name"] for c in json.load(file)["categories"]])
		frozen_modules = frozen_modules[:1]  # the vision trunk; the text encoder is gone
	total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
	half_frozen = args.frozen_dtype == "fp16" or (args.frozen_dtype == "auto" and total_gib < 16)
	if half_frozen:
		for module in frozen_modules:
			module.half()
	if half_frozen and args.scope == "heads":
		fp32_trunk = args.trunk_compute == "fp32" or (args.trunk_compute == "auto" and "GTX 16" in gpu_name)
		if fp32_trunk:
			compute_trunk_in_fp32(model.backbone.vision_backbone.trunk)
	force_contiguous_convs(model)
	model.to(device)

	n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
	n_total = sum(p.numel() for p in model.parameters())
	print(
		f"[model] scope={args.scope}  trainable {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M params  "
		f"frozen dtype={'fp16' if half_frozen else 'fp32'}  amp={dtype}  gpu={gpu_name} {total_gib:.1f} GiB"
	)

	trainable_params = [p for p in model.parameters() if p.requires_grad]
	criterion = build_criterion().to(device)
	optimizer = build_optimizer(model, args)
	scaler = torch.amp.GradScaler("cuda", init_scale=2**10, enabled=dtype == torch.float16)  # default 2**16 skips the first steps

	steps_per_epoch = len(train_loader) // args.grad_accum
	total_steps = steps_per_epoch * args.epochs
	schedulers = {}
	for group in optimizer.param_groups:
		schedulers[group["base_lr"]] = InverseSquareRootParamScheduler(
			base_lr=group["base_lr"],
			warmup_steps=min(args.warmup_steps, total_steps // 4),
			cooldown_steps=max(1, int(total_steps * args.cooldown_frac)),
			timescale=min(args.warmup_steps, total_steps // 4) or 1,
		)

	start_epoch, step, best_iou = 0, 0, -1.0
	if args.resume:
		state = torch.load(args.resume, map_location="cpu", weights_only=False)
		unexpected = model.load_state_dict(state["trainable"], strict=False).unexpected_keys
		assert not unexpected, f"unexpected keys in {args.resume}: {unexpected[:5]}"
		optimizer.load_state_dict(state["optimizer"])
		scaler.load_state_dict(state["scaler"])
		start_epoch, step, best_iou = state["epoch"] + 1, state["step"], state["best_iou"]
		print(f"[resume] {args.resume} -> epoch {start_epoch}, step {step}")

	log_file = (args.output_dir / "log.jsonl").open("a", encoding="utf-8")

	def log(record: dict) -> None:
		log_file.write(json.dumps(record) + "\n")
		log_file.flush()

	for epoch in range(start_epoch, args.epochs):
		set_train_mode(model, frozen_modules)
		running, skipped, started = defaultdict(float), 0, time.time()
		optimizer.zero_grad(set_to_none=True)
		progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
		for i, batch_dict in enumerate(progress):
			_, batch = batch_dict.popitem()
			batch = copy_data_to_device(batch, device, non_blocking=True)
			with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
				find_stages = model(batch)
				find_targets = [model.back_convert(t) for t in batch.find_targets]
				losses = criterion(find_stages, find_targets)
			loss = losses[CORE_LOSS_KEY]
			if not torch.isfinite(loss):
				skipped += 1
				optimizer.zero_grad(set_to_none=True)
				continue
			scaler.scale(loss / args.grad_accum).backward()
			for key in (CORE_LOSS_KEY, "loss_bbox", "loss_ce", "loss_mask", "loss_dice", "loss_semantic_seg"):
				if key in losses:
					running[key] += losses[key].item()

			if (i + 1) % args.grad_accum:
				continue
			for group in optimizer.param_groups:
				group["lr"] = schedulers[group["base_lr"]](step + 1, (step + 1) / total_steps)
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_norm)
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)
			step += 1
			if step % args.log_every == 0:
				record = {"step": step, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
				record.update({k: v / (i + 1 - skipped) for k, v in running.items()})
				log(record)
				progress.set_postfix(loss=f"{record[CORE_LOSS_KEY]:.3f}", lr=f"{record['lr']:.1e}")

		summary = {k: v / max(1, len(train_loader) - skipped) for k, v in running.items()}
		print(f"[epoch {epoch + 1}] train {summary}  skipped={skipped}  {time.time() - started:.0f}s")

		record = {"epoch": epoch, "step": step, **{f"train_{k}": v for k, v in summary.items()}}
		if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
			metrics = evaluate(model, val_loader, device, dtype, args.score_threshold)
			record.update(metrics)
			print(f"[epoch {epoch + 1}] {metrics}")
			if metrics["val_iou"] > best_iou:
				best_iou = metrics["val_iou"]
				save_checkpoint(args.output_dir / "best.pt", model, optimizer, scaler, epoch, step, args, best_iou)
		log(record)
		save_checkpoint(args.output_dir / "last.pt", model, optimizer, scaler, epoch, step, args, best_iou)
		torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	data = parser.add_argument_group("data")
	data.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
	data.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	data.add_argument("--rebuild_annotations", action="store_true")
	data.add_argument(
		"--prompts_per_image", type=int, default=1,
		help="text prompts per training image (1 fits a 4 GB GPU; the rest are negatives)",
	)
	data.add_argument("--pos_prob", type=float, default=0.7, help="chance that a training image keeps its own disease prompt")
	data.add_argument("--val_negatives", type=int, default=2, help="negative prompts per validation image")
	data.add_argument("--limit_train", type=int, default=None, help="use only N training images")
	data.add_argument("--limit_val", type=int, default=None, help="use only N validation images")
	data.add_argument("--workers", type=int, default=4)

	weights = parser.add_argument_group("model")
	weights.add_argument("--checkpoint", type=Path, default=None, help="pretrained sam3.pt (default: Hugging Face)")
	weights.add_argument("--no_pretrained", action="store_true", help="random init instead of pretrained weights")
	weights.add_argument("--resume", type=Path, default=None, help="last.pt / best.pt of an earlier run")
	weights.add_argument(
		"--scope", choices=("heads", "full"), default="heads",
		help="heads: freeze the ViT trunk and text encoder (fits small GPUs); full: train everything",
	)
	weights.add_argument("--frozen_dtype", choices=("auto", "fp16", "fp32"), default="auto")
	weights.add_argument("--amp", choices=("auto", "fp16", "bf16", "off"), default="auto")
	weights.add_argument(
		"--trunk_compute", choices=("auto", "fp16", "fp32"), default="auto",
		help="precision of the frozen ViT; auto: fp32 on GTX 16xx, whose fp16 GEMMs are slow",
	)

	optim = parser.add_argument_group("optimisation")
	optim.add_argument("--epochs", type=int, default=20)
	optim.add_argument("--batch_size", type=int, default=1)
	optim.add_argument("--grad_accum", type=int, default=8)
	optim.add_argument("--lr", type=float, default=8e-5, help="transformer / heads")
	optim.add_argument("--lr_vision", type=float, default=2.5e-5, help="vision neck (and trunk with --scope full)")
	optim.add_argument("--lr_language", type=float, default=5e-6, help="text encoder (--scope full)")
	optim.add_argument("--weight_decay", type=float, default=0.1)
	optim.add_argument("--clip_norm", type=float, default=0.1)
	optim.add_argument("--warmup_steps", type=int, default=100)
	optim.add_argument("--cooldown_frac", type=float, default=0.1)

	misc = parser.add_argument_group("logging")
	misc.add_argument("--val_every", type=int, default=1)
	misc.add_argument("--score_threshold", type=float, default=0.5)
	misc.add_argument("--log_every", type=int, default=10)
	misc.add_argument("--seed", type=int, default=0)
	return parser.parse_args()


if __name__ == "__main__":
	train(parse_args())
