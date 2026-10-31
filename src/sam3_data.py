"""PNG/COCO pseudo-instance datasets and Hugging Face processor batches."""

from __future__ import annotations

import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageOps
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from src.data import build_coco_split, load_metadata, load_pair, split_samples


class PlantSegDataset(Dataset):
    def __init__(self, args, split="train", training=True):
        path = build_coco_split(
            args.dataset_dir,
            split,
            args.output_dir,
            False,
            args.annotation_source,
            args.label_policy,
        )
        with path.open() as stream:
            data = json.load(stream)
        self.categories = data["categories"]
        self.annotations = defaultdict(list)
        for annotation in data["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)
        self.images = data["images"]
        limit = args.limit_train if training else args.limit_val
        if limit is not None:
            indices = list(range(len(self.images)))
            random.Random(args.seed).shuffle(indices)
            self.images = [self.images[i] for i in sorted(indices[:limit])]
        self.samples = {
            s["image_path"].name: s
            for s in split_samples(args.dataset_dir, split, load_metadata(args.dataset_dir))
        }
        self.training, self.label_policy = training, args.label_policy
        self.mask_size = args.loss_mask_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        image, _ = load_pair(self.samples[info["file_name"]], self.label_policy)
        flip = self.training and random.random() < 0.5
        if flip:
            image = ImageOps.mirror(image)
        boxes, masks = [], []
        union = np.zeros((image.height, image.width), dtype=np.uint8)
        for annotation in self.annotations[info["id"]]:
            segmentation = annotation["segmentation"]
            if isinstance(segmentation, list):
                segmentation = mask_utils.merge(
                    mask_utils.frPyObjects(segmentation, image.height, image.width)
                )
            region = mask_utils.decode(segmentation)
            if flip:
                region = region[:, ::-1].copy()
            union |= region
            target = torch.as_tensor(region.astype(np.float32))[None, None]
            masks.append(F.interpolate(target, (self.mask_size, self.mask_size), mode="area")[0, 0])
            x, y, width, height = annotation["bbox"]
            if flip:
                x = image.width - x - width
            boxes.append(
                [
                    x / image.width,
                    y / image.height,
                    (x + width) / image.width,
                    (y + height) / image.height,
                ]
            )
        semantic = F.interpolate(
            torch.as_tensor(union.astype(np.float32))[None, None],
            (self.mask_size, self.mask_size),
            mode="area",
        )[0]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp(0, 1),
            "masks": torch.stack(masks)
            if masks
            else torch.zeros((0, self.mask_size, self.mask_size)),
            "semantic_mask": semantic,
        }
        return {
            "image": image,
            "category_id": info["disease_id"],
            "target": target,
            "name": info["file_name"],
        }


def choose_categories(categories, present, count, positive_probability, rng=random):
    positives = [present] if rng.random() < positive_probability else []
    remaining_count = max(0, count - len(positives))
    plant = categories[present]["supercategory"]
    hard = [c["id"] for c in categories if c["id"] != present and c["supercategory"] == plant]
    easy = [c["id"] for c in categories if c["id"] != present and c["supercategory"] != plant]
    negatives = rng.sample(hard, min(len(hard), (remaining_count + 1) // 2))
    negatives += rng.sample(easy, min(len(easy), remaining_count - len(negatives)))
    unused = [c["id"] for c in categories if c["id"] != present and c["id"] not in negatives]
    negatives += rng.sample(unused, min(len(unused), remaining_count - len(negatives)))
    if not positives and not negatives:
        raise ValueError("No available prompt; positive-only datasets require --pos_prob 1")
    return positives + negatives


class PlantSegCollator:
    def __init__(self, processor, categories, prompts_per_image, pos_prob, max_text_length):
        self.processor, self.categories = processor, categories
        self.prompts_per_image, self.pos_prob = prompts_per_image, pos_prob
        self.max_text_length = max_text_length

    def __call__(self, samples):
        images, prompts, ids, targets = [], [], [], []
        for sample in samples:
            selected = choose_categories(
                self.categories, sample["category_id"], self.prompts_per_image, self.pos_prob
            )
            for category_id in selected:
                images.append(sample["image"])
                prompts.append(self.categories[category_id]["name"])
                ids.append(category_id)
                target = sample["target"]
                if category_id != sample["category_id"]:
                    target = {
                        "boxes": target["boxes"][:0],
                        "masks": target["masks"][:0],
                        "semantic_mask": torch.zeros_like(target["semantic_mask"]),
                    }
                targets.append(target)
        from src.sam3_adapter import prepare_inputs

        inputs = prepare_inputs(
            self.processor, images=images, text=prompts, max_text_length=self.max_text_length
        )
        return {
            "inputs": inputs,
            "prompt_ids": torch.tensor(ids, dtype=torch.long),
            "targets": targets,
        }


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value
