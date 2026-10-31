"""Supervised mask/box matching loss for Transformers Sam3Model outputs.

Transformers exposes prediction tensors rather than the original SAM3 training loss.
This objective supervises final-layer instance masks, query/presence confidence,
boxes and the semantic mask. It is not the upstream auxiliary/o2m recipe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou

CORE_LOSS_KEY = "loss"
LOSS_VERSION = "transformers-hungarian-v1"


def dice_loss(logits, targets):
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    return 1 - (2 * (probabilities * targets).sum(1) + 1) / (
        probabilities.sum(1) + targets.sum(1) + 1
    )


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    pt = probability * targets + (1 - probability) * (1 - targets)
    weight = alpha * targets + (1 - alpha) * (1 - targets)
    return bce * (1 - pt).pow(gamma) * weight


class PlantSegCriterion(torch.nn.Module):
    weights = {
        "query": 2.0,
        "presence": 1.0,
        "bbox": 5.0,
        "giou": 2.0,
        "mask_bce": 5.0,
        "mask_dice": 5.0,
        "semantic_bce": 5.0,
        "semantic_dice": 5.0,
    }

    @torch.no_grad()
    def match(self, masks, boxes, logits, target):
        count = len(target["boxes"])
        if count == 0:
            empty = torch.empty(0, dtype=torch.long, device=boxes.device)
            return empty, empty
        if count > len(boxes):
            raise ValueError(f"{count} target components exceed model's {len(boxes)} queries")
        predicted = masks.sigmoid().flatten(1)
        expected = target["masks"].flatten(1)
        intersection = predicted @ expected.T
        dice = 1 - (2 * intersection + 1) / (
            predicted.sum(1, keepdim=True) + expected.sum(1)[None] + 1
        )
        # Pairwise BCE in O(Q*N*H*W) arithmetic, without allocating a Q*N*H*W tensor.
        flat_logits = masks.flatten(1)
        bce = (
            F.softplus(flat_logits).mean(1, keepdim=True)
            - flat_logits @ expected.T / expected.shape[1]
        )
        l1 = torch.cdist(
            box_convert(boxes, "xyxy", "cxcywh"),
            box_convert(target["boxes"], "xyxy", "cxcywh"),
            p=1,
        )
        giou = generalized_box_iou(boxes, target["boxes"])
        cost = 2 * -logits.sigmoid()[:, None] + 5 * l1 - 2 * giou + 2 * bce + 5 * dice
        if not torch.isfinite(cost).all():
            raise FloatingPointError(
                "Nonfinite Hungarian matching cost; reduce AMP scale or use FP32"
            )
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(rows, device=boxes.device), torch.as_tensor(
            columns, device=boxes.device
        )

    def forward(self, output, targets):
        # Explicit FP32 arithmetic keeps BCE, Dice and GIoU stable under FP16 AMP.
        with torch.autocast(output.pred_masks.device.type, enabled=False):
            values = {name: [] for name in self.weights}
            for batch_index, target in enumerate(targets):
                shape = target["semantic_mask"].shape[-2:]
                masks = F.interpolate(
                    output.pred_masks[batch_index].float()[:, None],
                    shape,
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                boxes = output.pred_boxes[batch_index].float()
                logits = output.pred_logits[batch_index].float()
                rows, columns = self.match(masks.detach(), boxes.detach(), logits.detach(), target)
                query_target = torch.zeros_like(logits)
                query_target[rows] = 1
                values["query"].append(
                    focal_loss(logits, query_target).sum() / max(1, len(columns))
                )
                presence = output.presence_logits[batch_index].float().reshape(())
                values["presence"].append(
                    F.binary_cross_entropy_with_logits(
                        presence, presence.new_tensor(float(len(columns) > 0))
                    )
                )
                zero = (masks.sum() + boxes.sum()) * 0
                if len(rows):
                    gt_masks = target["masks"][columns]
                    values["bbox"].append(
                        F.l1_loss(
                            box_convert(boxes[rows], "xyxy", "cxcywh"),
                            box_convert(target["boxes"][columns], "xyxy", "cxcywh"),
                            reduction="none",
                        )
                        .sum(1)
                        .mean()
                    )
                    values["giou"].append(
                        (
                            1 - generalized_box_iou(boxes[rows], target["boxes"][columns]).diag()
                        ).mean()
                    )
                    values["mask_bce"].append(
                        F.binary_cross_entropy_with_logits(masks[rows], gt_masks)
                    )
                    values["mask_dice"].append(dice_loss(masks[rows], gt_masks).mean())
                else:
                    for name in ("bbox", "giou", "mask_bce", "mask_dice"):
                        values[name].append(zero)
                semantic = F.interpolate(
                    output.semantic_seg[batch_index : batch_index + 1].float(),
                    shape,
                    mode="bilinear",
                    align_corners=False,
                )[0]
                gt_semantic = target["semantic_mask"]
                values["semantic_bce"].append(
                    F.binary_cross_entropy_with_logits(semantic, gt_semantic)
                )
                values["semantic_dice"].append(dice_loss(semantic, gt_semantic).mean())
            losses = {name: torch.stack(items).mean() for name, items in values.items()}
            losses[CORE_LOSS_KEY] = sum(
                losses[name] * weight for name, weight in self.weights.items()
            )
        return losses
