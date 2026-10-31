"""Streaming pixel-confusion metrics. Values use [0, 1], never percentages."""

from __future__ import annotations

import numpy as np


def divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, float), np.asarray(denominator, float)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=denominator != 0,
    )


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else None


def scalar(value):
    return float(value) if np.isfinite(value) else None


class SegmentationMetrics:
    def __init__(self, categories: list[dict]):
        self.categories = categories
        self.confusion = np.zeros((len(categories) + 1, len(categories) + 1), dtype=np.int64)
        self.binary = np.zeros((2, 2), dtype=np.int64)
        self.image_scores = []

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        disease_id: int,
        name: str = "",
    ) -> dict:
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape or target.ndim != 2:
            raise ValueError(
                f"Prediction/target shapes differ for {name}: {prediction.shape}, {target.shape}"
            )
        count = len(self.confusion)
        for array in (prediction, target):
            if (
                not np.issubdtype(array.dtype, np.integer)
                or array.min() < 0
                or array.max() >= count
            ):
                raise ValueError(f"Expected integer labels 0..{count - 1} for {name}")
        self.confusion += np.bincount(
            (target.astype(np.int64) * count + prediction).ravel(),
            minlength=count * count,
        ).reshape(count, count)
        pred, gt = prediction > 0, target > 0
        confusion = np.bincount((gt.astype(np.int64) * 2 + pred).ravel(), minlength=4).reshape(2, 2)
        self.binary += confusion
        tn, fp, fn, tp = confusion.ravel()
        union = tp + fp + fn
        record = {
            "name": name,
            "disease_id": disease_id,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "foreground_ratio": float(gt.mean()),
            "iou": float(tp / union) if union else 1.0,
            "dice": float(2 * tp / (2 * tp + fp + fn)) if union else 1.0,
            "precision": scalar(divide(tp, tp + fp)),
            "recall": scalar(divide(tp, tp + fn)),
        }
        self.image_scores.append(record)
        return record

    def compute(self) -> dict:
        tp = np.diag(self.confusion).astype(float)
        gt, pred = self.confusion.sum(1), self.confusion.sum(0)
        iou = divide(tp, gt + pred - tp)
        recall = divide(tp, gt)
        dice = divide(2 * tp, gt + pred)
        precision = divide(tp, pred)
        per_class = []
        for i, name in enumerate(["background"] + [c["name"] for c in self.categories]):
            per_class.append(
                {
                    "label": i,
                    "name": name,
                    "gt_pixels": int(gt[i]),
                    "pred_pixels": int(pred[i]),
                    "iou": scalar(iou[i]),
                    "accuracy": scalar(recall[i]),
                    "dice": scalar(dice[i]),
                    "precision": scalar(precision[i]),
                }
            )
        tn, fp, fn, btp = self.binary.ravel()
        disease_means = []
        for c in self.categories:
            scores = [r["iou"] for r in self.image_scores if r["disease_id"] == c["id"]]
            if scores:
                disease_means.append(np.mean(scores))
        result = {
            "num_images": len(self.image_scores),
            "miou": finite_mean(iou[1:]),
            "macc": finite_mean(recall[1:]),
            "miou_with_background": finite_mean(iou),
            "macc_with_background": finite_mean(recall),
            "mean_dice": finite_mean(dice[1:]),
            "mean_precision": finite_mean(precision[1:]),
            "pixel_accuracy": scalar(divide(tp.sum(), gt.sum())),
            "foreground_iou": scalar(divide(btp, btp + fp + fn)),
            "foreground_dice": scalar(divide(2 * btp, 2 * btp + fp + fn)),
            "pixel_precision": scalar(divide(btp, btp + fp)),
            "pixel_recall": scalar(divide(btp, btp + fn)),
            "image_mean_iou": finite_mean([r["iou"] for r in self.image_scores]),
            "image_mean_dice": finite_mean([r["dice"] for r in self.image_scores]),
            "disease_macro_image_iou": finite_mean(disease_means),
            "classes_with_gt": int((gt[1:] > 0).sum()),
            "classes_with_union": int(((gt + pred - tp)[1:] > 0).sum()),
            "per_class": per_class,
            "metric_policy": {
                "units": "fraction",
                "background_label": 0,
                "disease_label": "Metadata.Index + 1",
                "miou": "pooled per-class TP/(TP+FP+FN), mean over disease classes with nonzero union",
                "macc": "pooled per-class TP/(TP+FN), mean over disease classes with GT support",
                "undefined": "null; omitted from macro averages",
                "empty_binary_pair": "image IoU and Dice = 1",
            },
        }
        return result
