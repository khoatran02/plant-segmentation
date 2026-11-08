import json

import numpy as np
import pytest
import torch

from src.evaluation import class_evidence
from src.metrics import SegmentationMetrics

CATEGORIES = [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}, {"id": 2, "name": "absent"}]


def test_multiclass_confusion_counts_disease_confusion():
    metrics = SegmentationMetrics(CATEGORIES)
    target = np.array([[0, 1], [1, 2]])
    prediction = np.array([[0, 2], [1, 0]])
    metrics.update(prediction, target, 0)
    result = metrics.compute()
    assert result["miou"] == pytest.approx((0.5 + 0) / 2)
    assert result["macc"] == pytest.approx((0.5 + 0) / 2)
    assert result["pixel_accuracy"] == 0.5
    assert result["foreground_iou"] == pytest.approx(2 / 3)
    assert result["miou_with_background"] == pytest.approx((0.5 + 0.5 + 0) / 3)
    assert result["per_class"][3]["iou"] is None
    json.dumps(result, allow_nan=False)


def test_false_positive_on_gt_absent_class_is_not_omitted_from_miou():
    metrics = SegmentationMetrics(CATEGORIES)
    metrics.update(np.array([[1, 3]]), np.array([[1, 0]]), 0)
    result = metrics.compute()
    assert result["miou"] == 0.5
    assert result["macc"] == 1
    assert result["classes_with_gt"] == 1
    assert result["classes_with_union"] == 2


def test_empty_prediction_and_empty_pair_policy():
    metrics = SegmentationMetrics(CATEGORIES)
    record = metrics.update(np.zeros((2, 2), int), np.ones((2, 2), int), 0)
    assert record["iou"] == record["dice"] == record["recall"] == 0
    assert record["precision"] is None
    record = metrics.update(np.zeros((2, 2), int), np.zeros((2, 2), int), 0)
    assert record["iou"] == record["dice"] == 1
    assert record["precision"] is record["recall"] is None


@pytest.mark.parametrize(
    "prediction", [np.zeros((3, 2), int), np.ones((2, 2), float), np.full((2, 2), 255)]
)
def test_invalid_prediction_rejected(prediction):
    with pytest.raises(ValueError):
        SegmentationMetrics(CATEGORIES).update(prediction, np.ones((2, 2), int), 0)


def test_interpolate_logits_before_thresholding_and_chunk_invariance():
    masks = torch.tensor([[[2.0, -1], [-3, 4]], [[-2.0, 3], [2, -1]]])
    scores = torch.tensor([0.8, 0.7])
    result = class_evidence(masks, scores, (5, 7), chunk_size=1)
    upsampled = torch.nn.functional.interpolate(
        masks[:, None], (5, 7), mode="bilinear", align_corners=False
    ).sigmoid()[:, 0]
    expected = (upsampled * scores[:, None, None] * (upsampled > 0.5)).amax(0)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result, class_evidence(masks, scores, (5, 7), chunk_size=8))
    assert not class_evidence(masks[:0], scores[:0], (5, 7)).any()
