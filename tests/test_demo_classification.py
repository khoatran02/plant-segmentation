from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.demo_classification import predict_disease


def classifier_fixture(batch_size=2):
    categories = [{"id": i, "name": f"disease {i}"} for i in range(3)]

    class Model:
        encodes = 0
        calls = 0
        queries = []

        def get_vision_features(self, pixel_values):
            self.encodes += 1
            return pixel_values

        def __call__(self, vision_embeds, selected):
            self.calls += 1
            self.queries = [*self.queries, *[c["name"] for c in selected]]
            ids = [c["id"] for c in selected]
            masks = torch.full((len(ids), 1, 2, 3), -10.0)
            if ids == [-1]:
                prompt = selected[0]["name"]
                if prompt == "diseased regions":
                    masks[:] = 10.0
                elif prompt == "left affected region":
                    masks[0, 0, 0, 0] = 10.0
                return SimpleNamespace(presence_logits=torch.ones(1), semantic_seg=masks)
            logits = torch.tensor([2.0, 3.0, 0.0])[ids]
            for index, category_id in enumerate(ids):
                masks[index, 0, 0, category_id] = 10.0
            return SimpleNamespace(presence_logits=logits, semantic_seg=masks)

    predictor = SimpleNamespace(
        device=torch.device("cpu"), dtype=None, model=Model(),
        processor=lambda **_: SimpleNamespace(pixel_values=torch.zeros(1, 3, 2, 3)),
        prompt_batch_size=batch_size, text_batches={}, known_disease=True, score_threshold=0.5,
        _expand_vision=lambda vision, count: vision,
    )

    def text_inputs(selected):
        predictor.text_batches["current"] = selected
        return {"selected": selected}

    predictor._text_inputs = text_inputs
    return predictor, categories


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_prompt_then_disease_ranking_uses_one_image_encode(batch_size):
    predictor, categories = classifier_fixture(batch_size)
    mask, classification = predict_disease(
        predictor, Image.new("RGB", (3, 2)), categories, "diseased regions", 0.5)
    assert classification["predicted_class"]["id"] == 1
    assert [c["id"] for c in classification["candidates"]] == [1, 0, 2]
    assert classification["experimental"] and classification["known_disease_checkpoint"]
    np.testing.assert_array_equal(mask, [[0, 1, 0], [0, 0, 0]])
    assert predictor.model.encodes == 1
    assert predictor.model.calls == 1 + (3 + batch_size - 1) // batch_size
    assert predictor.model.queries[0] == "diseased regions"
    assert set(predictor.model.queries[1:]) == {c["name"] for c in categories}
    assert not predictor.text_batches


def test_changing_prompt_changes_the_selected_disease_and_mask():
    predictor, categories = classifier_fixture()
    mask, classification = predict_disease(
        predictor, Image.new("RGB", (3, 2)), categories, "left affected region", 0.5)
    # Disease 1 has the higher presence score, but is outside the requested region.
    assert classification["predicted_class"]["id"] == 0
    np.testing.assert_array_equal(mask, [[1, 0, 0], [0, 0, 0]])


def test_no_prompt_region_does_not_fall_back_to_image_only_classification():
    predictor, categories = classifier_fixture()
    mask, classification = predict_disease(
        predictor, Image.new("RGB", (12, 8)), categories, "missing concept", 0.5)
    assert classification["predicted_class"] is None
    assert classification["reason"] == "no_prompt_region"
    assert not classification["candidates"] and not mask.any()
    assert predictor.model.calls == 1


def test_abstains_when_presence_score_is_below_threshold():
    predictor, categories = classifier_fixture()
    predictor.score_threshold = 0.99
    mask, classification = predict_disease(
        predictor, Image.new("RGB", (12, 8)), categories, "diseased regions", 0.5)
    assert classification["predicted_class"] is None
    assert classification["reason"] == "below_presence_threshold"
    assert mask.shape == (8, 12) and not mask.any()


def test_ranking_failure_clears_text_cache():
    predictor, categories = classifier_fixture()

    def fail(selected):
        predictor.text_batches["current"] = "device tensor"
        raise RuntimeError("encoding failed")

    predictor._text_inputs = fail
    with pytest.raises(RuntimeError, match="encoding failed"):
        predict_disease(predictor, Image.new("RGB", (3, 2)), categories, "diseased regions", 0.5)
    assert not predictor.text_batches


def test_ineligible_top_score_does_not_hide_an_eligible_disease():
    predictor, categories = classifier_fixture()
    forward = predictor.model.__call__

    class UnequalRegions:
        def get_vision_features(self, pixel_values):
            return pixel_values

        def __call__(self, vision_embeds, selected):
            output = forward(vision_embeds, selected)
            ids = [item["id"] for item in selected]
            if ids == [-1]:
                output.semantic_seg[:] = -10
                output.semantic_seg[0, 0, 0, 0] = 10
            else:
                for index, category_id in enumerate(ids):
                    output.semantic_seg[index] = -10
                    output.semantic_seg[index, 0, 0, 0] = 10
                    # Class 0: presence .4, overlap 1; class 1: presence .9, overlap 1/3.
                    output.presence_logits[index] = [-0.405465, 2.197225, -10][category_id]
                    if category_id == 1:
                        output.semantic_seg[index, 0, 0, :] = 10
            return output

    predictor.model = UnequalRegions()
    mask, classification = predict_disease(
        predictor, Image.new("RGB", (3, 2)), categories, "diseased regions", 0.5)
    assert classification["candidates"][0]["id"] == 0
    assert classification["predicted_class"]["id"] == 1
    assert classification["reason"] is None
    assert mask.any()
