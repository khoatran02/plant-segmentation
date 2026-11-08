"""Prompt-grounded disease ranking and lesion masks for the interactive demo."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def native_mask(logits, image, threshold):
    probabilities = F.interpolate(
        logits.reshape(1, 1, *logits.shape[-2:]),
        size=(image.height, image.width), mode="bilinear", align_corners=False,
    ).sigmoid()
    return probabilities[0, 0] > threshold


@torch.inference_mode()
def predict_disease(predictor, image, categories, prompt, mask_threshold):
    """Use the supplied prompt to locate a region, then rank diseases in it.

    One image encode serves both the user's verbatim text query and the candidate
    disease queries. Rank presence times the fraction of a candidate's mask in
    the requested region. The final mask is the intersection of that region and
    the winning disease mask. No sample labels or filenames select the class.

    This is an experimental SAM3 pipeline, not an instruction-tuned VLM or a
    calibrated classifier. Known-disease training does not supervise ranking.
    """
    if not categories:
        raise ValueError("Disease prediction requires a nonempty class vocabulary")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Disease prediction requires a text prompt")
    classification = {
        "predicted_class": None, "candidates": [],
        "method": "sam3_prompt_region_presence_ranking",
        "experimental": True,
        "known_disease_checkpoint": predictor.known_disease,
        "class_count": len(categories), "presence_threshold": predictor.score_threshold,
        "reason": None,
    }
    prediction = np.zeros((image.height, image.width), dtype=np.uint8)
    best_score = 0.0
    matching_region = False
    best_candidate = best_mask = None
    try:
        with torch.autocast(
            predictor.device.type, dtype=predictor.dtype, enabled=predictor.dtype is not None
        ):
            inputs = predictor.processor(images=image, return_tensors="pt")
            vision = predictor.model.get_vision_features(
                pixel_values=inputs.pixel_values.to(predictor.device)
            )
            output = predictor.model(
                vision_embeds=predictor._expand_vision(vision, 1),
                **predictor._text_inputs([{"id": -1, "name": prompt}]),
            )
            region_logits = output.semantic_seg[0].float().clone()
            del output
            predictor.text_batches.clear()
            if not torch.isfinite(region_logits).all():
                raise RuntimeError("Model produced a nonfinite prompt mask")
            region = region_logits.sigmoid() > mask_threshold
            if not region.any():
                classification["reason"] = "no_prompt_region"
                return prediction, classification

            for start in range(0, len(categories), predictor.prompt_batch_size):
                selected = categories[start : start + predictor.prompt_batch_size]
                try:
                    output = predictor.model(
                        vision_embeds=predictor._expand_vision(vision, len(selected)),
                        **predictor._text_inputs(selected),
                    )
                    logits = output.presence_logits.float().reshape(len(selected))
                    if not torch.isfinite(logits).all() or not torch.isfinite(output.semantic_seg).all():
                        raise RuntimeError("Model produced nonfinite disease scores or masks")
                    presence = logits.sigmoid()
                    for index, category in enumerate(selected):
                        candidate_mask = output.semantic_seg[index].float().sigmoid() > mask_threshold
                        overlap = float((candidate_mask & region).sum().item()) / max(
                            int(candidate_mask.sum().item()), 1)
                        presence_score = float(presence[index].item())
                        score = presence_score * overlap
                        candidate = {"id": category["id"], "name": category["name"],
                                     "score": score, "presence_score": presence_score,
                                     "region_overlap": overlap}
                        classification["candidates"].append(candidate)
                        matching_region |= overlap > 0
                        if presence_score > predictor.score_threshold and score > best_score:
                            best_score, best_candidate = score, candidate
                            best_mask = output.semantic_seg[index].float().clone()
                    del output
                finally:
                    predictor.text_batches.clear()
        classification["candidates"].sort(key=lambda item: item["score"], reverse=True)
        classification["candidates"] = classification["candidates"][:3]
        if best_candidate is None:
            classification["reason"] = (
                "below_presence_threshold" if matching_region else "no_matching_region")
        else:
            foreground = native_mask(best_mask, image, mask_threshold) & native_mask(
                region_logits, image, mask_threshold)
            if foreground.any():
                prediction = foreground.cpu().numpy().astype(np.uint8)
                classification["predicted_class"] = best_candidate
            else:
                classification["reason"] = "no_matching_region"
        return prediction, classification
    finally:
        predictor.text_batches.clear()
