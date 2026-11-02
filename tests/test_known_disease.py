import json

import numpy as np
import pytest
import torch

from src.data import categories_from_metadata, load_metadata
from src.evaluation import Sam3SemanticPredictor
from src.losses import KNOWN_DISEASE_LOSS_VERSION, KnownDiseaseCriterion
from src.runtime import load_adaptation, rng_state
from src.sam3_adapter import forward_model
from src.sam3_data import PlantSegDataset
from src.training import (
    checkpoint_payload,
    make_train_loader,
    parse_args,
    prepare_model,
    save_checkpoint,
    train,
)


def known_args(tiny_dataset, tiny_model_dir, tmp_path):
    return parse_args(
        [
            "--known_disease",
            "--dataset_dir",
            str(tiny_dataset),
            "--model_id",
            str(tiny_model_dir),
            "--local_files_only",
            "--output_dir",
            str(tmp_path / "known"),
            "--amp",
            "off",
            "--frozen_dtype",
            "fp32",
            "--num_workers",
            "0",
            "--loss_mask_size",
            "8",
        ]
    )


@pytest.mark.parametrize(
    "conflict",
    [
        ["--pos_prob", "0.7"],
        ["--prompts_per_image", "2"],
        ["--eval_mode", "all_classes"],
        ["--eval_mask_source", "instance"],
        ["--loss_mode", "combined"],
    ],
)
def test_known_disease_protocol_rejects_conflicting_settings(conflict):
    args = parse_args(["--known_disease"])
    assert (args.pos_prob, args.prompts_per_image) == (1.0, 1)
    assert (args.eval_mode, args.eval_mask_source, args.loss_mode) == (
        "oracle",
        "semantic",
        "semantic",
    )
    with pytest.raises(SystemExit):
        parse_args(["--known_disease", *conflict])


def test_known_disease_loss_trains_masks_without_confidence_gradients(
    tiny_dataset, tiny_model_dir, tmp_path
):
    args = known_args(tiny_dataset, tiny_model_dir, tmp_path)
    model, _ = prepare_model(args, torch.device("cpu"))
    dataset = PlantSegDataset(args)
    loader, _ = make_train_loader(dataset, args, torch.Generator().manual_seed(1), model)
    batch = next(iter(loader))
    assert len(batch["targets"]) == args.batch_size
    assert all(target["semantic_mask"].any() for target in batch["targets"])
    output = forward_model(model, batch["inputs"], batch["prompt_ids"])
    assert output.pred_masks is None
    output.pred_logits.retain_grad()
    output.presence_logits.retain_grad()
    losses = KnownDiseaseCriterion()(output, batch["targets"])
    assert set(losses) == {"loss", "semantic_bce", "semantic_dice"}
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.mask_decoder.semantic_projection.weight.grad.abs().sum() > 0
    assert model.text_projection.weight.grad.abs().sum() > 0
    assert output.pred_logits.grad is None and output.presence_logits.grad is None


def test_known_disease_mask_ignores_confidence_but_keeps_spatial_threshold(
    tiny_dataset, tiny_model_dir, tmp_path
):
    args = known_args(tiny_dataset, tiny_model_dir, tmp_path)
    model, _ = prepare_model(args, torch.device("cpu"))
    model.eval()
    image = PlantSegDataset(args)[0]["image"]
    categories = categories_from_metadata(load_metadata(tiny_dataset))

    def low_confidence(module, inputs, output):
        output.pred_logits.fill_(-100)
        output.presence_logits.fill_(-100)
        output.semantic_seg.fill_(-10)
        output.semantic_seg[..., :4, :4] = 10

    handle = model.register_forward_hook(low_confidence)
    try:
        gated = Sam3SemanticPredictor(
            model, torch.device("cpu"), None, 0.5, 0.5, mask_source="semantic"
        )
        known = Sam3SemanticPredictor(
            model,
            torch.device("cpu"),
            None,
            1.0,
            0.5,
            mask_source="semantic",
            known_disease=True,
        )
        assert not gated(image, categories[:1]).any()
        prediction = known(image, categories[:1])
        assert prediction.any() and not prediction.all()
        assert set(np.unique(prediction)) == {0, 1}
        known.mask_threshold = 1.0
        assert not known(image, categories[:1]).any()
        with pytest.raises(ValueError, match="exactly one"):
            known(image, categories)
    finally:
        handle.remove()


def test_known_disease_run_initializes_old_checkpoint_and_saves_matching_validation(
    tiny_dataset, tiny_model_dir, tmp_path, monkeypatch
):
    args = known_args(tiny_dataset, tiny_model_dir, tmp_path)
    args.known_disease = False
    args.loss_mode = "combined"
    original, _ = prepare_model(args, torch.device("cpu"))
    optimizer = torch.optim.AdamW([p for p in original.parameters() if p.requires_grad])
    old_checkpoint = tmp_path / "old.pt"
    save_checkpoint(
        old_checkpoint,
        checkpoint_payload(
            original,
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            args,
            40,
            861,
            0.06,
            0,
            [rng_state()],
            {},
        ),
    )
    args = known_args(tiny_dataset, tiny_model_dir, tmp_path)
    args.init_from = old_checkpoint
    args.max_updates = 2
    args.epochs = 3
    args.grad_accum = 1
    args.val_every = 20
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU test")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    train(args, torch.device("cpu"))
    record = json.loads((args.output_dir / "log.jsonl").read_text().splitlines()[-1])
    assert record["step"] == record["optimizer_updates"] == 2
    assert record["val_known_disease"] is True
    assert record["val_confidence_filtering"] is False
    assert record["val_eval_mode"] == "oracle"
    assert record["val_eval_mask_source"] == "semantic"
    assert record["val_score_threshold"] is None
    assert set(record["train_loss_components"]) == set(KnownDiseaseCriterion.weights)
    restored, _ = prepare_model(args, torch.device("cpu"))
    saved = load_adaptation(restored, args.output_dir / "best.pt", args)
    assert saved["model_identity"]["loss_version"] == KNOWN_DISEASE_LOSS_VERSION
    weight = "mask_decoder.semantic_projection.weight"
    assert not torch.equal(saved["trainable"][weight], original.state_dict()[weight])
    saved["config"]["known_disease"] = False
    inconsistent = tmp_path / "inconsistent.pt"
    torch.save(saved, inconsistent)
    with pytest.raises(ValueError, match="known_disease"):
        load_adaptation(restored, inconsistent, args)
