import random
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import categories_from_metadata, load_metadata
from src.evaluation import Sam3SemanticPredictor
from src.losses import PlantSegCriterion
from src.sam3_adapter import cache_text_features, configure_trainable, forward_model, prepare_inputs
from src.sam3_data import PlantSegCollator, PlantSegDataset, choose_categories
from src.training import TrainingLoss, make_train_loader, parse_args, prepare_model


def model_args(tiny_dataset, tiny_model_dir, tmp_path, scope="heads"):
    return parse_args(
        [
            "--dataset_dir",
            str(tiny_dataset),
            "--model_id",
            str(tiny_model_dir),
            "--local_files_only",
            "--output_dir",
            str(tmp_path / "output"),
            "--scope",
            scope,
            "--amp",
            "off",
            "--frozen_dtype",
            "fp32",
            "--num_workers",
            "0",
            "--loss_mask_size",
            "8",
            "--pos_prob",
            "1",
        ]
    )


def test_real_safetensors_load_train_and_infer_without_original_sam3(
    tiny_dataset, tiny_model_dir, tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "sam3", None)
    args = model_args(tiny_dataset, tiny_model_dir, tmp_path)
    model, frozen = prepare_model(args, torch.device("cpu"))
    assert model._plantseg_identity["backend"] == "transformers"
    assert model.text_encoder is None
    assert all(not p.requires_grad for p in model.vision_encoder.backbone.parameters())
    assert any(p.requires_grad for p in model.vision_encoder.neck.parameters())
    dataset = PlantSegDataset(args)
    loader, _ = make_train_loader(dataset, args, torch.Generator().manual_seed(1), model)
    batch = next(iter(loader))
    criterion = PlantSegCriterion()
    model.train()
    for module in frozen:
        module.eval()
    losses = TrainingLoss(model, criterion)(batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.text_projection.weight.grad is not None
    assert model.mask_decoder.instance_projection.weight.grad is not None
    assert model.vision_encoder.backbone.layers[0].attention.q_proj.weight.grad is None
    model.eval()
    predictor = Sam3SemanticPredictor(model, torch.device("cpu"), None, 0.0, 0.5)
    categories = categories_from_metadata(load_metadata(tiny_dataset))
    prediction = predictor(dataset[0]["image"], categories)
    assert prediction.shape == (8, 12)
    assert set(np.unique(prediction)) <= {0, 1, 2}
    assert not list(tiny_model_dir.glob("*.pt"))
    assert (tiny_model_dir / "model.safetensors").is_file()


def test_model_load_passes_explicit_token_to_all_loaders(
    tiny_dataset, tiny_model_dir, tmp_path, monkeypatch
):
    from huggingface_hub import constants
    from transformers import Sam3Config, Sam3Model, Sam3Processor

    token = "hf_unit_test_not_a_real_token"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    called = []

    def wrap(loader, name):
        def authenticated(*args, **kwargs):
            assert kwargs.get("token") == token
            called.append(name)
            return loader(*args, **kwargs)

        return authenticated

    for cls in (Sam3Config, Sam3Model, Sam3Processor):
        monkeypatch.setattr(cls, "from_pretrained", wrap(cls.from_pretrained, cls.__name__))
    args = model_args(tiny_dataset, tiny_model_dir, tmp_path)
    model, _ = prepare_model(args, torch.device("cpu"))
    assert set(called) == {"Sam3Config", "Sam3Model", "Sam3Processor"}
    assert token not in str(model._plantseg_identity)
    assert token not in str(vars(args))


def test_cached_text_matches_uncached_and_projection_receives_grad(
    tiny_dataset, tiny_model_dir, tmp_path
):
    args = model_args(tiny_dataset, tiny_model_dir, tmp_path, "full")
    args.gradient_checkpointing = False
    model, _ = prepare_model(args, torch.device("cpu"))
    model.eval()
    categories = categories_from_metadata(load_metadata(tiny_dataset))
    inputs = prepare_inputs(
        model._plantseg_processor,
        images=PlantSegDataset(args)[0]["image"],
        text=[categories[0]["name"]],
        max_text_length=8,
    )
    with torch.no_grad():
        before = model(**inputs).pred_masks
    configure_trainable(model, "heads")
    cache_text_features(model, categories)
    after = forward_model(model, inputs, torch.tensor([0])).pred_masks
    torch.testing.assert_close(before, after)
    after.sum().backward()
    assert model.text_projection.weight.grad is not None


@pytest.mark.parametrize("image_size", [16, 48])
@pytest.mark.parametrize("scope", ["heads", "full"])
def test_resized_safetensors_forward_backward_and_native_evaluation(
    tiny_dataset, tiny_model_dir, tmp_path, image_size, scope
):
    from transformers import Sam3Config, Sam3Processor

    args = model_args(tiny_dataset, tiny_model_dir, tmp_path, scope)
    args.image_size = image_size
    model, frozen = prepare_model(args, torch.device("cpu"))
    assert model.config.image_size == image_size
    assert model.config.vision_config.backbone_config.pretrain_image_size == 32
    # Learned positional tensors retain their original checkpoint dimensions.
    assert model.vision_encoder.backbone.embeddings.position_embeddings.shape == (1, 16, 32)
    grid = image_size // 8
    assert model.config.vision_config.backbone_feature_sizes == [
        [4 * grid, 4 * grid],
        [2 * grid, 2 * grid],
        [grid, grid],
    ]
    layer = model.vision_encoder.backbone.layers[0]
    assert layer.position_ids.shape == (grid * grid, 2)
    dataset = PlantSegDataset(args)
    loader, _ = make_train_loader(dataset, args, torch.Generator().manual_seed(1), model)
    batch = next(iter(loader))
    assert batch["inputs"]["pixel_values"].shape == (1, 3, image_size, image_size)
    model.train()
    for module in frozen:
        module.eval()
    loss = TrainingLoss(model, PlantSegCriterion())(batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.mask_decoder.instance_projection.weight.grad).all()
    if scope == "full":
        assert torch.isfinite(layer.attention.q_proj.weight.grad).all()
    model.eval()
    categories = categories_from_metadata(load_metadata(tiny_dataset))
    prediction = Sam3SemanticPredictor(model, torch.device("cpu"), None, 0.0, 0.5, 2)(
        dataset[0]["image"], categories
    )
    assert prediction.shape == (8, 12)
    saved = tmp_path / "resized-processor"
    model._plantseg_processor.save_pretrained(saved)
    processor = Sam3Processor.from_pretrained(saved, local_files_only=True)
    assert processor.image_processor.size == {"height": image_size, "width": image_size}
    assert processor.target_size == image_size
    # Source files are not rewritten when overriding resolution.
    assert Sam3Config.from_pretrained(tiny_model_dir).image_size == 32


def test_resolution_checkpoint_guard_and_weights_only_initialization(
    tiny_dataset, tiny_model_dir, tmp_path
):
    from src.runtime import load_adaptation, rng_state
    from src.training import checkpoint_payload, save_checkpoint

    args = model_args(tiny_dataset, tiny_model_dir, tmp_path)
    model, _ = prepare_model(args, torch.device("cpu"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = checkpoint_payload(model, optimizer, scaler, args, 0, 1, 0.5, 0, [rng_state()], {})
    path = tmp_path / "last.pt"
    save_checkpoint(path, state)
    args.image_size = 16
    resized, _ = prepare_model(args, torch.device("cpu"))
    for expected_args in (None, args):
        with pytest.raises(ValueError, match="Checkpoint input resolution"):
            load_adaptation(resized, path, expected_args)
    load_adaptation(resized, path, allow_resolution_change=True)
    torch.testing.assert_close(
        resized.mask_decoder.instance_projection.weight,
        model.mask_decoder.instance_projection.weight,
    )
    # Legacy checkpoints without a size field used the original model resolution.
    state.pop("input_image_size")
    save_checkpoint(path, state)
    with pytest.raises(ValueError, match="Checkpoint input resolution"):
        load_adaptation(resized, path)
    load_adaptation(model, path)


def test_672_input_with_production_patch_window_and_global_attention(
    tiny_dataset, tiny_model_dir, tmp_path
):
    from transformers import Sam3Config, Sam3Model, Sam3Processor

    # Retain small hidden dimensions, but use SAM3's real spatial geometry.
    config = Sam3Config.from_pretrained(tiny_model_dir)
    backbone = config.vision_config.backbone_config
    backbone.image_size = 1008
    backbone.patch_size = 14
    backbone.pretrain_image_size = 336
    backbone.window_size = 24
    backbone.num_hidden_layers = 2
    backbone.global_attn_indexes = [1]
    config.vision_config.backbone_feature_sizes = [[288, 288], [144, 144], [72, 72]]
    source = tmp_path / "sam3-spatial"
    Sam3Model(config).save_pretrained(source, safe_serialization=True)
    processor = Sam3Processor.from_pretrained(tiny_model_dir)
    processor.image_processor.size = {"height": 1008, "width": 1008}
    processor.save_pretrained(source)
    args = model_args(tiny_dataset, source, tmp_path, "full")
    args.image_size = 672
    model, _ = prepare_model(args, torch.device("cpu"))
    assert model.config.vision_config.backbone_feature_sizes == [[192, 192], [96, 96], [48, 48]]
    assert model.vision_encoder.backbone.layers[0].position_ids.shape == (24 * 24, 2)
    assert model.vision_encoder.backbone.layers[1].position_ids.shape == (48 * 48, 2)
    loader, _ = make_train_loader(
        PlantSegDataset(args), args, torch.Generator().manual_seed(1), model
    )
    batch = next(iter(loader))
    assert batch["inputs"]["pixel_values"].shape == (1, 3, 672, 672)
    model.train()
    outputs = forward_model(model, batch["inputs"], batch["prompt_ids"])
    assert outputs.pred_masks.shape[-2:] == (192, 192)
    loss = PlantSegCriterion()(outputs, batch["targets"])["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.vision_encoder.backbone.layers[1].attention.q_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    from src.evaluation import evaluate

    args.eval_mode = "all_classes"
    args.eval_prompt_batch_size = 2
    result = evaluate(model, args, "val", torch.device("cpu"), None, tmp_path / "validation")
    assert result["num_images"] == 3
    assert result["input_image_size"] == [672, 672]
    assert np.isfinite(result["miou"])


@pytest.mark.parametrize("scope", ["heads", "full"])
def test_batched_evaluation_matches_original_forward_and_reuses_text(
    tiny_dataset, tiny_model_dir, tmp_path, scope
):
    from src.evaluation import class_evidence

    args = model_args(tiny_dataset, tiny_model_dir, tmp_path, scope)
    model, _ = prepare_model(args, torch.device("cpu"))
    model.eval()
    image = PlantSegDataset(args)[0]["image"]
    categories = categories_from_metadata(load_metadata(tiny_dataset))
    expected_outputs = []
    expected = torch.zeros((image.height, image.width), dtype=torch.int64)
    best = torch.zeros_like(expected, dtype=torch.float32)
    with torch.inference_mode():
        for category in categories:
            inputs = prepare_inputs(
                model._plantseg_processor,
                images=image,
                text=[category["name"]],
                max_text_length=8,
            )
            output = forward_model(model, inputs, torch.tensor([category["id"]]))
            expected_outputs.append(output)
            scores = (output.pred_logits.sigmoid() * output.presence_logits.sigmoid())[0]
            evidence = class_evidence(output.pred_masks[0], scores, best.shape)
            expected[evidence > best] = category["id"] + 1
            best = torch.maximum(best, evidence)
    calls = []
    vision_calls = []
    projection_calls = []
    handles = [
        model.register_forward_hook(lambda m, a, o: calls.append(o)),
        model.vision_encoder.register_forward_hook(lambda m, a, o: vision_calls.append(1)),
        model.text_projection.register_forward_hook(lambda m, a, o: projection_calls.append(1)),
    ]
    # Same text forces exact class ties; input order must survive chunk boundaries.
    tied = [{**c, "name": categories[0]["name"]} for c in categories]
    try:
        for batch_size in (1, 2, 3):
            calls.clear()
            vision_calls.clear()
            projection_calls.clear()
            predictor = Sam3SemanticPredictor(
                model, torch.device("cpu"), None, 0.0, 0.5, batch_size
            )
            actual = predictor(image, categories)
            np.testing.assert_array_equal(actual, expected.numpy())
            torch.testing.assert_close(
                torch.cat([o.pred_masks for o in calls]),
                torch.cat([o.pred_masks for o in expected_outputs]),
                rtol=1e-4,
                atol=1e-5,
            )
            torch.testing.assert_close(
                torch.cat([o.pred_logits for o in calls]),
                torch.cat([o.pred_logits for o in expected_outputs]),
                rtol=1e-4,
                atol=1e-5,
            )
            count = len(projection_calls)
            predictor(image, categories)
            assert len(projection_calls) == count  # no tokenization/CLIP/projection per image
            assert len(vision_calls) == 2  # backbone once per image, independent of prompt count
        if scope == "heads":
            model._plantseg_text_hidden[1].copy_(model._plantseg_text_hidden[0])
            model._plantseg_text_mask[1].copy_(model._plantseg_text_mask[0])
        for batch_size in (1, 2):
            predictor = Sam3SemanticPredictor(
                model, torch.device("cpu"), None, 0.0, 0.0, batch_size
            )
            prediction = predictor(image, tied)
            assert set(np.unique(prediction)) == {1}
        predictor = Sam3SemanticPredictor(model, torch.device("cpu"), None, 1.0, 0.5, 2)
        assert not predictor(image, categories).any()
    finally:
        for handle in handles:
            handle.remove()


def test_processor_batches_positive_and_negative_targets(tiny_dataset, tiny_model_dir, tmp_path):
    args = model_args(tiny_dataset, tiny_model_dir, tmp_path)
    model, _ = prepare_model(args, torch.device("cpu"))
    dataset = PlantSegDataset(args)
    collator = PlantSegCollator(model._plantseg_processor, dataset.categories, 2, 1, 8)
    batch = collator([dataset[0]])
    assert batch["inputs"]["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["targets"][0]["boxes"].shape == (2, 4)
    assert batch["targets"][1]["boxes"].shape == (0, 4)
    assert not batch["targets"][1]["semantic_mask"].any()
    losses = TrainingLoss(model, PlantSegCriterion())(batch)
    assert all(torch.isfinite(value) for value in losses.values())


def test_prompt_sampling_excludes_true_class_from_negatives():
    categories = [
        {"id": 0, "supercategory": "apple"},
        {"id": 1, "supercategory": "apple"},
        {"id": 2, "supercategory": "bean"},
    ]
    assert choose_categories(categories, 0, 1, 0, random.Random(7)) == [1]
    assert set(choose_categories(categories, 0, 3, 1, random.Random(7))) == {0, 1, 2}


def test_hungarian_loss_is_invariant_to_target_order_and_handles_negative():
    torch.manual_seed(8)
    raw = torch.randn(1, 3, 4, 4, requires_grad=True)
    logits = torch.randn(1, 3, requires_grad=True)
    boxes = torch.tensor(
        [[[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9], [0.1, 0.2, 0.8, 0.8]]], requires_grad=True
    )
    output = SimpleNamespace(
        pred_masks=raw,
        pred_logits=logits,
        pred_boxes=boxes,
        presence_logits=torch.zeros(1, 1, requires_grad=True),
        semantic_seg=torch.randn(1, 1, 4, 4, requires_grad=True),
    )
    target = {
        "boxes": boxes.detach()[0, :2],
        "masks": (torch.rand(2, 4, 4) > 0.5).float(),
        "semantic_mask": torch.ones(1, 4, 4),
    }
    criterion = PlantSegCriterion()
    loss = criterion(output, [target])["loss"]
    permuted = {
        "boxes": target["boxes"].flip(0),
        "masks": target["masks"].flip(0),
        "semantic_mask": target["semantic_mask"],
    }
    torch.testing.assert_close(loss, criterion(output, [permuted])["loss"])
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    negative = {
        "boxes": target["boxes"][:0],
        "masks": target["masks"][:0],
        "semantic_mask": torch.zeros(1, 4, 4),
    }
    assert torch.isfinite(criterion(output, [negative])["loss"])


def test_real_transformers_ddp_checkpointing(tiny_dataset, tiny_model_dir, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            str(Path(__file__).with_name("hf_ddp_worker.py")),
            str(tiny_dataset),
            str(tiny_model_dir),
            str(tmp_path / "ddp"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_transformers_adaptation_roundtrip(tiny_dataset, tiny_model_dir, tmp_path):
    from src.runtime import load_adaptation, rng_state
    from src.sam3_adapter import build_optimizer
    from src.training import checkpoint_payload, save_checkpoint

    args = model_args(tiny_dataset, tiny_model_dir, tmp_path)
    model, _ = prepare_model(args, torch.device("cpu"))
    loader, _ = make_train_loader(
        PlantSegDataset(args), args, torch.Generator().manual_seed(1), model
    )
    batch = next(iter(loader))
    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    TrainingLoss(model, PlantSegCriterion())(batch)["loss"].backward()
    optimizer.step()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, checkpoint_payload(model, optimizer, scaler, args, 0, 1, 0.5, 0, [rng_state()], {})
    )
    restored, _ = prepare_model(args, torch.device("cpu"))
    state = load_adaptation(restored, path, args)
    assert state["format_version"] == 3
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = forward_model(model, batch["inputs"], batch["prompt_ids"]).pred_masks
        actual = forward_model(restored, batch["inputs"], batch["prompt_ids"]).pred_masks
    torch.testing.assert_close(expected, actual)


def test_full_scope_backward_with_checkpointing(tiny_dataset, tiny_model_dir, tmp_path):
    args = model_args(tiny_dataset, tiny_model_dir, tmp_path, "full")
    model, frozen = prepare_model(args, torch.device("cpu"))
    assert frozen == []
    loader, _ = make_train_loader(
        PlantSegDataset(args), args, torch.Generator().manual_seed(1), model
    )
    TrainingLoss(model, PlantSegCriterion())(next(iter(loader)))["loss"].backward()
    gradient = model.text_encoder.text_model.embeddings.token_embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert model.vision_encoder.backbone.layers[0].attention.q_proj.weight.grad is not None


def test_processor_collator_spawn_worker(tiny_dataset, tiny_model_dir, tmp_path):
    args = model_args(tiny_dataset, tiny_model_dir, tmp_path)
    args.workers = 1
    model, _ = prepare_model(args, torch.device("cpu"))
    loader, _ = make_train_loader(
        PlantSegDataset(args), args, torch.Generator().manual_seed(1), model
    )
    batches = list(loader)
    assert len(batches) == 3
    assert all(batch["inputs"]["pixel_values"].shape == (1, 3, 32, 32) for batch in batches)
