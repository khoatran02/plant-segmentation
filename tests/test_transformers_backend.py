from pathlib import Path
import random
from types import SimpleNamespace
import socket
import subprocess
import sys

import numpy as np
import torch

from src.data import categories_from_metadata, load_metadata
from src.evaluation import Sam3SemanticPredictor
from src.losses import PlantSegCriterion
from src.sam3_adapter import configure_trainable, cache_text_features, forward_model, prepare_inputs
from src.sam3_data import PlantSegDataset, PlantSegCollator, choose_categories
from src.training import TrainingLoss, parse_args, prepare_model, make_train_loader


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
    from src.training import checkpoint_payload, save_checkpoint
    from src.sam3_adapter import build_optimizer

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
