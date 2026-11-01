import json
import shutil
import socket
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from src.evaluation import evaluate
from src.runtime import (
    accumulation_window,
    load_adaptation,
    lr_factor,
    restore_rng,
    rng_state,
    seed_everything,
)
from src.training import parse_args


def test_broken_native_dependency_fails_before_ddp_or_dataset_preparation(monkeypatch, tmp_path):
    from src import training

    error = OSError("_torchaudio.abi3.so: undefined symbol: torch_library_impl")
    broken_transformers = ModuleType("transformers")

    def broken_import(name):
        if name in {"Sam3Config", "Sam3Model", "Sam3Processor"}:
            raise error
        raise AttributeError(name)

    broken_transformers.__getattr__ = broken_import
    monkeypatch.setitem(sys.modules, "transformers", broken_transformers)
    output = tmp_path / "must-not-be-created"
    monkeypatch.setattr(sys, "argv", ["train.py", "--output_dir", str(output)])
    monkeypatch.setattr(
        training, "setup", lambda _: pytest.fail("DDP started before checking dependencies")
    )
    with pytest.raises(RuntimeError, match="torch_library_impl") as caught:
        training.main()
    assert caught.value.__cause__ is error
    assert "pip install -r requirements.txt" in str(caught.value)
    assert not output.exists()


@pytest.mark.parametrize("token", [None, "hf_unit_test_not_a_real_token"])
def test_gated_model_fails_before_ddp_or_dataset_work(monkeypatch, tmp_path, token):
    import httpx
    import huggingface_hub
    from huggingface_hub import constants
    from huggingface_hub.errors import GatedRepoError

    from src import training

    calls = []
    response = httpx.Response(
        401,
        request=httpx.Request(
            "HEAD", "https://huggingface.co/facebook/sam3/resolve/main/config.json"
        ),
    )
    error = GatedRepoError("Access to model facebook/sam3 is restricted.", response=response)

    def denied(url, *, token):
        calls.append((url, token))
        raise error

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: token)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", denied)
    output = tmp_path / "must-not-be-created"
    monkeypatch.setattr(sys, "argv", ["train.py", "--output_dir", str(output)])
    monkeypatch.setattr(
        training, "setup", lambda _: pytest.fail("DDP started before model authorization check")
    )
    with pytest.raises(RuntimeError, match="Hugging Face denied access") as caught:
        training.main()
    assert caught.value.__cause__ is error
    assert len(calls) == 1 and calls[0][1] == token
    assert "/facebook/sam3/resolve/main/config.json" in calls[0][0]
    if token is None:
        assert "No Hugging Face token was found" in str(caught.value)
    else:
        assert "A Hugging Face token was found" in str(caught.value)
        assert token not in str(caught.value)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["local_directory", "local_files_only", "offline"])
def test_local_or_offline_model_access_never_uses_network(monkeypatch, tmp_path, mode):
    import huggingface_hub
    from huggingface_hub import constants

    from src.sam3_adapter import check_model_access

    args = parse_args([])
    if mode == "local_directory":
        args.model_id = str(tmp_path)
    args.local_files_only = mode == "local_files_only"
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", mode == "offline")
    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        lambda *a, **kw: pytest.fail("Local/offline mode made a network request"),
    )
    check_model_access(args)


def test_partial_accumulation_and_lr():
    assert [accumulation_window(i, 5, 3) for i in range(5)] == [
        (3, False),
        (3, False),
        (3, True),
        (2, False),
        (2, True),
    ]
    assert accumulation_window(0, 1, 8) == (1, True)
    assert lr_factor(1, 1, 0) == 1
    assert lr_factor(1, 10, 2) == 0.5
    assert lr_factor(2, 10, 2) == 1
    assert lr_factor(10, 10, 2) == 0


def test_cli_aliases():
    args = parse_args(
        [
            "--gradient_accumulation_steps",
            "4",
            "--num_workers",
            "0",
            "--gradient_clip",
            "1",
            "--no-gradient_checkpointing",
        ]
    )
    assert (args.grad_accum, args.workers, args.clip_norm, args.gradient_checkpointing) == (
        4,
        0,
        1,
        False,
    )
    with pytest.raises(SystemExit):
        parse_args(["--batch_size", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--eval_prompt_batch_size", "0"])


def test_rng_and_loader_generator_roundtrip():
    import random

    seed_everything(77)
    generator = torch.Generator().manual_seed(98)
    state = rng_state(generator)
    values = random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator)
    restore_rng(state, generator)
    assert random.random() == values[0]
    assert np.random.rand() == values[1]
    torch.testing.assert_close(torch.rand(2), values[2])
    torch.testing.assert_close(torch.rand(2, generator=generator), values[3])


def test_checkpoint_parameter_mismatch_fails(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "wrong.pt"
    torch.save({"trainable": {"wrong": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="parameter set"):
        load_adaptation(model, path)


def test_resume_can_opt_in_to_validation_schedule_change_only(tmp_path):
    from src.training import checkpoint_payload

    model = torch.nn.Linear(2, 1)
    model._plantseg_identity = {"base": "same"}
    args = parse_args([])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = checkpoint_payload(model, optimizer, scaler, args, 3, 100, 0.4, 2, [rng_state()], {})
    path = tmp_path / "last.pt"
    torch.save(state, path)
    args.val_every = 3
    with pytest.raises(ValueError, match="Resume mismatch for val_every"):
        load_adaptation(model, path, args)
    args.resume_allow_val_schedule_change = True
    args.eval_prompt_batch_size = 4
    restored = load_adaptation(model, path, args)
    assert (restored["epoch"], restored["step"], restored["best_metric"]) == (3, 100, 0.4)
    for key, value in (("eval_mode", "all_classes"), ("limit_val", 10), ("batch_size", 2)):
        previous = getattr(args, key)
        setattr(args, key, value)
        with pytest.raises(ValueError, match=f"Resume mismatch for {key}"):
            load_adaptation(model, path, args)
        setattr(args, key, previous)


def test_external_predictions_end_to_end(tiny_dataset, tmp_path):
    predictions = tmp_path / "predictions"
    shutil.copytree(tiny_dataset / "annotations/test", predictions)
    args = Namespace(
        dataset_dir=tiny_dataset,
        eval_mode="external",
        score_threshold=0.5,
        mask_threshold=0.5,
        label_policy="mask",
        prediction_dir=predictions,
        limit_test=None,
    )
    result = evaluate(None, args, "test", torch.device("cpu"), None, tmp_path / "results/test")
    assert result["miou"] == result["macc"] == result["foreground_dice"] == 1
    assert result["num_images"] == 3
    assert json.loads((tmp_path / "results/test.json").read_text())["num_images"] == 3
    assert (tmp_path / "results/test-per-class.csv").is_file()
    assert (tmp_path / "results/test-per-image.csv").is_file()
    assert np.load(tmp_path / "results/test-confusion.npy").sum() == 3 * 8 * 12


def test_two_process_ddp_and_evaluation(tiny_dataset, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("ddp_worker.py")),
            str(tiny_dataset),
            str(tmp_path / "distributed"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = json.loads((tmp_path / "distributed/test.json").read_text())
    assert result["num_images"] == 3  # 3 is indivisible by world size: no padding/duplication
    assert result["miou"] == result["macc"] == 1


def test_checkpoint_roundtrip_and_base_identity_guard(tmp_path):
    from src.training import checkpoint_payload, save_checkpoint

    model = torch.nn.Linear(2, 1)
    model._plantseg_identity = {"base_sha256": "original"}
    args = parse_args([])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    saved_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = checkpoint_payload(model, optimizer, scaler, args, 2, 7, 0.5, 1, [rng_state()], {})
    path = tmp_path / "last.pt"
    save_checkpoint(path, state)
    with torch.no_grad():
        model.weight.zero_()
    restored = load_adaptation(model, path, args)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved_weights[name])
    assert (restored["epoch"], restored["step"], restored["bad_epochs"]) == (2, 7, 1)
    optimizer.load_state_dict(restored["optimizer"])
    assert len(optimizer.state) == 2
    model._plantseg_identity["base_sha256"] = "different"
    with pytest.raises(ValueError, match="Base checkpoint"):
        load_adaptation(model, path)
