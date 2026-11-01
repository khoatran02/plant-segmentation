"""SAM3 through Hugging Face Transformers: model IDs, processors and safetensors.

No upstream sam3 checkout or manually supplied sam3.pt is used by this backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.losses import LOSS_VERSION, PlantSegCriterion
from src.runtime import file_sha256


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model_id", default="facebook/sam3", help="HF model ID or local save_pretrained directory"
    )
    parser.add_argument(
        "--model_revision", default="main", help="HF commit/tag; resolved commit is recorded"
    )
    parser.add_argument(
        "--cache_dir", type=Path, help="Hugging Face download cache (e.g. /kaggle/working/hf-cache)"
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use cached/local model files without network access",
    )
    parser.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument(
        "--image_size",
        type=int,
        help="Square input resolution (e.g. 672); omit to use the pretrained model size",
    )


def configure_input_resolution(config, processor, image_size):
    """Set resolution before constructing ViT's nonpersistent global RoPE grids.

    Keep pretrain_image_size and learned positional weights unchanged so the
    original safetensors load exactly. The embedding layer tiles these weights.
    """
    if image_size is None:
        return
    patch = config.vision_config.backbone_config.patch_size
    if not isinstance(patch, int):
        if len(set(patch)) != 1:
            raise ValueError("--image_size requires square ViT patches")
        patch = patch[0]
    if image_size <= 0 or image_size % (2 * patch):
        raise ValueError(f"--image_size must be a positive multiple of {2 * patch}")
    grid = image_size // patch
    config.image_size = image_size
    sizes = [int(grid * scale) for scale in config.vision_config.scale_factors[:-1]]
    config.vision_config.backbone_feature_sizes = [[size, size] for size in sizes]
    processor.image_processor.size = {"height": image_size, "width": image_size}
    processor.image_processor.mask_size = {"height": sizes[0], "width": sizes[0]}
    processor.target_size = image_size


def prepare_inputs(processor, images=None, text=None, max_text_length=32):
    """Separate image/text kwargs; SAM3 processor fixes text length to 32 internally."""
    inputs = {}
    if images is not None:
        inputs["pixel_values"] = processor(images=images, return_tensors="pt")["pixel_values"]
    if text is not None:
        encoded = processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        inputs.update({key: encoded[key] for key in ("input_ids", "attention_mask")})
    return inputs


def load_sam3_api():
    """Check native/optional dependencies before dataset preparation or DDP startup."""
    try:
        from transformers import Sam3Config, Sam3Model, Sam3Processor
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Could not import Transformers SAM3: {error}\n"
            "Install the matching torch/torchvision/torchaudio releases with "
            "'python -m pip install -r requirements.txt', then restart the notebook kernel. "
            "This import check does not download model weights."
        ) from error
    return Sam3Config, Sam3Model, Sam3Processor


def pretrained_options(args):
    from huggingface_hub import get_token

    return {
        "revision": args.model_revision,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "local_files_only": args.local_files_only,
        # Explicit auth also works when HF_HUB_DISABLE_IMPLICIT_TOKEN is enabled.
        # Keep this transient: tokens never enter args, checkpoints or model identity.
        "token": get_token(),
    }


def check_model_access(args):
    """Check remote config access before DDP/dataset work, even if a config is cached."""
    from huggingface_hub import constants, get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if Path(args.model_id).is_dir() or args.local_files_only or constants.HF_HUB_OFFLINE:
        return
    options = pretrained_options(args)
    url = hf_hub_url(args.model_id, "config.json", revision=args.model_revision)
    try:
        get_hf_file_metadata(url, token=options["token"])
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if not isinstance(error, GatedRepoError) and status not in (401, 403):
            raise
        authentication = (
            "No Hugging Face token was found in this process. Set os.environ['HF_TOKEN'] "
            "in a Python notebook cell before launching torchrun; rerun that cell after "
            "a kernel restart."
            if options["token"] is None
            else "A Hugging Face token was found, but access was denied. Verify it with "
            "HfApi().whoami(), confirm that the same account has approved model access, "
            "and that the token has permission to read this gated model."
        )
        raise RuntimeError(
            f"Hugging Face denied access to '{args.model_id}' (HTTP {status or '401/403'}). "
            f"{authentication} Model access: https://huggingface.co/{args.model_id}. "
            "Dependency installation cannot grant model access."
        ) from error


def build_model(args):
    import transformers

    Sam3Config, Sam3Model, Sam3Processor = load_sam3_api()
    options = pretrained_options(args)
    try:
        config = Sam3Config.from_pretrained(args.model_id, **options)
        resolved = getattr(config, "_commit_hash", None)
        if resolved is not None:
            options["revision"] = resolved
        processor = Sam3Processor.from_pretrained(args.model_id, **options)
        native_size = config.image_size
        native_size = (
            [native_size, native_size] if isinstance(native_size, int) else list(native_size)
        )
        configure_input_resolution(config, processor, getattr(args, "image_size", None))
        if args.no_pretrained:
            config._attn_implementation = args.attn_implementation
            model = Sam3Model(config)
        else:
            model = Sam3Model.from_pretrained(
                args.model_id,
                config=config,
                use_safetensors=True,
                dtype=torch.float32,
                attn_implementation=args.attn_implementation,
                **options,
            )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not load Transformers SAM3 '{args.model_id}': {error}\n"
            "For facebook/sam3, accept model access on Hugging Face, provide HF_TOKEN, and enable Internet. "
            "For offline use, pass --model_id /path/to/save_pretrained_directory --local_files_only."
        ) from error
    # Keep the CLI reference immutable; the actual commit lives in the model identity.
    local_path = Path(args.model_id)
    local_hashes = None
    if local_path.is_dir():
        local_hashes = {
            p.name: file_sha256(p)
            for p in sorted(local_path.iterdir())
            if p.is_file() and p.suffix in {".json", ".txt", ".safetensors"}
        }
        if not args.no_pretrained and not any(
            name.endswith(".safetensors") for name in local_hashes
        ):
            raise ValueError(
                "Local model directory must contain save_pretrained safetensors weights"
            )
    model._plantseg_identity = {
        "backend": "transformers",
        "model_id": args.model_id if not local_path.is_dir() else "local-safetensors",
        "resolved_revision": resolved if resolved is not None else args.model_revision,
        "local_files_sha256": local_hashes,
        "transformers_version": transformers.__version__,
        "loss_version": LOSS_VERSION,
        "scratch_seed": args.seed if args.no_pretrained else None,
        "metadata_sha256": file_sha256(args.dataset_dir / "Metadata.csv"),
    }
    model._plantseg_processor = processor
    model._plantseg_native_image_size = native_size
    size = model.config.image_size
    model._plantseg_image_size = [size, size] if isinstance(size, int) else list(size)
    return model


def configure_trainable(model, scope):
    frozen = [model.vision_encoder.backbone, model.text_encoder] if scope == "heads" else []
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for module in frozen:
        module.requires_grad_(False)
    return frozen


@torch.no_grad()
def cache_text_features(model, categories):
    """Cache frozen CLIP hidden states; the SAM3 text projection remains trainable."""
    processor = model._plantseg_processor
    model.text_encoder.eval()
    encoded = prepare_inputs(
        processor,
        text=[c["name"] for c in categories],
        max_text_length=model.config.text_config.max_position_embeddings,
    )
    hidden = model.text_encoder(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], return_dict=True
    ).last_hidden_state
    model.register_buffer("_plantseg_text_hidden", hidden.detach(), persistent=False)
    model.register_buffer(
        "_plantseg_text_mask", encoded["attention_mask"].detach(), persistent=False
    )
    # Avoid placing a frozen text encoder on each 16 GB GPU.
    model.text_encoder = None
    return [model.vision_encoder.backbone]


def forward_model(model, inputs, prompt_ids=None):
    if hasattr(model, "_plantseg_text_hidden"):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        if prompt_ids is None:
            raise ValueError("Cached text features require category prompt_ids")
        inputs = dict(inputs)
        inputs.pop("input_ids", None)
        hidden = model._plantseg_text_hidden[prompt_ids]
        inputs["text_embeds"] = BaseModelOutputWithPooling(
            pooler_output=model.text_projection(hidden)
        )
        inputs["attention_mask"] = model._plantseg_text_mask[prompt_ids]
    return model(**inputs)


def set_train_mode(model, frozen_modules):
    model.train()
    for module in frozen_modules:
        module.eval()


def build_optimizer(model, args):
    groups = {
        "vision": (args.lr_vision, [], []),
        "language": (args.lr_language, [], []),
        "decoder": (args.lr, [], []),
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = (
            "vision"
            if name.startswith("vision_encoder.")
            else "language"
            if name.startswith("text_encoder.")
            else "decoder"
        )
        decay, no_decay = groups[key][1:]
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    parameters = []
    for lr, decay, no_decay in groups.values():
        if decay:
            parameters.append({"params": decay, "weight_decay": args.weight_decay, "base_lr": lr})
        if no_decay:
            parameters.append({"params": no_decay, "weight_decay": 0.0, "base_lr": lr})
    return torch.optim.AdamW(parameters, lr=args.lr)


def build_criterion():
    return PlantSegCriterion()


def configure_activation_checkpointing(model, enabled):
    """Checkpoint SAM3 blocks even though HF's outer Sam3Model does not expose GC.

    Use PyTorch's nonreentrant implementation: parameters may require gradients even
    when frozen-encoder features do not. Dropout RNG and AMP state are preserved.
    """
    from functools import wraps

    from torch.utils.checkpoint import checkpoint

    blocks = [
        *model.vision_encoder.backbone.layers,
        *model.detr_encoder.layers,
        *model.detr_decoder.layers,
        model.vision_encoder.neck,
        model.mask_decoder,
    ]
    if model.text_encoder is not None:
        blocks.extend(model.text_encoder.text_model.encoder.layers)
    for module in blocks:
        if not hasattr(module, "_plantseg_original_forward"):
            module._plantseg_original_forward = module.forward
        original = module._plantseg_original_forward
        if not enabled:
            module.forward = original
            continue

        def wrap(block, forward):
            @wraps(forward)
            def checkpointed(*args, **kwargs):
                if block.training and torch.is_grad_enabled():
                    return checkpoint(forward, *args, use_reentrant=False, **kwargs)
                return forward(*args, **kwargs)

            return checkpointed

        module.forward = wrap(module, original)
