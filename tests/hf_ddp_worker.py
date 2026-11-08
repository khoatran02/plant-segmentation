"""Train a real tiny Transformers SAM3 with DDP + nonreentrant checkpointing on CPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.runtime import cleanup, gather_objects, setup
from src.sam3_adapter import build_criterion, set_train_mode
from src.sam3_data import PlantSegDataset
from src.training import TrainingLoss, make_train_loader, parse_args, prepare_model


def main():
    device = setup("cpu")
    try:
        args = parse_args(
            [
                "--dataset_dir",
                sys.argv[1],
                "--model_id",
                sys.argv[2],
                "--output_dir",
                sys.argv[3],
                "--local_files_only",
                "--amp",
                "off",
                "--frozen_dtype",
                "fp32",
                "--num_workers",
                "0",
                "--loss_mask_size",
                "8",
                "--image_size",
                "16",
                "--pos_prob",
                "1",
            ]
        )
        if len(sys.argv) > 4 and sys.argv[4] in ("semantic", "known_disease"):
            args.loss_mode = args.eval_mask_source = "semantic"
            args.known_disease = sys.argv[4] == "known_disease"
        # Avoid cache creation races in this isolated CPU test.
        from src.data import build_coco_split
        from src.runtime import barrier, rank

        if rank() == 0:
            build_coco_split(args.dataset_dir, "train", args.output_dir)
        barrier()
        model, frozen = prepare_model(args, device)
        dataset = PlantSegDataset(args)
        loader, _ = make_train_loader(dataset, args, torch.Generator().manual_seed(3), model)
        wrapper = DDP(
            TrainingLoss(model, build_criterion(args)),
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        set_train_mode(model, frozen)
        tracked = (
            model.mask_decoder.semantic_projection.weight
            if args.loss_mode == "semantic"
            else model.mask_decoder.instance_projection.weight
        )
        initial = tracked.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            wrapper(next(iter(loader)))["loss"].backward()
            optimizer.step()
        assert not torch.equal(initial, tracked)
        checksum = tracked.detach().sum().item()
        assert len(set(gather_objects(checksum))) == 1
    finally:
        cleanup()


if __name__ == "__main__":
    main()
