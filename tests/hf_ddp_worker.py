"""Train a real tiny Transformers SAM3 with DDP + nonreentrant checkpointing on CPU."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from src.losses import PlantSegCriterion
from src.runtime import setup, cleanup, gather_objects
from src.sam3_adapter import set_train_mode
from src.sam3_data import PlantSegDataset
from src.training import parse_args, prepare_model, TrainingLoss, make_train_loader


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
                "--pos_prob",
                "1",
            ]
        )
        # Avoid cache creation races in this isolated CPU test.
        from src.runtime import rank, barrier
        from src.data import build_coco_split

        if rank() == 0:
            build_coco_split(args.dataset_dir, "train", args.output_dir)
        barrier()
        model, frozen = prepare_model(args, device)
        dataset = PlantSegDataset(args)
        loader, _ = make_train_loader(dataset, args, torch.Generator().manual_seed(3), model)
        wrapper = DDP(
            TrainingLoss(model, PlantSegCriterion()),
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        set_train_mode(model, frozen)
        initial = model.mask_decoder.instance_projection.weight.detach().clone()
        wrapper(next(iter(loader)))["loss"].backward()
        optimizer.step()
        assert not torch.equal(initial, model.mask_decoder.instance_projection.weight)
        checksum = model.mask_decoder.instance_projection.weight.detach().sum().item()
        assert len(set(gather_objects(checksum))) == 1
    finally:
        cleanup()


if __name__ == "__main__":
    main()
