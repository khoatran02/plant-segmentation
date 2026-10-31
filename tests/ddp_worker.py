"""Real two-process Gloo smoke test; no CUDA/SAM3 weights required."""

from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.evaluation import evaluate
from src.runtime import accumulation_window, barrier, cleanup, rank, setup
from src.training import TrainingLoss


class FakeSAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.unused = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return {"prediction": self.linear(x)}


class Loss(torch.nn.Module):
    def forward(self, stages, targets):
        return {"loss": (stages["prediction"] - targets[0]).square().mean()}


def main():
    device = setup("cpu")
    try:
        torch.manual_seed(7)
        network = FakeSAM()
        reference = FakeSAM()
        reference.load_state_dict(network.state_dict())
        model = DDP(TrainingLoss(network, Loss()), find_unused_parameters=True)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for index in range(5):
            length, update = accumulation_window(index, 5, 3)
            batch = {
                "inputs": {"x": torch.tensor([[float(index + rank())]])},
                "targets": [torch.tensor([[1.0]])],
            }
            with nullcontext() if update else model.no_sync():
                (model(batch)["loss"] / length).backward()
            ref_x = torch.tensor([[float(index)], [float(index + 1)]])
            (reference(x=ref_x)["prediction"].sub(1).square().mean() / length).backward()
            if update:
                optimizer.step()
                ref_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ref_optimizer.zero_grad(set_to_none=True)
        for actual, expected in zip(network.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        root, output = Path(sys.argv[1]), Path(sys.argv[2])
        args = Namespace(
            dataset_dir=root,
            prediction_dir=root / "annotations/test",
            eval_mode="external",
            score_threshold=0.5,
            mask_threshold=0.5,
            label_policy="mask",
            limit_test=None,
        )
        evaluate(None, args, "test", device, None, output / "test")
        barrier()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
