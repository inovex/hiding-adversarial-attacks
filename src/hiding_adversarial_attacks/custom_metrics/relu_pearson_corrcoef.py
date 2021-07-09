import torch
from torch.nn.functional import relu
from torchmetrics import Metric

from hiding_adversarial_attacks.custom_metrics.pearson_corrcoef import (
    custom_pearson_corrcoef,
)


class ReluBatchedPearsonCorrCoef(Metric):
    def __init__(self, device: torch.device, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.device = device
        self.add_state(
            "pcc",
            default=torch.tensor([], device=self.device),
            dist_reduce_fx="mean",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        r = relu_pearson_corrcoef(preds, target)
        self.pcc = torch.cat((self.pcc, r), dim=0)

    def compute(self):
        return torch.mean(self.pcc)


def relu_pearson_corrcoef(
    preds: torch.Tensor, target: torch.Tensor, windowed: bool = False
) -> torch.Tensor:
    r = custom_pearson_corrcoef(preds, target, windowed)
    _r = relu(r)
    return _r
