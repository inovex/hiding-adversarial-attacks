import torch
from torchmetrics import Metric

from hiding_adversarial_attacks.custom_metrics.pearson_corrcoef import (
    custom_pearson_corrcoef,
)


class NormalizedBatchedPearsonCorrcoef(Metric):
    def __init__(self, device: torch.device, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.device = device
        self.add_state(
            "pcc",
            default=[],
            dist_reduce_fx="mean",
        )
        self.add_state(
            "normalized_pcc",
            default=[],
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
        r = normalized_batched_pearson_corrcoef
        self.pcc.append(torch.mean(r))
        self.normalized_pcc.append(torch.mean(r))

    def compute(self):
        return torch.mean(torch.tensor(self.normalized_pcc, device=self.device))


def normalized_batched_pearson_corrcoef(
    preds: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    r = custom_pearson_corrcoef(preds, target)
    # Normalize to range [0,1]
    normalized_r = (1 + r) / 2
    return normalized_r
