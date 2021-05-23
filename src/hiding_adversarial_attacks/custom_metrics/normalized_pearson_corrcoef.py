import torch
from torchmetrics import Metric
from torchmetrics.functional.regression.pearson import pearson_corrcoef

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
        r = custom_pearson_corrcoef(preds, target)
        normalized_r = (1 + r) / 2
        self.pcc.append(torch.mean(r))
        self.normalized_pcc.append(torch.mean(normalized_r))

    def compute(self):
        return torch.mean(torch.tensor(self.normalized_pcc, device=self.device))


def normalized_batched_pearson_corrcoef(
    preds: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    assert preds.shape == target.shape
    if preds.ndim > 1 or target.ndim > 1:
        r = torch.tensor(
            [
                pearson_corrcoef(preds_single.view(-1), target_single.view(-1))
                for preds_single, target_single in zip(preds, target)
            ],
            device=preds.device,
        )
    else:
        r = pearson_corrcoef(preds, target)
    # Normalize to range [0,1]
    normalized_r = (1 + r) / 2
    return normalized_r
