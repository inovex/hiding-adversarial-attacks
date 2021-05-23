import torch
from torchmetrics import Metric
from torchmetrics.functional import pearson_corrcoef
from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute


class BatchedPearsonCorrCoef(Metric):
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
        r = custom_pearson_corrcoef(preds, target)
        self.pcc = torch.cat((self.pcc, r), dim=0)

    def compute(self):
        return torch.mean(torch.tensor(self.pcc, device=self.device))


def custom_pearson_corrcoef(preds, target):
    if preds.ndim > 1 or target.ndim > 1:
        r = torch.tensor(
            [
                pearson_corrcoef(preds_single.view(-1), target_single.view(-1))
                for preds_single, target_single in zip(preds, target)
            ],
            device=preds.device,
        )
    else:
        r = _pearson_corrcoef_compute(preds, target)
    return r
