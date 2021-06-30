import torch
from torch.nn.functional import relu
from torchmetrics import Metric
from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute


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
        return torch.mean(torch.tensor(self.pcc, device=self.device))


def relu_pearson_corrcoef(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert preds.shape == target.shape
    if preds.ndim > 1 or target.ndim > 1:
        r = pearson_corrcoef_compute(preds, target)
    else:
        r = _pearson_corrcoef_compute(preds, target)
    _r = relu(r)
    return _r


def pearson_corrcoef_compute(
    preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """ Custom implementation of the PCC for data with 4 dimensions: (B, C, M, N) """

    preds_diff = preds.view(preds.shape[0], -1) - preds.mean(dim=(1, 2, 3)).view(
        preds.shape[0], -1
    )
    target_diff = target.view(target.shape[0], -1) - target.mean(dim=(1, 2, 3)).view(
        target.shape[0], -1
    )

    cov = (preds_diff * target_diff).mean(dim=1)
    preds_std = torch.sqrt((preds_diff * preds_diff).mean(dim=1))
    target_std = torch.sqrt((target_diff * target_diff).mean(dim=1))

    denom = preds_std * target_std
    # prevent division by zero
    zero_mask = denom == 0
    if torch.nonzero(zero_mask).numel():
        denom[zero_mask] = eps

    corrcoef = cov / denom
    return torch.clamp(corrcoef, -1.0, 1.0)
