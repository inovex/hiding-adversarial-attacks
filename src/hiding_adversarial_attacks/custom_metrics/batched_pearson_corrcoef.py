import torch
from torch import Tensor
from torchmetrics import PearsonCorrcoef
from torchmetrics.functional.regression.pearson import (
    _pearson_corrcoef_compute,
    _pearson_corrcoef_update,
)
from torchmetrics.utilities.checks import _check_same_shape


class BatchedPearsonCorrcoef(PearsonCorrcoef):
    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        _check_same_shape(preds, target)
        if preds.ndim > 1 or target.ndim > 1:
            for preds_single, target_single in zip(preds, target):
                _preds, _target = _pearson_corrcoef_update(
                    preds_single.view(-1), target_single.view(-1)
                )
                self.preds.append(_preds)
                self.target.append(_target)

    def compute(self):
        """
        Computes pearson correlation coefficient over state.
        """
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        return _pearson_corrcoef_compute(preds, target)
