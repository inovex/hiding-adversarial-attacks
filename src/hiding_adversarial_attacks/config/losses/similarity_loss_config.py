from dataclasses import dataclass
from functools import partial

from omegaconf import MISSING
from torchmetrics.functional import mean_squared_error, ssim

from hiding_adversarial_attacks.custom_metrics.relu_pearson_corrcoef import (
    relu_pearson_corrcoef,
)


class SimilarityLossNames:
    MSE: str = "MSE"
    PCC: str = "PCC"
    SSIM: str = "SSIM"


pssim = partial(ssim, kernel_size=(5, 5), sigma=(0.3, 0.3), data_range=1)

SimilarityLossMapping = {
    SimilarityLossNames.MSE: mean_squared_error,
    SimilarityLossNames.SSIM: pssim,
    SimilarityLossNames.PCC: relu_pearson_corrcoef,
}


@dataclass
class SimilarityLoss:
    name: str = MISSING


@dataclass
class MSELoss(SimilarityLoss):
    name: str = SimilarityLossNames.MSE


@dataclass
class PCCLoss(SimilarityLoss):
    name: str = SimilarityLossNames.PCC


@dataclass
class SSIMLoss(SimilarityLoss):
    name: str = SimilarityLossNames.SSIM
