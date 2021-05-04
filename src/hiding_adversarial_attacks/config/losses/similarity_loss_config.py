from dataclasses import dataclass

from omegaconf import MISSING
from torchmetrics.functional import mean_squared_error, pearson_corrcoef, ssim


class SimilarityLossNames:
    MSE: str = "MSE"
    PCC: str = "PCC"
    SSIM: str = "SSIM"


SimilarityLossMapping = {
    SimilarityLossNames.MSE: mean_squared_error,
    SimilarityLossNames.SSIM: ssim,
    SimilarityLossNames.PCC: pearson_corrcoef,
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
