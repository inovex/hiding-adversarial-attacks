from dataclasses import dataclass

from omegaconf import MISSING


class DeepLiftBaselineNames:
    ZERO = "zero"
    BLUR = "blur"
    LOCAL_MEAN = "local_mean"


@dataclass
class DeepLiftBaselineConfig:
    name: str = MISSING


@dataclass
class ZeroBaselineConfig(DeepLiftBaselineConfig):
    name: str = DeepLiftBaselineNames.ZERO


@dataclass
class LocalMeanBaselineConfig(DeepLiftBaselineConfig):
    name: str = DeepLiftBaselineNames.LOCAL_MEAN


@dataclass
class BlurBaselineConfig(DeepLiftBaselineConfig):
    name: str = DeepLiftBaselineNames.BLUR
