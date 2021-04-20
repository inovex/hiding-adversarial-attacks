from dataclasses import dataclass

from omegaconf import MISSING


class DeepLiftBaselineNames:
    ZERO = "zero"
    BLUR = "blur"
    LOCAL_MEAN = "local_mean"


@dataclass(frozen=True)
class DeepLiftBaselineConfig:
    name: str = MISSING


@dataclass(frozen=True)
class ZeroBaselineConfig(DeepLiftBaselineConfig):
    name: str = DeepLiftBaselineNames.ZERO


@dataclass(frozen=True)
class LocalMeanBaselineConfig(DeepLiftBaselineConfig):
    name: str = DeepLiftBaselineNames.LOCAL_MEAN


@dataclass(frozen=True)
class BlurBaselineConfig(DeepLiftBaselineConfig):
    name: str = DeepLiftBaselineNames.BLUR
