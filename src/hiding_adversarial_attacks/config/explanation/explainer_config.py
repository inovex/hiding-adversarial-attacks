from dataclasses import dataclass

from omegaconf import MISSING

from hiding_adversarial_attacks.config.explanation.deep_lift_baseline_config import (
    DeepLiftBaselineConfig,
)


class ExplainerNames:
    DEEP_LIFT = "DeepLIFT"
    GRAD_CAM = "GradCAM"


@dataclass
class ExplainerConfig:
    name: str = MISSING
    baseline: DeepLiftBaselineConfig = MISSING


@dataclass
class DeepLiftConfig(ExplainerConfig):
    name: str = ExplainerNames.DEEP_LIFT
    baseline: DeepLiftBaselineConfig = MISSING
