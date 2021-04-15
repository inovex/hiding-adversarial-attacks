from dataclasses import dataclass

from omegaconf import MISSING

from hiding_adversarial_attacks.config.explainers.deep_lift_baseline_config import (
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
    multiply_by_inputs: bool = False


@dataclass
class LayerGradCamConfig(ExplainerConfig):
    name: str = ExplainerNames.GRAD_CAM
    layer_name: str = "conv2"
    relu_attributions: bool = False
