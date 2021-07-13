from dataclasses import dataclass

from omegaconf import MISSING

from hiding_adversarial_attacks.config.explainers.deep_lift_baseline_config import (
    DeepLiftBaselineConfig,
)


class ExplainerNames:
    LAYER_DEEP_LIFT = "LayerDeepLIFT"
    DEEP_LIFT = "DeepLIFT"
    GUIDED_BACKPROP = "GuidedBackprop"
    GRAD_CAM = "GradCAM"
    INPUT_X_GRADIENT = "InputXGradient"


@dataclass
class ExplainerConfig:
    name: str = MISSING
    baseline: DeepLiftBaselineConfig = MISSING


@dataclass
class DeepLiftConfig(ExplainerConfig):
    name: str = ExplainerNames.DEEP_LIFT
    baseline: DeepLiftBaselineConfig = MISSING
    multiply_by_inputs: bool = False
    relu_attributions: bool = False


@dataclass
class LayerDeepLiftConfig(ExplainerConfig):
    name: str = ExplainerNames.LAYER_DEEP_LIFT
    baseline: DeepLiftBaselineConfig = MISSING
    layer_name: str = "conv2"
    multiply_by_inputs: bool = False
    relu_attributions: bool = False


@dataclass
class GuidedBackpropConfig(ExplainerConfig):
    name: str = ExplainerNames.GUIDED_BACKPROP


@dataclass
class InputXGradientConfig(ExplainerConfig):
    name: str = ExplainerNames.INPUT_X_GRADIENT


@dataclass
class LayerGradCamConfig(ExplainerConfig):
    name: str = ExplainerNames.GRAD_CAM
    layer_name: str = "conv2"
    relu_attributions: bool = False
