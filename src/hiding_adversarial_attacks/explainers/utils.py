from typing import Union

import pytorch_lightning as pl

from hiding_adversarial_attacks.config.explainers.explainer_config import ExplainerNames
from hiding_adversarial_attacks.explainers.deep_lift import DeepLiftExplainer
from hiding_adversarial_attacks.explainers.guided_backprop import (
    GuidedBackpropExplainer,
)
from hiding_adversarial_attacks.explainers.input_x_gradients import (
    InputXGradientExplainer,
)
from hiding_adversarial_attacks.explainers.integrated_gradients import (
    IntegratedGradientsExplainer,
)
from hiding_adversarial_attacks.explainers.layer_deep_lift import LayerDeepLiftExplainer
from hiding_adversarial_attacks.explainers.layer_grad_cam import LayerGradCamExplainer
from hiding_adversarial_attacks.explainers.lrp import LRPExplainer


def get_explainer(
    model: pl.LightningModule, config
) -> Union[
    DeepLiftExplainer,
    LayerDeepLiftExplainer,  # noqa: F821
    GuidedBackpropExplainer,
    InputXGradientExplainer,
    LayerGradCamExplainer,
    LRPExplainer,
]:

    explainer_name = config.explainer.name
    if explainer_name == ExplainerNames.DEEP_LIFT:
        return DeepLiftExplainer(
            model,
            baseline_name=config.explainer.baseline.name,
            multiply_by_inputs=config.explainer.multiply_by_inputs,
            relu_attributions=config.explainer.relu_attributions,
        )
    elif explainer_name == ExplainerNames.LAYER_DEEP_LIFT:
        return LayerDeepLiftExplainer(  # noqa: F821
            model,
            layer_name=config.explainer.layer_name,
            image_shape=(
                config.data_set.image_width,
                config.data_set.image_height,
            ),
            baseline_name=config.explainer.baseline.name,
            multiply_by_inputs=config.explainer.multiply_by_inputs,
            relu_attributions=config.explainer.relu_attributions,
        )
    elif explainer_name == ExplainerNames.INTEGRATED_GRADIENTS:
        return IntegratedGradientsExplainer(
            model,
            baseline_name=config.explainer.baseline.name,
            multiply_by_inputs=config.explainer.multiply_by_inputs,
            relu_attributions=config.explainer.relu_attributions,
        )
    elif explainer_name == ExplainerNames.GUIDED_BACKPROP:
        return GuidedBackpropExplainer(
            model,
        )
    elif explainer_name == ExplainerNames.INPUT_X_GRADIENT:
        return InputXGradientExplainer(
            model,
        )
    elif explainer_name == ExplainerNames.LRP:
        return LRPExplainer(
            model,
        )
    elif explainer_name == ExplainerNames.GRAD_CAM:
        return LayerGradCamExplainer(
            model,
            layer_name=config.explainer.layer_name,
            image_shape=(
                config.data_set.image_width,
                config.data_set.image_height,
            ),
            relu_attributions=config.explainer.relu_attributions,
        )
    else:
        raise SystemExit("ERROR: Unknown explainer. Exiting.")
