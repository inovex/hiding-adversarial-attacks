from typing import Union

import pytorch_lightning as pl

from hiding_adversarial_attacks.config.explainers.explainer_config import ExplainerNames
from hiding_adversarial_attacks.explainers.deep_lift import DeepLiftExplainer
from hiding_adversarial_attacks.explainers.layer_grad_cam import LayerGradCamExplainer


def get_explainer(
    model: pl.LightningModule, config
) -> Union[DeepLiftExplainer, LayerGradCamExplainer]:

    explainer_name = config.explainer.name
    if explainer_name == ExplainerNames.DEEP_LIFT:
        return DeepLiftExplainer(
            model,
            baseline_name=config.explainer.baseline.name,
            multiply_by_inputs=config.explainer.multiply_by_inputs,
        )
    elif explainer_name == ExplainerNames.GRAD_CAM:
        return LayerGradCamExplainer(
            model,
            layer_name=config.explainer.layer_name,
            image_shape=(config.data_set.image_width, config.data_set.image_height),
            relu_attributions=config.explainer.relu_attributions,
        )
    else:
        raise SystemExit("ERROR: Unknown explainer. Exiting.")
