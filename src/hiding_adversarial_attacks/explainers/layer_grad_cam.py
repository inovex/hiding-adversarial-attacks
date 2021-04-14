from typing import Tuple

import pytorch_lightning as pl
import torch
from captum.attr import LayerAttribution, LayerGradCam

from hiding_adversarial_attacks.explainers.base import BaseExplainer


class LayerGradCamExplainer(BaseExplainer):
    def __init__(
        self,
        model: pl.LightningModule,
        layer_name: str,
        image_shape: Tuple[int, int],
        relu_attributions: bool = False,
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed)
        self._layer_name = layer_name
        self._image_shape = image_shape
        self._relu_attributions = relu_attributions
        self._layer = self._get_conv2d_layer_by_name(self._layer_name)
        self._xai_algorithm = LayerGradCam(self._model, self._layer)

    def explain(self, image: torch.Tensor, target: torch.Tensor, **kwargs):
        self._model.zero_grad()
        attribution = self.xai_algorithm.attribute(
            image, target=target, relu_attributions=self._relu_attributions, **kwargs
        )
        # attribution needs to be interpolated to match the inout size
        interpolated_attribution = LayerAttribution.interpolate(
            attribution, self._image_shape
        )
        return interpolated_attribution

    def _get_conv2d_layer_by_name(self, layer_name: str):
        named_modules = dict(self._model.named_modules())
        assert layer_name in named_modules, f"Layer name '{layer_name}' not in model."
        assert (
            type(named_modules[layer_name]) == torch.nn.modules.conv.Conv2d
        ), f"Specified layer '{layer_name}' is not of type Conv2d."
        return named_modules[layer_name]
