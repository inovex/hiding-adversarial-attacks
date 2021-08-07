from typing import Tuple

import pytorch_lightning as pl
import torch
from captum.attr import LayerGradCam

from hiding_adversarial_attacks.classifiers.utils import get_conv2d_layer_by_name
from hiding_adversarial_attacks.explainers.base import BaseExplainer


class LayerGradCamExplainer(BaseExplainer):
    def __init__(
        self,
        model: pl.LightningModule,
        layer_name: str,
        image_shape: Tuple[int, int],
        relu_attributions: bool = False,
    ):
        super().__init__(model)
        self._layer_name = layer_name
        self._image_shape = image_shape
        self._relu_attributions = relu_attributions
        self._layer = get_conv2d_layer_by_name(self._model, self._layer_name)
        self._xai_algorithm = LayerGradCam(self._model, self._layer)

    def explain(self, image: torch.Tensor, target: torch.Tensor, **kwargs):
        attribution = self.xai_algorithm.attribute(
            image,
            target=target,
            relu_attributions=self._relu_attributions,
            **kwargs,
        )
        return attribution
