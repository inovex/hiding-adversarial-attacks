import pytorch_lightning as pl
import torch
from captum.attr import InputXGradient

from hiding_adversarial_attacks.explainers.base import BaseExplainer
from hiding_adversarial_attacks.explainers.captum_patches import (
    custom_compute_gradients,
)


class InputXGradientExplainer(InputXGradient, BaseExplainer):
    def __init__(
        self,
        model: pl.LightningModule,
    ):
        super().__init__(model)
        self.gradient_func = custom_compute_gradients

    def explain(self, image: torch.Tensor, target: torch.Tensor, **kwargs):
        attribution = self.attribute(image, target=target, **kwargs)
        return attribution
