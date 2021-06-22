import pytorch_lightning as pl
import torch
from captum.attr import GuidedBackprop

from hiding_adversarial_attacks.explainers.base import BaseExplainer


class GuidedBackpropExplainer(BaseExplainer):
    def __init__(
        self,
        model: pl.LightningModule,
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed)
        self._xai_algorithm = GuidedBackprop(self._model)

    def explain(self, image: torch.Tensor, target: torch.Tensor, **kwargs):
        self._model.zero_grad()
        return self.xai_algorithm.attribute(image, target=target, **kwargs)
