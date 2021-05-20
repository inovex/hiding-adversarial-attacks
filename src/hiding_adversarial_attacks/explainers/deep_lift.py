import pytorch_lightning as pl
import torch
import torchvision
from captum.attr import DeepLift

from hiding_adversarial_attacks.config.explainers.deep_lift_baseline_config import (
    DeepLiftBaselineNames,
)
from hiding_adversarial_attacks.explainers.base import BaseExplainer


class DeepLiftExplainer(BaseExplainer):
    _gaussian_blur = torchvision.transforms.GaussianBlur(11, 0.5)

    def __init__(
        self,
        model: pl.LightningModule,
        baseline_name: str = DeepLiftBaselineNames.ZERO,
        multiply_by_inputs: bool = False,
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed)
        self._multiply_by_inputs = multiply_by_inputs
        self._xai_algorithm = DeepLift(self._model, self._multiply_by_inputs)
        self._baseline_name = baseline_name
        self._baseline = self._baseline_wrapper()

    def explain(self, image: torch.Tensor, target: torch.Tensor, **kwargs):
        self._model.zero_grad()
        return self.xai_algorithm.attribute(
            image, target=target, baselines=self._baseline(image), **kwargs
        )

    def _baseline_wrapper(self):
        def inner(image: torch.Tensor):
            if self._baseline_name == DeepLiftBaselineNames.ZERO:
                baseline = image * 0
            elif self._baseline_name == DeepLiftBaselineNames.BLUR:
                baseline = self._gaussian_blur(image)
            elif self._baseline_name == DeepLiftBaselineNames.LOCAL_MEAN:
                batch_size = image.shape[0]
                if image.shape[1] == 3:  # rgb
                    means = image.view(batch_size, 3, -1).mean(2, keepdim=True)
                else:
                    means = image.view(batch_size, -1).mean(1, keepdim=True)
                ones = torch.ones_like(image)
                baseline = torch.ones_like(image) * means.view(
                    batch_size, image.shape[1], 1, 1
                ).expand_as(ones)
            else:
                raise NotImplementedError("Unknown baseline specified.")

            return baseline

        return inner
