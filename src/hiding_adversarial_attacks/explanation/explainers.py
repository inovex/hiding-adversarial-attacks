from abc import abstractmethod
from typing import Union

import pytorch_lightning as pl
import torch
import torchvision
from captum.attr import DeepLift

from hiding_adversarial_attacks.config.create_explanations_config import (
    ExplanationConfig,
)
from hiding_adversarial_attacks.config.explanation.deep_lift_baseline_config import (
    DeepLiftBaselineNames,
)
from hiding_adversarial_attacks.config.explanation.explainer_config import (
    ExplainerNames,
)


class AbstractExplainer:
    def __init__(self, model: pl.LightningModule, random_seed: int = 42):
        self._model = model
        self._random_seed = random_seed

    @property
    def xai_algorithm(self):
        try:
            return self._xai_algorithm
        except AttributeError:
            raise NotImplementedError("No XAI algorithm specified.")

    @abstractmethod
    def explain(self, image: torch.Tensor, target: torch.Tensor, **kwargs):
        pass


class DeepLiftExplainer(AbstractExplainer):
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
                means = image.view(batch_size, -1).mean(1, keepdim=True)
                ones = torch.ones_like(image)
                baseline = torch.ones_like(image) * means.view(
                    batch_size, image.shape[1], 1, 1
                ).expand_as(ones)
            else:
                raise NotImplementedError("Unknown baseline specified.")

            return baseline

        return inner


def get_explainer(
    model: pl.LightningModule, config: ExplanationConfig
) -> Union[DeepLiftExplainer]:

    explainer_name = config.explainer.name
    if explainer_name == ExplainerNames.DEEP_LIFT:
        return DeepLiftExplainer(
            model,
            baseline_name=config.explainer.baseline.name,
            multiply_by_inputs=config.explainer.multiply_by_inputs,
        )
    elif explainer_name == ExplainerNames.GRAD_CAM:
        raise NotImplementedError("GradCAM not implemented yet.")
    else:
        raise SystemExit("ERROR: Unknown explainer. Exiting.")
