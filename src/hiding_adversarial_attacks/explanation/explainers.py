from abc import abstractmethod
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision
from captum.attr import DeepLift, LayerAttribution, LayerGradCam

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


class LayerGradCamExplainer(AbstractExplainer):
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
