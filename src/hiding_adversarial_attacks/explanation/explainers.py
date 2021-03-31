import pytorch_lightning as pl
import torch
import torchvision
from captum.attr import DeepLift

from hiding_adversarial_attacks.config import DeepLiftBaselineConfig


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

    def explain(self, image: torch.Tensor, target_label: torch.Tensor, **kwargs):
        return self.xai_algorithm.attribute(image, target=target_label, **kwargs)


class DeepLiftExplainer(AbstractExplainer):
    _gaussian_blur = torchvision.transforms.GaussianBlur(11, 0.5)

    def __init__(
        self,
        model: pl.LightningModule,
        baseline_name: str = DeepLiftBaselineConfig.ZERO,
        random_seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(model=model, random_seed=random_seed)
        self._xai_algorithm = DeepLift(self._model)
        self._baseline_name = baseline_name
        self._device = device
        self._baseline = self._baseline_wrapper()

    def explain(
        self,
        image: torch.Tensor,
        target_label: torch.Tensor,
        **kwargs,
    ):
        return self.xai_algorithm.attribute(
            image, target=target_label, baselines=self._baseline(image), **kwargs
        )

    def _baseline_wrapper(self):
        device = self._device

        def inner(image: torch.Tensor):
            if self._baseline_name == DeepLiftBaselineConfig.ZERO:
                baseline = image * 0
            elif self._baseline_name == DeepLiftBaselineConfig.BLUR:
                baseline = self._gaussian_blur(image)
            elif self._baseline_name == DeepLiftBaselineConfig.LOCAL_MEAN:
                batch_size = image.shape[0]
                means = image.view(batch_size, -1).mean(1, keepdim=True)
                baseline = torch.Tensor().to(device)
                ones = torch.ones(1, image.shape[1], image.shape[2], image.shape[3]).to(
                    device
                )
                for mean in means:
                    mean_img = ones * mean
                    baseline = torch.cat((baseline, mean_img), 0)
            else:
                raise NotImplementedError("Unknown baseline specified.")

            return baseline

        return inner
