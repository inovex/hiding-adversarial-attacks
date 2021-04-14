from abc import abstractmethod

import pytorch_lightning as pl
import torch


class BaseExplainer:
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
