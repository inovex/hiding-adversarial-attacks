from __future__ import print_function

import torch
from torch.optim.lr_scheduler import StepLR

from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.manipulation.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)


class ManipulatedFashionMNISTNet(ManipulatedMNISTNet):
    def __init__(self, model: MNISTNet, hparams):
        super().__init__(model=model, hparams=hparams)
        self.gamma = hparams.gamma
        self.steps_lr = hparams.steps_lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.steps_lr > 0 or self.gamma != 1.0:
            scheduler = StepLR(optimizer, step_size=self.steps_lr, gamma=self.gamma)
            return [optimizer], [scheduler]
        return [optimizer]
