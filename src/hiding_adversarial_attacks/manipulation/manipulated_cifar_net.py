from __future__ import print_function

import torch

from hiding_adversarial_attacks.manipulation.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)


class ManipulatedCIFARNet(ManipulatedMNISTNet):
    def __init__(self, model, hparams):
        super().__init__(model=model, hparams=hparams)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return [optimizer]
