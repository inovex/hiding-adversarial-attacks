from __future__ import print_function

import torch
from torch.optim.lr_scheduler import StepLR

from hiding_adversarial_attacks.manipulation.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)


class ManipulatedCIFARNet(ManipulatedMNISTNet):
    def __init__(self, model, hparams):
        super().__init__(model=model, hparams=hparams)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.hparams["weight_decay"],
        )
        if self.hparams["steps_lr"] > 0 and self.hparams["gamma"] != 1.0:
            scheduler = StepLR(
                optimizer,
                step_size=self.hparams["steps_lr"],
                gamma=self.hparams["gamma"],
            )
            return [optimizer], [scheduler]
        return [optimizer]
