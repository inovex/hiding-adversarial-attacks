from __future__ import print_function

from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.manipulation.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)


class ManipulatedFashionMNISTNet(ManipulatedMNISTNet):
    def __init__(self, model: MNISTNet, hparams):
        super().__init__(model=model, hparams=hparams)
