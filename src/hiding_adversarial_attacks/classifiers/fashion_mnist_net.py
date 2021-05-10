from __future__ import print_function

from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet


class FashionMNISTNet(MNISTNet):
    """
    Simple Convolutional FashionMNIST classifier.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
