import argparse
import os
from functools import wraps
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToPILImage

from hiding_adversarial_attacks.classifiers.cifar_net import CifarNet
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames

toPilImage = ToPILImage()

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f"function {func.__name__} took {1000*(te-ts):.1f} ms")
        return result

    return wrap


def get_model(config):
    if (
        config.data_set.name == DataSetNames.MNIST
        or config.data_set.name == DataSetNames.FASHION_MNIST
    ):
        return MNISTNet(config)
    elif config.data_set.name == DataSetNames.CIFAR10:
        return CifarNet(config)
    else:
        raise SystemExit(f"Unknown data set specified: {config.data_set}. Exiting.")


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


def display_tensor_as_image(tensor: torch.Tensor, cmap: str = "gray"):
    plt.imshow(tensor.squeeze().numpy(), cmap=cmap)
    plt.show()


def display_adversarial_difference_image(
    adversarial: torch.Tensor, original: torch.Tensor, cmap: str = "gray"
):
    adv_difference = torch.abs(adversarial - original)
    display_tensor_as_image(adv_difference, cmap=cmap)


def tensor_to_pil_numpy(rgb_tensor):
    return np.transpose(rgb_tensor.cpu().detach().numpy(), (0, 2, 3, 1))


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def to_pil_image(tensor, mode="L"):
    return ToPILImage(mode=mode)(tensor)


if __name__ == "__main__":
    orig = torch.load(
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/MNIST/DeepFool/epsilon_0.225/"
        "class_1/training_orig.pt"
    )
    adv = torch.load(
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/MNIST/DeepFool/epsilon_0.225/"
        "class_1/training_adv.pt"
    )
    display_adversarial_difference_image(adv[0][0], orig[0][0])
