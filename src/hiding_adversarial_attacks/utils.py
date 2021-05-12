import argparse
import os
from functools import wraps
from time import time
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from captum.attr._utils import visualization as viz
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from numpy import ndarray
from torchvision.transforms import ToPILImage

from hiding_adversarial_attacks.classifiers.cifar_net import CifarNet
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames


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
    if config.data_set.name in [DataSetNames.MNIST, DataSetNames.FASHION_MNIST]:
        return MNISTNet(config)
    elif config.data_set.name == DataSetNames.CIFAR10:
        return CifarNet(config)
    else:
        raise SystemExit(
            f"Unknown data set specified: {config.data_set.name}. Exiting."
        )


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


def display_tensor_as_image(tensor: torch.Tensor, cmap: str = "gray"):
    plt.imshow(tensor.squeeze().numpy(), cmap=cmap)
    plt.show()


def visualize_adversarial_difference_image(
    adversarial: torch.Tensor, original: torch.Tensor, cmap: str = "gray"
):
    adv_difference = torch.abs(adversarial - original)
    display_tensor_as_image(adv_difference, cmap=cmap)


def visualize_difference_image_np(
    adversarial: ndarray,
    original: ndarray,
    title: str = None,
    cmap: str = "gray",
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    display_figure: bool = True,
):
    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if display_figure:
            plt_fig, plt_axis = plt.subplots(figsize=(6, 6))
        else:
            plt_fig = Figure(figsize=(6, 6))
            plt_axis = plt_fig.subplots()

    diff_image = np.abs(adversarial - original)
    plt_axis.imshow(diff_image, cmap=cmap)

    if title:
        plt_axis.set_title(title)

    if display_figure:
        plt.show()

    return plt_fig, plt_axis


def tensor_to_pil_numpy(rgb_tensor):
    return np.transpose(rgb_tensor.cpu().detach().numpy(), (0, 2, 3, 1))


def to_pil_image(tensor, mode="L"):
    return ToPILImage(mode=mode)(tensor)


def visualize_explanations(
    images: torch.Tensor,
    explanations: torch.Tensor,
    titles: List[str],
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    display_figure: bool = False,
):

    imgs = tensor_to_pil_numpy(images)
    expls = tensor_to_pil_numpy(explanations)
    figures = []

    for image, explanation, title in zip(imgs, expls, titles):
        ax, fig = visualize_single_explanation(
            image, explanation, title, plt_fig_axis, display_figure
        )
        figures.append((fig, ax))
    return figures


def visualize_single_explanation(
    image: np.array,
    explanation: np.array,
    title: str,
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    display_figure: bool = False,
):
    fig, ax = viz.visualize_image_attr(
        explanation,
        image,
        method="blended_heat_map",
        sign="all",
        show_colorbar=True,
        plt_fig_axis=plt_fig_axis,
        title=title,
        use_pyplot=display_figure,
    )
    return ax, fig


def save_confusion_matrix(matrix: np.array, log_path: str):
    _matrix = matrix.astype("int")
    index = list(range(0, matrix.shape[0]))
    columns = range(0, matrix.shape[1])
    df = pd.DataFrame(_matrix, index=index, columns=columns)
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
    fig.savefig(os.path.join(log_path, "confusion_matrix.png"))
    fig.show()


def normalize_to_range(x: torch.Tensor, min: int = 0, max: int = 1):
    return min + ((x - torch.min(x)) * (max - min)) / (torch.max(x) - torch.min(x))


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
    visualize_adversarial_difference_image(adv[0][0], orig[0][0])


def assert_not_none(tensor, loss_name):
    assert not torch.isnan(tensor).any(), f"NaN in {loss_name}!"
