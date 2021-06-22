import argparse
import os
from functools import wraps
from time import time
from typing import Any, List, Tuple, Union

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

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f"function {func.__name__} took {1000*(te-ts):.1f} ms")
        return result

    return wrap


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


def display_tensor_as_image(
    tensor: torch.Tensor, title: str = None, cmap: str = "gray"
):
    np_img = tensor_to_pil_numpy(tensor)
    plt.imshow(np_img, cmap=cmap)
    if title is not None:
        plt.title(title)
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
    if len(rgb_tensor.shape) == 3:
        return np.transpose(rgb_tensor.cpu().detach().numpy(), (1, 2, 0))
    else:
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


def assert_not_none(tensor, loss_name):
    assert not torch.isnan(tensor).any(), f"NaN in {loss_name}!"


def display_random_original_and_adversarial_training_image(path: str):
    orig_images, orig_labels = torch.load(os.path.join(path, "training_orig.pt"))
    adv_images, adv_labels = torch.load(os.path.join(path, "training_adv.pt"))
    random_idx = np.random.randint(0, len(orig_labels))
    orig_img, orig_label = orig_images[random_idx], orig_labels[random_idx]
    adv_img, adv_label = adv_images[random_idx], adv_labels[random_idx]
    display_tensor_as_image(
        orig_img, title=f"Original, label={orig_label}, idx={random_idx}"
    )
    display_tensor_as_image(
        adv_img, title=f"Adversarial, label={adv_label}, idx={random_idx}"
    )


def display_random_original_and_adversarial_explanation(path: str):
    orig_images, orig_labels = torch.load(os.path.join(path, "training_orig.pt"))
    orig_explanations, _, _ = torch.load(os.path.join(path, "training_orig_exp.pt"))
    adv_images, adv_labels = torch.load(os.path.join(path, "training_adv.pt"))
    adv_explanations, _, _ = torch.load(os.path.join(path, "training_adv_exp.pt"))
    random_idx = np.random.randint(0, len(orig_labels))
    orig_img, orig_label = orig_images[random_idx].unsqueeze(0), int(
        orig_labels[random_idx]
    )
    adv_img, adv_label = adv_images[random_idx].unsqueeze(0), int(
        adv_labels[random_idx]
    )
    orig_expl, adv_expl = (
        orig_explanations[random_idx].unsqueeze(0),
        adv_explanations[random_idx].unsqueeze(0),
    )
    visualize_explanations(
        orig_img,
        orig_expl,
        [f"Original explanation, label={orig_label}, idx={random_idx}"],
        display_figure=True,
    )
    visualize_explanations(
        adv_img,
        adv_expl,
        [f"Adversarial explanation, label={adv_label}, idx={random_idx}"],
        display_figure=True,
    )


def create_mask(source: torch.Tensor, included_classes: List[Any]):
    if ALL_CLASSES in included_classes:
        return torch.ones(len(source), dtype=torch.bool, device=source.device)
    mask = torch.zeros(len(source), dtype=torch.bool, device=source.device)
    for c in included_classes:
        mask += source == c
    return mask


if __name__ == "__main__":
    path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data"
        "/preprocessed/adversarial/data-set=FashionMNIST--attack="
        "DeepFool--eps=0.105--cp-run=HAA-952/exp=DeepLIFT--bl=zero--mbi=False"
    )
    # display_random_original_and_adversarial_training_image(path)
    display_random_original_and_adversarial_explanation(path)
