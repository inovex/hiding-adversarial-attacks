import os
from typing import Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from numpy import ndarray

from hiding_adversarial_attacks.visualization.helpers import display_tensor_as_image


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
