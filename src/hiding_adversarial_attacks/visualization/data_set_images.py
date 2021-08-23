import os
from typing import List, Tuple, Union

import numpy as np
import torch
from matplotlib import pylab
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from numpy import ndarray

from hiding_adversarial_attacks.visualization.config import DATA_SET_MAPPING
from hiding_adversarial_attacks.visualization.helpers import (
    display_tensor_as_image,
    tensor_to_pil_numpy,
)


def display_random_original_and_adversarial_training_image(path: str, output_path: str):
    orig_images, orig_labels = torch.load(os.path.join(path, "training_orig.pt"))
    adv_images, adv_labels = torch.load(os.path.join(path, "training_adv.pt"))
    random_idx = np.random.randint(0, len(orig_labels))
    orig_img, orig_label = orig_images[random_idx], orig_labels[random_idx]
    adv_img, adv_label = adv_images[random_idx], adv_labels[random_idx]
    display_tensor_as_image(
        orig_img, title=f"Original, label={orig_label}, idx={random_idx}"
    )
    orig_path = os.path.join(output_path, f"{int(orig_label)}_{random_idx}_orig.png")
    plt.imsave(orig_path, tensor_to_pil_numpy(orig_img))
    display_tensor_as_image(
        adv_img, title=f"Adversarial, label={adv_label}, idx={random_idx}"
    )
    adv_path = os.path.join(
        output_path, f"{int(orig_label)}_{int(adv_label)}_{random_idx}_adv.png"
    )
    plt.imsave(adv_path, tensor_to_pil_numpy(adv_img))


def display_original_and_adversarial_image_grid(
    paths: List[str],
    class_ids_list: List,
    data_set_names: List[str],
    title: str = "",
):

    params = {
        "legend.fontsize": "large",
        "figure.figsize": (16, 8),
        "figure.titlesize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
    }
    pylab.rcParams.update(params)

    fig, axes = plt.subplots(
        len(class_ids_list[0]), 4, figsize=(12, 10), sharex=True, sharey=True
    )

    cols = [0, 1]
    cmap = "gray"
    for i, (path, data_set_name) in enumerate(zip(paths, data_set_names)):
        orig_images, orig_labels = torch.load(os.path.join(path, "training_orig.pt"))
        adv_images, adv_labels = torch.load(os.path.join(path, "training_adv.pt"))
        mapping = DATA_SET_MAPPING[data_set_name]

        display_image_grid(
            orig_images,
            orig_labels,
            adv_images,
            adv_labels,
            axes,
            class_ids_list[i],
            mapping,
            cols=cols,
            cmap=cmap,
        )
        cols = [col + 2 for col in cols]
    plt.axis("off")
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    fig.show()


def display_image_grid(
    orig_images,
    orig_labels,
    adv_images,
    adv_labels,
    axes,
    class_ids,
    mapping,
    cols: List[int],
    cmap: str = "gray",
):
    for idx, class_id in enumerate(class_ids):
        class_indeces = torch.nonzero(orig_labels == class_id)
        rand = np.random.randint(0, len(class_indeces))
        rand_mask_idx = class_indeces[rand]
        orig_label = orig_labels[rand_mask_idx]
        orig_img = tensor_to_pil_numpy(orig_images[rand_mask_idx].squeeze(0))
        adv_label = adv_labels[rand_mask_idx]
        adv_img = tensor_to_pil_numpy(adv_images[rand_mask_idx].squeeze(0))

        axes[idx][cols[0]].set_title(f"Original label:\n{mapping[int(orig_label)]}")
        axes[idx][cols[0]].imshow(orig_img, cmap=cmap, interpolation="nearest")
        axes[idx][cols[0]].set_axis_off()

        axes[idx][cols[1]].set_title(f"Adversarial label:\n{mapping[int(adv_label)]}")
        axes[idx][cols[1]].imshow(adv_img, cmap=cmap, interpolation="nearest")
        axes[idx][cols[1]].set_axis_off()


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


if __name__ == "__main__":
    class_ids = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
    paths = [
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack="
        "DeepFool--eps=0.105--cp-run=HAA-1728",
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack="
        "DeepFool--eps=0.105--cp-run=HAA-1728",
    ]
    data_set_names = ["FashionMNIST", "FashionMNIST"]
    display_original_and_adversarial_image_grid(
        paths,
        class_ids,
        data_set_names,
        "Original and adversarial Fashion-MNIST images",
    )
