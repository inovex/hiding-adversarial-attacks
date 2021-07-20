import os
from typing import List, Tuple, Union

import numpy as np
import torch
from captum.attr._utils import visualization as viz
from matplotlib.pyplot import axis, figure

from hiding_adversarial_attacks.visualization.helpers import tensor_to_pil_numpy


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
