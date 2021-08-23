import os
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from torchvision.transforms import transforms

from hiding_adversarial_attacks.eda.initial_similarities_visualization import (
    load_explanations,
)
from hiding_adversarial_attacks.visualization.config import DATA_SET_MAPPING
from hiding_adversarial_attacks.visualization.explanations import (
    interpolate_explanations,
    visualize_single_explanation,
)
from hiding_adversarial_attacks.visualization.helpers import tensor_to_pil_numpy

to_pil_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


def get_explanations_and_images_by_class(data_path, class_ids, size=(28, 28)):
    (
        orig_expl,
        _,
        adv_expl,
        _,
    ) = load_explanations(data_path)
    orig_images, orig_labels = torch.load(os.path.join(data_path, "training_orig.pt"))
    adv_images, adv_labels = torch.load(os.path.join(data_path, "training_adv.pt"))
    rand_indeces = torch.tensor([]).long()
    for idx, class_id in enumerate(class_ids):
        class_indeces = torch.nonzero(orig_labels == class_id)
        rand_mask_idx = class_indeces[np.random.randint(0, len(class_indeces))]
        rand_indeces = torch.cat((rand_indeces, rand_mask_idx), dim=0)
    print(rand_indeces)

    orig_label = orig_labels[rand_indeces]
    orig_exp = orig_expl[rand_indeces]
    orig_exp = interpolate(orig_exp, size=28)
    orig_img = orig_images[rand_indeces]
    orig_img = interpolate(orig_img, size=28)
    adv_label = adv_labels[rand_indeces]
    adv_exp = adv_expl[rand_indeces]
    adv_exp = interpolate(adv_exp, size=28)
    adv_img = adv_images[rand_indeces]
    adv_img = interpolate(adv_img, size=28)

    return orig_img, orig_exp, orig_label, adv_img, adv_exp, adv_label


def get_explanations_and_images_by_indices(data_path, rand_indices, size=(28, 28)):
    (
        orig_expl,
        _,
        adv_expl,
        _,
    ) = load_explanations(data_path)
    orig_images, orig_labels = torch.load(os.path.join(data_path, "training_orig.pt"))
    adv_images, adv_labels = torch.load(os.path.join(data_path, "training_adv.pt"))
    print(rand_indices)

    orig_label = orig_labels[rand_indices]
    orig_exp = orig_expl[rand_indices]
    orig_exp = interpolate(orig_exp, size=28)
    orig_img = orig_images[rand_indices]
    orig_img = interpolate(orig_img, size=28)
    adv_label = adv_labels[rand_indices]
    adv_exp = adv_expl[rand_indices]
    adv_exp = interpolate(adv_exp, size=28)
    adv_img = adv_images[rand_indices]
    adv_img = interpolate(adv_img, size=28)

    return orig_img, orig_exp, orig_label, adv_img, adv_exp, adv_label


def plot_initial_explanations(
    paths: List[str],
    data_set_names: List[str],
    indices: torch.Tensor,
):
    fig, axes = plt.subplots(
        len(indices), 4, figsize=(12, 10), sharex=True, sharey=True
    )

    cols = [0, 1]
    for path, data_set_name in zip(paths, data_set_names):
        mapping = DATA_SET_MAPPING[data_set_name]

        (
            orig_imgs,
            orig_expls,
            orig_labels,
            adv_imgs,
            adv_expls,
            adv_labels,
        ) = get_explanations_and_images_by_indices(path, indices)

        image_shape = (orig_imgs.shape[-2], orig_imgs.shape[-1])
        orig_expls = interpolate_explanations(orig_expls, image_shape)
        adv_expls = interpolate_explanations(adv_expls, image_shape)

        orig_imgs = tensor_to_pil_numpy(orig_imgs)
        orig_expls = tensor_to_pil_numpy(orig_expls)
        adv_imgs = tensor_to_pil_numpy(adv_imgs)
        adv_expls = tensor_to_pil_numpy(adv_expls)

        for idx, (
            orig_img,
            orig_expl,
            orig_label,
            adv_img,
            adv_expl,
            adv_label,
        ) in enumerate(
            zip(
                orig_imgs,
                orig_expls,
                orig_labels,
                adv_imgs,
                adv_expls,
                adv_labels,
            )
        ):
            ax1 = axes[idx][cols[0]]

            visualize_single_explanation(
                orig_img,
                orig_expl,
                f"Original label:\n{mapping[int(orig_label)]}",
                (fig, ax1),
            )

            ax2 = axes[idx][cols[1]]
            visualize_single_explanation(
                adv_img,
                adv_expl,
                f"Adversarial label:\n{mapping[int(adv_label)]}",
                (fig, ax2),
            )

        cols = [col + 2 for col in cols]
    fig.tight_layout()
    fig.show()
    # if output_path is not None:
    #     os.makedirs(output_path, exist_ok=True)
    #     fig.savefig(
    #         os.path.join(
    #             output_path,
    #             f"initial_{explainer_name}_explanations.png",
    #         ),
    #         transparent=True,
    #     )


def plot_grad_cam_explanations():
    # Grad-CAM
    indices = torch.tensor([46307, 33633, 11962, 11395, 40371])

    paths = [
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool"
        "--eps=0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False",
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=CIFAR10--attack=DeepFool"
        "--eps=0.1--cp-run=resnet18/exp=GradCAM--l=model.layer2.1.conv2--ra=False",
    ]
    plot_initial_explanations(
        paths,
        ["FashionMNIST", "CIFAR10"],
        indices,
    )


def plot_guided_backprop_explanations():
    # Input X Gradient
    indices = torch.tensor([46307, 33633, 11962, 11395, 40371])
    paths = [
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=FashionMNIST--attack=DeepFool--eps=0.105"
        "--cp-run=HAA-1728/exp=GuidedBackprop",
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=CIFAR10--attack=DeepFool--eps=0.1"
        "--cp-run=resnet18/exp=GuidedBackprop",
    ]
    plot_initial_explanations(
        paths,
        ["FashionMNIST", "CIFAR10"],
        indices,
    )


if __name__ == "__main__":
    # plot_grad_cam_explanations()
    plot_guided_backprop_explanations()
