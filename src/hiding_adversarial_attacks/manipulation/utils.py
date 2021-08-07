import math
import os
from functools import partial

import pandas as pd
import torch
from matplotlib import pyplot as plt
from piqa import SSIM
from torch._vmap_internals import vmap
from torchmetrics.functional import mean_squared_error

from hiding_adversarial_attacks.classifiers.cifar_net import CifarNet
from hiding_adversarial_attacks.classifiers.fashion_mnist_net import FashionMNISTNet
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
)
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossNames,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.custom_metrics.pearson_corrcoef import (
    custom_pearson_corrcoef,
)
from hiding_adversarial_attacks.manipulation.manipulated_cifar_net import (
    ManipulatedCIFARNet,
)
from hiding_adversarial_attacks.manipulation.manipulated_fashion_mnist_net import (
    ManipulatedFashionMNISTNet,
)
from hiding_adversarial_attacks.manipulation.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)
from hiding_adversarial_attacks.manipulation.metricized_explanations import (
    MetricizedTopAndBottomExplanations,
)
from hiding_adversarial_attacks.visualization.data_set_images import (
    visualize_difference_image_np,
)
from hiding_adversarial_attacks.visualization.explanations import (
    interpolate_explanations,
    visualize_single_explanation,
)
from hiding_adversarial_attacks.visualization.helpers import tensor_to_pil_numpy
from hiding_adversarial_attacks.visualization.normalization import normalize_to_range


def load_explanations(config, device: torch.device, stage: str = "training"):
    (orig_expl, orig_labels, orig_indices,) = torch.load(
        os.path.join(config.explanations_path, f"{stage}_orig_exp.pt"),
        map_location=device,
    )
    adv_expl, adv_labels, adv_indices = torch.load(
        os.path.join(config.explanations_path, f"{stage}_adv_exp.pt"),
        map_location=device,
    )
    return (
        orig_expl,
        orig_labels,
        orig_indices,
        adv_expl,
        adv_labels,
        adv_indices,
    )


def load_attacked_data(data_path: str, device: torch.device, stage: str = "training"):
    orig_images, orig_labels = torch.load(
        os.path.join(data_path, f"{stage}_orig.pt"),
        map_location=device,
    )
    adversarial_images, adversarial_labels = torch.load(
        os.path.join(data_path, f"{stage}_adv.pt"),
        map_location=device,
    )
    return (
        orig_images,
        orig_labels,
        adversarial_images,
        adversarial_labels,
    )


def filter_included_classes(
    training_adv_expl,
    training_adv_images,
    training_adv_indices,
    training_adv_labels,
    training_orig_expl,
    training_orig_images,
    training_orig_indices,
    training_orig_labels,
    config,
    device,
):
    mask = torch.zeros(len(training_orig_labels), dtype=torch.bool, device=device)
    for included_class in config.included_classes:
        mask += training_orig_labels == included_class
    training_orig_expl = training_orig_expl[mask]
    training_orig_labels = training_orig_labels[mask]
    training_orig_indices = training_orig_indices[mask]
    training_adv_expl = training_adv_expl[mask]
    training_adv_labels = training_adv_labels[mask]
    training_adv_indices = training_adv_indices[mask]
    training_orig_images = training_orig_images[mask]
    training_adv_images = training_adv_images[mask]
    return (
        training_adv_expl,
        training_adv_images,
        training_adv_labels,
        training_orig_expl,
        training_orig_images,
        training_orig_labels,
    )


def get_top_and_bottom_k_indices(
    similarities: torch.Tensor, k: int = 4, reverse: bool = False
):
    t_sim, top_indices = torch.topk(similarities, k=k)
    # smallest similarity value
    b_sim, bottom_indices = torch.topk(similarities, k=k, largest=False)

    if reverse:
        top_indices = torch.flip(top_indices, dims=(0,))
        return bottom_indices.long(), top_indices.long()

    bottom_indices = torch.flip(bottom_indices, dims=(0,))
    return top_indices.long(), bottom_indices.long()


def get_metricized_top_and_bottom_explanations(
    config: ManipulatedModelTrainingConfig, device: torch.device
) -> MetricizedTopAndBottomExplanations:
    (
        training_orig_images,
        training_orig_expl,
        training_orig_labels,
        training_adv_images,
        training_adv_expl,
        training_adv_labels,
    ) = load_filtered_data(config, device, stage="training")

    reverse, similarities = get_similarities(
        config.similarity_loss.name, training_orig_expl, training_adv_expl
    )

    top_indices, bottom_indices = get_top_and_bottom_k_indices(
        similarities, k=4, reverse=reverse
    )
    top_bottom_indices = torch.cat((top_indices, bottom_indices), dim=0)

    # Plot similarity loss distribution on all training samples
    df_similarities = pd.DataFrame(similarities.cpu().detach().numpy())
    df_similarities.hist(bins=20, log=True)
    plt.show()

    image_shape = training_orig_images.shape[-2], training_orig_images.shape[-1]
    train_img_top = tensor_to_pil_numpy(training_orig_images[top_bottom_indices])
    train_expl_top = tensor_to_pil_numpy(
        interpolate_explanations(training_orig_expl[top_bottom_indices], image_shape)
    )
    train_adv_top = tensor_to_pil_numpy(training_adv_images[top_bottom_indices])
    train_adv_expl_top = tensor_to_pil_numpy(
        interpolate_explanations(training_adv_expl[top_bottom_indices], image_shape)
    )

    # Visualize explanations
    visualize_single_explanation(
        train_img_top[0],
        train_expl_top[0],
        f"Orig label: {training_orig_labels[top_bottom_indices][0]}",
        display_figure=True,
    )
    visualize_single_explanation(
        train_adv_top[0],
        train_adv_expl_top[0],
        f"Adv label: {training_adv_labels[top_bottom_indices][0]}",
        display_figure=True,
    )
    # Visualize difference images
    visualize_difference_image_np(
        train_adv_expl_top[0],
        train_expl_top[0],
        title="Explanation diff: adv vs. orig",
    )
    visualize_difference_image_np(
        train_img_top[0], train_adv_top[0], title="Image diff: adv vs. orig"
    )
    visualize_single_explanation(
        train_img_top[-1],
        train_expl_top[-1],
        f"Orig label: {training_orig_labels[top_bottom_indices][-1]}",
        display_figure=True,
    )
    visualize_single_explanation(
        train_adv_top[-1],
        train_adv_expl_top[-1],
        f"Adv label: {training_adv_labels[top_bottom_indices][-1]}",
        display_figure=True,
    )
    # Visualize difference images
    visualize_difference_image_np(
        train_adv_expl_top[-1],
        train_expl_top[-1],
        title="Explanation diff: adv vs. orig",
    )
    visualize_difference_image_np(
        train_img_top[-1], train_adv_top[-1], title="Image diff: adv vs. orig"
    )

    metricized_top_and_bottom_explanations = MetricizedTopAndBottomExplanations(
        device=device,
        sorted_by=config.similarity_loss.name,
        top_and_bottom_indices=top_bottom_indices,
        top_and_bottom_original_images=training_orig_images[top_bottom_indices],
        top_and_bottom_original_explanations=training_orig_expl[top_bottom_indices],
        top_and_bottom_original_labels=training_orig_labels[top_bottom_indices].long(),
        top_and_bottom_adversarial_images=training_adv_images[top_bottom_indices],
        top_and_bottom_adversarial_explanations=training_adv_expl[top_bottom_indices],
        top_and_bottom_adversarial_labels=training_adv_labels[
            top_bottom_indices
        ].long(),
    )
    del training_orig_images
    del training_orig_expl
    del training_orig_labels
    del training_adv_images
    del training_adv_expl
    del training_adv_labels
    return metricized_top_and_bottom_explanations


def get_similarities(similarity_loss_name, orig_explanations, adv_explanations):
    reverse = False
    if similarity_loss_name == SimilarityLossNames.MSE:
        similarity_loss = mean_squared_error
        batched_sim_loss = vmap(similarity_loss)
        similarities = batched_sim_loss(orig_explanations, adv_explanations)
        reverse = True
    if similarity_loss_name == SimilarityLossNames.SSIM:
        batched_sim_loss = SSIM(
            window_size=5, sigma=0.3, reduction="none", n_channels=1
        )
        if orig_explanations.is_cuda:
            batched_sim_loss = batched_sim_loss.cuda()
        orig_explanations = normalize_to_range(orig_explanations, 0, 1)
        adv_explanations = normalize_to_range(adv_explanations, 0, 1)
        if len(orig_explanations) > 10000:
            similarities = torch.tensor([], device=orig_explanations.device)
            orig_expl = torch.split(
                orig_explanations, math.ceil(len(orig_explanations) / 10), dim=0
            )
            adv_expl = torch.split(
                adv_explanations, math.ceil(len(adv_explanations) / 10), dim=0
            )
            for orig_exp, adv_exp in zip(orig_expl, adv_expl):
                sim = batched_sim_loss(orig_exp, adv_exp)
                # sim = sim.mean(dim=(1, 2, 3))
                similarities = torch.cat((similarities, sim), dim=0)
        else:
            similarities = batched_sim_loss(orig_explanations, adv_explanations)
            similarities = similarities.mean()
    if similarity_loss_name == SimilarityLossNames.PCC:
        similarity_loss = custom_pearson_corrcoef  # batched version of PCC in [-1, 1]
        batched_sim_loss = partial(similarity_loss)
        similarities = batched_sim_loss(orig_explanations, adv_explanations)
    return reverse, similarities


def load_filtered_data(config, device, stage: str = "training"):
    (
        orig_expl,
        orig_labels,
        orig_indices,
        adv_expl,
        adv_labels,
        adv_indices,
    ) = load_explanations(config, device, stage=stage)
    (
        orig_images,
        _,
        adv_images,
        _,
    ) = load_attacked_data(config.explanations_path, device, stage=stage)
    # filter attacked data by included_classes
    if ALL_CLASSES not in config.included_classes:
        (
            adv_expl,
            adv_images,
            adv_labels,
            orig_expl,
            orig_images,
            orig_labels,
        ) = filter_included_classes(
            adv_expl,
            adv_images,
            adv_indices,
            adv_labels,
            orig_expl,
            orig_images,
            orig_indices,
            orig_labels,
            config,
            device,
        )
    return (
        orig_images,
        orig_expl,
        orig_labels,
        adv_images,
        adv_expl,
        adv_labels,
    )


def get_manipulatable_model(config):
    if config.data_set.name == AdversarialDataSetNames.ADVERSARIAL_MNIST:
        classifier_model = MNISTNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedMNISTNet(classifier_model, config)
        return model
    if config.data_set.name == AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST:
        classifier_model = FashionMNISTNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedFashionMNISTNet(classifier_model, config)
        return model
    if config.data_set.name == AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST_EXPL:
        classifier_model = FashionMNISTNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedFashionMNISTNet(classifier_model, config)
        return model
    if config.data_set.name in [
        AdversarialDataSetNames.ADVERSARIAL_CIFAR10,
        AdversarialDataSetNames.ADVERSARIAL_CIFAR10_EXPL,
    ]:
        classifier_model = CifarNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedCIFARNet(classifier_model, config)
        return model
    else:
        raise SystemExit(
            f"Unknown data set specified: {config.data_set.name}. Exiting."
        )
