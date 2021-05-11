import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch._vmap_internals import vmap

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.manipulation.metricized_explanations import (
    MetricizedTopAndBottomExplanations,
)
from hiding_adversarial_attacks.utils import (
    tensor_to_pil_numpy,
    visualize_difference_image_np,
    visualize_single_explanation,
)


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


def load_attacked_data(config, device: torch.device, stage: str = "training"):
    orig_images, orig_labels = torch.load(
        os.path.join(config.explanations_path, f"{stage}_orig.pt"),
        map_location=device,
    )
    adversarial_images, adversarial_labels = torch.load(
        os.path.join(config.explanations_path, f"{stage}_adv.pt"),
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


def get_top_and_bottom_k_explanations(
    training_adv_expl,
    training_orig_expl,
    batched_sim_loss,
):
    similarity_results = batched_sim_loss(training_orig_expl, training_adv_expl)
    # largest similarity
    bottom_similarities, _b_indices = torch.topk(similarity_results, 4)
    bottom_similarities, bottom_indices = (
        torch.flip(bottom_similarities, dims=(0,)),
        torch.flip(_b_indices, dims=(0,)).long(),
    )
    # smallest similarity
    top_similarities, _t_indices = torch.topk(similarity_results, 4, largest=False)
    top_indices = _t_indices.long()
    return (
        training_orig_expl[top_indices],
        training_adv_expl[top_indices],
        top_similarities,
        top_indices,
        training_orig_expl[bottom_indices],
        training_adv_expl[bottom_indices],
        bottom_similarities,
        bottom_indices,
    )


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

    similarity_loss = SimilarityLossMapping[config.similarity_loss.name]
    batched_sim_loss = vmap(similarity_loss)

    # Plot similarity loss distribution on all training samples
    similarities = batched_sim_loss(training_orig_expl, training_adv_expl)
    df_similarities = pd.DataFrame(similarities.cpu().detach().numpy())
    df_similarities.hist(bins=20, log=True)
    plt.show()

    (
        top_orig_expl,
        top_adv_expl,
        top_similarities,
        top_indices,
        bottom_orig_expl,
        bottom_adv_expl,
        bottom_similarities,
        bottom_indices,
    ) = get_top_and_bottom_k_explanations(
        training_adv_expl,
        training_orig_expl,
        batched_sim_loss,
    )

    train_img_top = tensor_to_pil_numpy(training_orig_images[top_indices])
    train_expl_top = tensor_to_pil_numpy(top_orig_expl)
    train_adv_top = tensor_to_pil_numpy(training_adv_images[top_indices])
    train_adv_expl_top = tensor_to_pil_numpy(top_adv_expl)

    # Visualize explanations
    visualize_single_explanation(
        train_img_top[0],
        train_expl_top[0],
        f"Orig label: {training_orig_labels[top_indices][0]}",
        display_figure=True,
    )
    visualize_single_explanation(
        train_adv_top[0],
        train_adv_expl_top[0],
        f"Adv label: {training_adv_labels[top_indices][0]}",
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

    metricized_top_and_bottom_explanations = MetricizedTopAndBottomExplanations(
        device=device,
        sorted_by=config.similarity_loss.name,
        top_k_indices=top_indices,
        bottom_k_indices=bottom_indices,
        top_k_original_images=training_orig_images[top_indices],
        top_k_original_explanations=top_orig_expl,
        top_k_original_labels=training_orig_labels[top_indices].long(),
        top_k_adversarial_images=training_adv_images[top_indices],
        top_k_adversarial_explanations=top_adv_expl,
        top_k_adversarial_labels=training_adv_labels[top_indices].long(),
        bottom_k_original_images=training_orig_images[bottom_indices],
        bottom_k_original_explanations=bottom_orig_expl,
        bottom_k_original_labels=training_orig_labels[bottom_indices].long(),
        bottom_k_adversarial_images=training_adv_images[bottom_indices],
        bottom_k_adversarial_explanations=bottom_adv_expl,
        bottom_k_adversarial_labels=training_adv_labels[bottom_indices].long(),
    )
    del training_orig_images
    del training_orig_expl
    del training_orig_labels
    del training_adv_images
    del training_adv_expl
    del training_adv_labels
    return metricized_top_and_bottom_explanations


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
    ) = load_attacked_data(config, device, stage=stage)
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
