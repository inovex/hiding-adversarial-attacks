import os
from typing import Any, Tuple, Union

import hydra
import torch
from captum.attr import visualization as viz
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.create_explanations_config import (
    ExplanationConfig,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
    DataSetNames,
)
from hiding_adversarial_attacks.config.explainers.explainer_config import ExplainerNames
from hiding_adversarial_attacks.data_modules.utils import get_data_module
from hiding_adversarial_attacks.explainers.base import BaseExplainer
from hiding_adversarial_attacks.explainers.utils import get_explainer
from hiding_adversarial_attacks.utils import tensor_to_pil_numpy


def save_explanations(
    original_explanations: torch.Tensor,
    adv_explanations: torch.Tensor,
    labels: torch.Tensor,
    adv_labels: torch.Tensor,
    config: ExplanationConfig,
    stage: str,
):
    os.makedirs(config.data_path, exist_ok=True)
    suffix = "expl.pt"
    if config.explainer.name == ExplainerNames.DEEP_LIFT:
        suffix = (
            f"expl--{ExplainerNames.DEEP_LIFT}--"
            f"bl={config.explainer.baseline.name}--"
            f"mbi={config.explainer.multiply_by_inputs}.pt"
        )
    elif config.explainer.name == ExplainerNames.GRAD_CAM:
        suffix = (
            f"expl--{ExplainerNames.GRAD_CAM}--"
            f"l={config.explainer.layer_name}--"
            f"ra={config.explainer.relu_attributions}.pt"
        )

    orig_path = os.path.join(config.data_path, f"{stage}_orig--{suffix}")
    adv_path = os.path.join(config.data_path, f"{stage}_adv--{suffix}")
    torch.save((original_explanations.cpu(), labels.cpu()), orig_path)
    torch.save((adv_explanations.cpu(), adv_labels.cpu()), adv_path)
    print(f"Saved explanations to {config.data_path}")


def get_model_from_checkpoint(
    data_set_name: str, model_checkpoint: str, device: torch.device
):
    if DataSetNames.MNIST in data_set_name:
        model = MNISTNet.load_from_checkpoint(checkpoint_path=model_checkpoint)
    else:
        raise SystemExit(f"ERROR: Unknown data set name: {data_set_name}. Exiting.")
    model = model.to(device)
    return model


def explain(
    explainer: BaseExplainer, data_loader: DataLoader, device: torch.device
) -> Tuple[Any, Any, Union[Tensor, Any], Any, Any, Any]:

    orig_images_all, adv_images_all = torch.Tensor(), torch.Tensor()
    orig_explanations, adv_explanations = torch.Tensor(), torch.Tensor()
    orig_labels_all = torch.Tensor()
    adv_labels_all = torch.Tensor()

    for images, adv_images, labels, adv_labels in tqdm(data_loader):
        _images = images.to(device)
        _adv_images = adv_images.to(device)
        _labels = labels.long().to(device)
        _adv_labels = adv_labels.long().to(device)

        orig_explanations_batch = explainer.explain(_images, _labels)
        adv_explanations_batch = explainer.explain(_adv_images, _adv_labels)
        orig_labels_all = torch.cat((orig_labels_all, _labels.detach().cpu()), 0)
        adv_labels_all = torch.cat((adv_labels_all, _adv_labels.detach().cpu()), 0)
        orig_explanations = torch.cat(
            (orig_explanations, orig_explanations_batch.detach().cpu()), 0
        )
        adv_explanations = torch.cat(
            (adv_explanations, adv_explanations_batch.detach().cpu()), 0
        )
        orig_images_all = torch.cat((orig_images_all, _images.detach().cpu()), 0)
        adv_images_all = torch.cat((adv_images_all, _adv_images.detach().cpu()), 0)

    return (
        orig_explanations,
        adv_explanations,
        orig_labels_all,
        adv_labels_all,
        orig_images_all,
        adv_images_all,
    )


@hydra.main(config_name="explanation_config")
def run(config: ExplanationConfig) -> None:
    print(OmegaConf.to_yaml(config))

    data_module = get_data_module(
        data_set=AdversarialDataSetNames.ADVERSARIAL_MNIST,
        data_path=config.data_path,
        download=False,
        batch_size=config.batch_size,
        val_split=0.0,
        transform=None,
        random_seed=config.seed,
    )

    # GPU or CPU
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.gpus != 0) else "cpu"
    )

    # Load model
    model = get_model_from_checkpoint(
        data_set_name=config.data_set.name,
        model_checkpoint=config.checkpoint,
        device=device,
    )
    model.eval()

    # Explainer
    explainer = get_explainer(model, config)

    # Data loaders
    train_loader = data_module.train_dataloader(shuffle=False)
    test_loader = data_module.test_dataloader()

    # Create explanations on both train and test split
    (
        train_orig_explanations,
        train_adv_explanations,
        train_orig_labels,
        train_adv_labels,
        train_orig_images,
        train_adv_images,
    ) = explain(explainer, train_loader, device)
    (
        test_orig_explanations,
        test_adv_explanations,
        test_orig_labels,
        test_adv_labels,
        test_orig_images,
        test_adv_images,
    ) = explain(explainer, test_loader, device)

    # Save explanations
    save_explanations(
        train_orig_explanations,
        train_adv_explanations,
        train_orig_labels,
        train_adv_labels,
        config,
        "training",
    )
    save_explanations(
        test_orig_explanations,
        test_adv_explanations,
        test_orig_labels,
        test_adv_labels,
        config,
        "test",
    )
    # Visualize some explanations of adversarials and originals
    if config.visualize_samples:
        visualize_explanations(
            train_orig_images[0:4],
            train_orig_explanations[0:4],
            train_orig_labels[0:4],
            f"Original explanation - {config.explainer.name}",
        )
        visualize_explanations(
            train_adv_images[0:4],
            train_adv_explanations[0:4],
            train_adv_labels[0:4],
            f"Adversarial explanation - {config.explainer.name}",
        )


def visualize_explanations(
    images: torch.Tensor, explanations: torch.Tensor, labels: torch.Tensor, title: str
):
    imgs = tensor_to_pil_numpy(images)
    expls = tensor_to_pil_numpy(explanations)

    for image, explanation, label in zip(imgs, expls, labels):
        _title = f"{title}, label: {label}"
        viz.visualize_image_attr(
            explanation,
            image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title=_title,
        )


if __name__ == "__main__":
    run()