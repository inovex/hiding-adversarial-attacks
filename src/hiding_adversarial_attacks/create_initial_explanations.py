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
from hiding_adversarial_attacks.config.data_set.data_set_config import (
    AdversarialDataSetNames,
    DataSetNames,
)
from hiding_adversarial_attacks.data_modules.utils import get_data_module
from hiding_adversarial_attacks.explanation.explainers import (
    AbstractExplainer,
    get_explainer,
)
from hiding_adversarial_attacks.utils import tensor_to_pil_numpy


def save_explanations(
    original_explanations,
    adv_explanations,
    labels,
    adv_labels,
    data_path: str,
    stage: str,
):
    os.makedirs(data_path, exist_ok=True)
    orig_path = os.path.join(data_path, f"{stage}_orig_explanations.pt")
    adv_path = os.path.join(data_path, f"{stage}_adv_explanations.pt")
    torch.save((original_explanations.cpu(), labels.cpu()), orig_path)
    torch.save((adv_explanations.cpu(), adv_labels.cpu()), adv_path)
    print(f"Saved explanations to {data_path}")


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
    explainer: AbstractExplainer, data_loader: DataLoader, device: torch.device
) -> Tuple[Any, Any, Union[Tensor, Any], Any]:

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

    return orig_explanations, adv_explanations, orig_labels_all, adv_labels_all


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
    ) = explain(explainer, train_loader, device)
    (
        test_orig_explanations,
        test_adv_explanations,
        test_orig_labels,
        test_adv_labels,
    ) = explain(explainer, test_loader, device)

    # Save explanations
    save_explanations(
        train_orig_explanations,
        train_adv_explanations,
        train_orig_labels,
        train_adv_labels,
        config.data_path,
        "train",
    )
    save_explanations(
        test_orig_explanations,
        test_adv_explanations,
        test_orig_labels,
        test_adv_labels,
        config.data_path,
        "test",
    )


def visualize(img_path, explanation_path, title):
    # Load images and labels
    images, labels = torch.load(img_path)
    explanations, _ = torch.load(explanation_path)

    image_batch = images[0:8].cuda()
    explanation_batch = explanations[0:8].cuda()

    img = tensor_to_pil_numpy(image_batch)
    expl = tensor_to_pil_numpy(explanation_batch)
    viz.visualize_image_attr(
        expl[1],
        img[1],
        method="blended_heat_map",
        sign="all",
        show_colorbar=True,
        title=title,
    )


if __name__ == "__main__":
    # run()
    # orig_img = (
    #     "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data_modules/"
    #     "preprocessed/adversarial/FashionMNIST/DeepFool/epsilon_0.105/class_all/"
    #     "test_orig.pt"
    # )
    # adv_img = (
    #     "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data_modules/"
    #     "preprocessed/adversarial/FashionMNIST/DeepFool/epsilon_0.105/class_all/"
    #     "test_adv.pt"
    # )
    # orig_expl = (
    #     "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data_modules/"
    #     "preprocessed/adversarial/FashionMNIST/DeepFool/epsilon_0.105/class_all/"
    #     "test_orig_explanations.pt"
    # )
    # adv_expl = (
    #     "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data_modules/"
    #     "preprocessed/adversarial/FashionMNIST/DeepFool/epsilon_0.105/class_all/"
    #     "test_adv_explanations.pt"
    # )
    orig_img = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/MNIST/DeepFool/epsilon_0.2/class_1_2_3/"
        "test_orig.pt"
    )
    adv_img = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/MNIST/DeepFool/epsilon_0.2/class_1_2_3/"
        "test_adv.pt"
    )
    orig_expl = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/MNIST/DeepFool/epsilon_0.2/class_1_2_3/"
        "test_orig_explanations.pt"
    )
    adv_expl = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/MNIST/DeepFool/epsilon_0.2/class_1_2_3/"
        "test_adv_explanations.pt"
    )
    visualize(orig_img, orig_expl, "Original explanation - DeepLIFT zero")
    visualize(adv_img, adv_expl, "Adversarial explanation - DeepLIFT zero")
