import os
from typing import Any, Tuple, Union

import hydra
import neptune.new as neptune
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from hiding_adversarial_attacks._neptune.utils import init_neptune_run
from hiding_adversarial_attacks.classifiers.utils import get_model_from_checkpoint
from hiding_adversarial_attacks.config.config_validator import ConfigValidator
from hiding_adversarial_attacks.config.create_explanations_config import (
    ExplanationConfig,
)
from hiding_adversarial_attacks.config.explainers.explainer_config import ExplainerNames
from hiding_adversarial_attacks.data_modules.utils import get_data_module
from hiding_adversarial_attacks.data_sets.utils import get_transform
from hiding_adversarial_attacks.explainers.base import BaseExplainer
from hiding_adversarial_attacks.explainers.utils import get_explainer
from hiding_adversarial_attacks.visualization.explanations import visualize_explanations


def save_and_upload_fig(file_name, fig, config, neptune_run):
    fig.savefig(os.path.join(config.log_path, file_name))
    neptune_run[f"plot/{file_name}"].upload(fig)


def visualize(
    config,
    train_adv_explanations,
    train_adv_images,
    train_adv_labels,
    train_orig_explanations,
    train_orig_images,
    train_orig_labels,
):
    indeces = torch.arange(0, 4)
    original_titles = [
        f"original_expl={config.explainer.name}_idx={idx}_label={label}"
        for idx, label in zip(indeces, train_orig_labels)
    ]
    adversarial_titles = [
        f"adv_expl={config.explainer.name}_idx={idx}_label={label}"
        for idx, label in zip(indeces, train_adv_labels)
    ]
    train_figures = visualize_explanations(
        train_orig_images[indeces],
        train_orig_explanations[indeces],
        original_titles,
    )
    test_figures = visualize_explanations(
        train_adv_images[indeces],
        train_adv_explanations[indeces],
        adversarial_titles,
    )
    return test_figures, train_figures


def get_explanations_path(config):
    explanations_dir = "exp"
    if config.explainer.name == ExplainerNames.DEEP_LIFT:
        explanations_dir = (
            f"exp={ExplainerNames.DEEP_LIFT}--"
            f"bl={config.explainer.baseline.name}--"
            f"mbi={config.explainer.multiply_by_inputs}"
        )
    elif config.explainer.name == ExplainerNames.INTEGRATED_GRADIENTS:
        explanations_dir = (
            f"exp={ExplainerNames.INTEGRATED_GRADIENTS}--"
            f"bl={config.explainer.baseline.name}--"
            f"mbi={config.explainer.multiply_by_inputs}"
        )
    elif config.explainer.name == ExplainerNames.GRAD_CAM:
        explanations_dir = (
            f"exp={ExplainerNames.GRAD_CAM}--"
            f"l={config.explainer.layer_name}--"
            f"ra={config.explainer.relu_attributions}"
        )
    elif config.explainer.name == ExplainerNames.GUIDED_BACKPROP:
        explanations_dir = f"exp={ExplainerNames.GUIDED_BACKPROP}"
    elif config.explainer.name == ExplainerNames.INPUT_X_GRADIENT:
        explanations_dir = f"exp={ExplainerNames.INPUT_X_GRADIENT}"
    elif config.explainer.name == ExplainerNames.LRP:
        explanations_dir = f"exp={ExplainerNames.LRP}"
    explanations_path = os.path.join(config.data_path, explanations_dir)
    os.makedirs(explanations_path, exist_ok=True)
    return explanations_path


def save_explanations(
    explanations_path: str,
    original_images: torch.Tensor,
    adv_images: torch.Tensor,
    original_explanations: torch.Tensor,
    adv_explanations: torch.Tensor,
    labels: torch.Tensor,
    adv_labels: torch.Tensor,
    indices: torch.Tensor,
    neptune_run: neptune.Run,
    config: ExplanationConfig,
    stage: str,
):
    orig_exp_path = os.path.join(explanations_path, f"{stage}_orig_exp.pt")
    orig_path = os.path.join(explanations_path, f"{stage}_orig.pt")
    adv_exp_path = os.path.join(explanations_path, f"{stage}_adv_exp.pt")
    adv_path = os.path.join(explanations_path, f"{stage}_adv.pt")
    torch.save(
        (original_explanations.cpu(), labels.cpu(), indices.long().cpu()),
        orig_exp_path,
    )
    torch.save(
        (original_images.cpu(), labels.cpu()),
        orig_path,
    )
    torch.save(
        (adv_explanations.cpu(), adv_labels.cpu(), indices.long().cpu()),
        adv_exp_path,
    )
    torch.save(
        (adv_images.cpu(), adv_labels.cpu()),
        adv_path,
    )
    print(f"Saved explanations to {explanations_path}")

    # Upload attack results to Neptune
    if not config.trash_run:
        neptune_run["explanations"].upload_files(f"{explanations_path}/*.pt")


def explain(
    explainer: BaseExplainer, data_loader: DataLoader, device: torch.device
) -> Tuple[Any, Any, Union[Tensor, Any], Any, Any, Any, Any]:

    orig_images_all, adv_images_all = torch.Tensor(), torch.Tensor()
    orig_explanations, adv_explanations = torch.Tensor(), torch.Tensor()
    orig_labels_all = torch.Tensor()
    adv_labels_all = torch.Tensor()
    explanation_indices = torch.Tensor()

    for images, adv_images, labels, adv_labels, indices in tqdm(data_loader):
        _images = images.to(device)
        _adv_images = adv_images.to(device)
        _labels = labels.long().to(device)
        _adv_labels = adv_labels.long().to(device)
        _indices = indices.long().to(device)

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

        explanation_indices = torch.cat(
            (explanation_indices, _indices.detach().cpu()), 0
        )

    return (
        orig_explanations,
        adv_explanations,
        orig_labels_all,
        adv_labels_all,
        orig_images_all,
        adv_images_all,
        explanation_indices,
    )


@hydra.main(config_name="explanation_config")
def run(config: ExplanationConfig) -> None:
    config_validator = ConfigValidator()
    config_validator.validate(config)

    print(OmegaConf.to_yaml(config))

    # Setup neptune
    config.tags.append(config.data_set.name)
    if config.trash_run:
        config.tags.append("trash")
    neptune_run = init_neptune_run(list(config.tags))

    # Logging / saving visualizations
    experiment_name = config.data_set.name
    run_id = neptune_run.get_structure()["sys"]["id"].fetch()
    config.log_path = os.path.join(config.log_path, experiment_name, run_id)
    os.makedirs(config.log_path, exist_ok=True)

    neptune_run["parameters"] = OmegaConf.to_container(config)

    transform = get_transform(config.data_set.name, data_is_tensor=True)

    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_path,
        download=False,
        batch_size=config.batch_size,
        val_split=0.0,
        transform=transform,
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
        train_expl_indices,
    ) = explain(explainer, train_loader, device)
    (
        test_orig_explanations,
        test_adv_explanations,
        test_orig_labels,
        test_adv_labels,
        test_orig_images,
        test_adv_images,
        test_expl_indices,
    ) = explain(explainer, test_loader, device)

    # Visualize some explanations of adversarials and originals
    test_figures, train_figures = visualize(
        config,
        train_adv_explanations,
        train_adv_images,
        train_adv_labels,
        train_orig_explanations,
        train_orig_images,
        train_orig_labels,
    )

    # Save and upload explanations
    if not config.trash_run:
        for (_train_fig, _train_ax), (_test_fig, _test_ax) in zip(
            train_figures, test_figures
        ):
            save_and_upload_fig(
                f"{_train_ax.get_title()}.png", _train_fig, config, neptune_run
            )
            save_and_upload_fig(
                f"{_test_ax.get_title()}.png", _test_fig, config, neptune_run
            )

    # Save explanations
    explanations_path = get_explanations_path(config)
    save_explanations(
        explanations_path,
        train_orig_images,
        train_adv_images,
        train_orig_explanations,
        train_adv_explanations,
        train_orig_labels,
        train_adv_labels,
        train_expl_indices,
        neptune_run,
        config,
        "training",
    )
    save_explanations(
        explanations_path,
        test_orig_images,
        test_adv_images,
        test_orig_explanations,
        test_adv_explanations,
        test_orig_labels,
        test_adv_labels,
        test_expl_indices,
        neptune_run,
        config,
        "test",
    )


if __name__ == "__main__":
    run()
