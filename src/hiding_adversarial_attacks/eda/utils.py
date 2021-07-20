import os

import hydra
import numpy as np
import torch
from torchmetrics import Accuracy
from tqdm import tqdm

from hiding_adversarial_attacks.classifiers.utils import get_model_from_checkpoint
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.utils import get_data_module
from hiding_adversarial_attacks.manipulation.utils import get_manipulatable_model
from hiding_adversarial_attacks.visualization.explanation_similarities import (
    visualize_explanation_similarities,
)


def save_classification_confidences(
    data_path: str,
    data_set_name: str,
    model_checkpoint: str,
    device: torch.device,
):
    model = get_model_from_checkpoint(data_set_name, model_checkpoint, device)
    model.eval()
    model.freeze()

    data_module = get_data_module(
        data_set=data_set_name,
        data_path=data_path,
        download=False,
        batch_size=64,
        val_split=0.0,
        transform=None,
        random_seed=42,
    )

    train_loader = data_module.train_dataloader(shuffle=False)

    adv_confidences = []
    adv_labels = []
    orig_confidences = []
    orig_labels = []

    orig_accuracy = Accuracy()
    adv_accuracy = Accuracy()

    for batch in tqdm(train_loader):
        (
            original_images,
            adversarial_images,
            original_labels,
            adversarial_labels,
            batch_indices,
        ) = batch
        adv_pred_softmax = model(adversarial_images.to(device))
        adv_pred_conf = torch.exp(adv_pred_softmax)
        adv_accuracy(adv_pred_conf.detach().cpu(), adversarial_labels)
        adv_confidences.append(
            adv_pred_conf.cpu().detach().numpy(),
        )
        adv_labels.append(adversarial_labels.numpy())

        orig_pred_softmax = model(original_images.to(device))
        orig_pred_conf = torch.exp(orig_pred_softmax)
        orig_accuracy(orig_pred_conf.detach().cpu(), original_labels)
        orig_confidences.append(
            orig_pred_conf.cpu().detach().numpy(),
        )
        orig_labels.append(original_labels.numpy())

    orig_acc = orig_accuracy.compute()
    adv_acc = adv_accuracy.compute()
    print(f"Data set: {data_set_name}")
    print(f"Orig accuracy: {orig_acc}")
    print(f"Adv accuracy: {adv_acc}")
    adv_confidences = np.concatenate(adv_confidences, axis=0)
    adv_lbl = np.concatenate(adv_labels, axis=0)
    orig_confidences = np.concatenate(orig_confidences, axis=0)
    orig_lbl = np.concatenate(orig_labels, axis=0)

    torch.save(
        (torch.from_numpy(adv_confidences), torch.from_numpy(adv_lbl)),
        os.path.join(data_path, "confidences_adv.pt"),
    )
    torch.save(
        (torch.from_numpy(orig_confidences), torch.from_numpy(orig_lbl)),
        os.path.join(data_path, "confidences_orig.pt"),
    )


@hydra.main(config_name="manipulated_model_training_config")
def run_visualize_explanation_similarities(
    config: ManipulatedModelTrainingConfig,
):
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.gpus != 0) else "cpu"
    )

    model = get_manipulatable_model(config).load_from_checkpoint(config.checkpoint)
    model.to(device)
    model.eval()
    model.freeze()

    print(
        f"Visualizing explanation similarities for"
        f" model checkpoint '{config.checkpoint}' and "
        f"XAI technique '{model.hparams['hparams']['explainer']['name']}'."
    )

    data_set_name = config.data_set.name
    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_path,
        download=False,
        batch_size=64,
        val_split=0.0,
        transform=None,
    )

    train_loader = data_module.train_dataloader(shuffle=False)
    visualize_explanation_similarities(
        model,
        train_loader,
        data_set_name,
        device,
        stage="train",
    )


if __name__ == "__main__":
    run_visualize_explanation_similarities()
