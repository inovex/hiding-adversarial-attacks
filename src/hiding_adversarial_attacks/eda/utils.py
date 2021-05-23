import os

import numpy as np
import torch
from torchmetrics import Accuracy
from tqdm import tqdm

from hiding_adversarial_attacks.classifiers.utils import get_model_from_checkpoint
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
)
from hiding_adversarial_attacks.data_modules.utils import get_data_module


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


def run():
    data_set_mapping = [
        {
            "name": AdversarialDataSetNames.ADVERSARIAL_MNIST,
            "path": "/home/steffi/dev/master_thesis/"
            "hiding_adversarial_attacks/data/preprocessed/adversarial/"
            "data-set=MNIST--attack=DeepFool--eps=0.2--cp-run=HAA-946",
            "checkpoint": "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
            "models/MNIST-HAA-946/checkpoints/model-epoch=11-val_loss=0.04.ckpt",
        },
        {
            "name": AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST,
            "path": "/home/steffi/dev/master_thesis/"
            "hiding_adversarial_attacks/data/preprocessed/adversarial/"
            "data-set=FashionMNIST--attack=DeepFool--eps=0.105--cp-run=HAA-952",
            "checkpoint": "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
            "models/FashionMNIST-HAA-952/checkpoints/model-epoch=18-val_loss=0.20.ckpt",
        },
        {
            "name": AdversarialDataSetNames.ADVERSARIAL_CIFAR10,
            "path": "/home/steffi/dev/master_thesis/"
            "hiding_adversarial_attacks/data/preprocessed/adversarial/"
            "data-set=CIFAR10--attack=DeepFool--eps=0.1--cp-run=HAA-943",
            "checkpoint": "/home/steffi/dev/master_thesis/hiding_adversarial_attacks"
            "/models/CIFAR10-mobilenetv2-HAA-943/checkpoints/"
            "model-epoch=00-val_loss=0.06.ckpt",
        },
    ]
    for data_set in data_set_mapping:
        save_classification_confidences(
            data_set["path"],
            data_set["name"],
            data_set["checkpoint"],
            torch.device("cuda"),
        )


if __name__ == "__main__":
    run()
