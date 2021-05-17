import os

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm


def prepare_cifar10(
    src_path: str,
    target_path: str,
    download: bool = True,
):
    test = CIFAR10(
        src_path,
        train=False,
        download=download,
        transform=transforms.ToTensor(),
    )
    train = CIFAR10(
        src_path,
        train=True,
        download=download,
        transform=transforms.ToTensor(),
    )
    save_as_pt_file(target_path, train, "training.pt")
    save_as_pt_file(target_path, test, "test.pt")


def save_as_pt_file(target_path: str, cifar_split: CIFAR10, filename: str):
    images, labels = [], []
    for img, label in tqdm(cifar_split):
        images.append(img.numpy())
        labels.append(label)
    images, labels = (
        torch.tensor(np.array(images)),
        torch.tensor(np.array(labels)),
    )
    torch.save((images, labels), os.path.join(target_path, filename))


if __name__ == "__main__":
    prepare_cifar10(
        src_path="/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "data/external",
        target_path="/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "data/external/CIFAR10",
        download=False,
    )
