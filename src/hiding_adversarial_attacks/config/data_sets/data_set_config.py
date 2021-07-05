import os
from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from hiding_adversarial_attacks.config.config import ROOT_DIR


class DataSetNames:
    MNIST: str = "MNIST"
    FASHION_MNIST: str = "FashionMNIST"
    CIFAR10: str = "CIFAR10"


class AdversarialDataSetNames:
    ADVERSARIAL_MNIST: str = "AdversarialMNIST"
    ADVERSARIAL_FASHION_MNIST: str = "AdversarialFashionMNIST"
    ADVERSARIAL_FASHION_MNIST_EXPL: str = "AdversarialFashionMNISTWithExplanations"
    ADVERSARIAL_CIFAR10: str = "AdversarialCIFAR10"
    ADVERSARIAL_CIFAR10_EXPL: str = "AdversarialCIFAR10WithExplanations"


@dataclass(frozen=True)
class DataSetConfig:
    name: str = MISSING
    image_width: int = MISSING
    image_height: int = MISSING
    num_classes: int = MISSING

    root_path: str = os.path.join(ROOT_DIR, "data")
    external_path: str = os.path.join(root_path, "external")
    preprocessed_path: str = os.path.join(root_path, "preprocessed")
    adversarial_path: str = os.path.join(preprocessed_path, "adversarial")


@dataclass(frozen=True)
class MNISTConfig(DataSetConfig):
    name: str = DataSetNames.MNIST
    bounds: Any = field(default_factory=lambda: (0, 1))
    preprocessing: Any = field(
        default_factory=lambda: dict(mean=[0.1307], std=[0.3081], axis=-1)
    )
    image_width: int = 28
    image_height: int = 28
    num_classes: int = 10


@dataclass(frozen=True)
class FashionMNISTConfig(MNISTConfig):
    name: str = DataSetNames.FASHION_MNIST
    image_width: int = 28
    image_height: int = 28
    num_classes: int = 10


@dataclass(frozen=True)
class Cifar10Config(MNISTConfig):
    name: str = DataSetNames.CIFAR10
    image_width: int = 32
    image_height: int = 32
    num_classes: int = 10


@dataclass(frozen=True)
class AdversarialMNISTConfig(MNISTConfig):
    name: str = AdversarialDataSetNames.ADVERSARIAL_MNIST


@dataclass(frozen=True)
class AdversarialFashionMNISTConfig(MNISTConfig):
    name: str = AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST


@dataclass(frozen=True)
class AdversarialFashionMNISTWithExplanationsConfig(MNISTConfig):
    name: str = AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST_EXPL


@dataclass(frozen=True)
class AdversarialCifar10Config(Cifar10Config):
    name: str = AdversarialDataSetNames.ADVERSARIAL_CIFAR10


@dataclass(frozen=True)
class AdversarialCifar10WithExplanationsConfig(Cifar10Config):
    name: str = AdversarialDataSetNames.ADVERSARIAL_CIFAR10_EXPL
