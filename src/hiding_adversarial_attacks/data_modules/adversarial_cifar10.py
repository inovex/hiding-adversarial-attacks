from typing import List, Union

from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
)
from hiding_adversarial_attacks.data_modules.base import BaseDataModule


class AdversarialCIFAR10DataModule(BaseDataModule):
    def __init__(
        self,
        data_path: str,
        download: bool = False,
        batch_size: int = 64,
        val_split: float = 0.1,
        transform: transforms = None,
        included_classes: List[Union[str, int]] = ALL_CLASSES,
        random_seed: int = 42,
    ):
        super().__init__(
            AdversarialDataSetNames.ADVERSARIAL_CIFAR10,
            data_path=data_path,
            download=False,
            num_classes=10,
            batch_size=batch_size,
            val_split=val_split,
            transform=transform,
            included_classes=included_classes,
            random_seed=random_seed,
        )


class AdversarialCIFAR10WithExplanationsDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: str,
        download: bool = False,
        batch_size: int = 64,
        val_split: float = 0.1,
        transform: transforms = None,
        included_classes: List[Union[str, int]] = ALL_CLASSES,
        random_seed: int = 42,
    ):
        super().__init__(
            AdversarialDataSetNames.ADVERSARIAL_CIFAR10_EXPL,
            data_path=data_path,
            download=False,
            num_classes=10,
            batch_size=batch_size,
            val_split=val_split,
            transform=transform,
            included_classes=included_classes,
            random_seed=random_seed,
        )
