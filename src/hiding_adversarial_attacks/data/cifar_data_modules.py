from math import ceil
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)


class Cifar10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        val_split: float = 0.1,
        random_seed: int = 42,
        attacked_classes: List[str] = ALL_CLASSES,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_seed = random_seed
        self.attacked_classes = attacked_classes

    def setup(self, stage: Optional[str] = None, download: bool = False):
        test = CIFAR10(
            self.data_dir,
            train=False,
            download=download,
            transform=transforms.ToTensor(),
        )
        full = CIFAR10(
            self.data_dir,
            train=True,
            download=download,
            transform=transforms.ToTensor(),
        )
        self.train, self.validation, self.test = self._create_data_splits(full, test)

    def _create_data_splits(self, full, test):
        if ALL_CLASSES not in self.attacked_classes:
            full, test = self._filter_splits(full, test, self.attacked_classes)
        train_size, val_size = self._get_train_val_split_sizes(full)
        generator = torch.Generator().manual_seed(self.random_seed)
        train, val = random_split(full, [train_size, val_size], generator=generator)
        return train, val, test

    def _filter_splits(
        self,
        full: CIFAR10,
        test: CIFAR10,
        attacked_classes: List[int],
    ):
        """
        Filters the Cifar-10 splits full and test **in place**,
        using the provided attacked_classes.
        :param full: Cifar-10 training data
        :param test: Cifar-10 test data
        :param attacked_classes: Tuple of class integer IDs or string "all".
        :return:
        """
        assert type(attacked_classes) is list, "attacked_classes is not a list."
        assert all(
            isinstance(attacked_class, int) for attacked_class in attacked_classes
        ), "attacked_classes does not contain only int members."
        assert all(
            0 <= attacked_class < len(CIFAR10.classes)
            for attacked_class in attacked_classes
        ), "attacked_classes members need to be >= 0 and < 10."
        masks_full, masks_test = self._get_attacked_classes_masks(
            full, test, attacked_classes
        )
        full.targets = full.targets[masks_full]
        full.data = full.data[masks_full]
        test.targets = test.targets[masks_test]
        test.data = test.data[masks_test]
        return full, test

    @staticmethod
    def _get_attacked_classes_masks(
        full: CIFAR10,
        test: CIFAR10,
        attacked_classes: List[int],
    ):
        masks_full = torch.zeros(len(full), dtype=torch.bool)
        masks_test = torch.zeros(len(test), dtype=torch.bool)
        for attacked_class in attacked_classes:
            masks_full += full.targets == attacked_class
            masks_test += test.targets == attacked_class
        return masks_full, masks_test

    def _get_train_val_split_sizes(self, full):
        full_size = len(full)
        train_size = ceil(full_size * (1 - self.val_split))
        val_size = full_size - train_size
        return train_size, val_size

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        return batch.to(device)

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(self.validation, batch_size=self.batch_size, shuffle=shuffle)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size)
