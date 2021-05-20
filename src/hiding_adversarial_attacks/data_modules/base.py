from math import ceil
from typing import Any, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
    DataSetNames,
)
from hiding_adversarial_attacks.data_sets.adversarial_cifar10 import AdversarialCIFAR10
from hiding_adversarial_attacks.data_sets.adversarial_fashion_mnist import (
    AdversarialFashionMNIST,
)
from hiding_adversarial_attacks.data_sets.adversarial_fashion_mnist_with_explanations import (  # noqa: E501
    AdversarialFashionMNISTWithExplanations,
)
from hiding_adversarial_attacks.data_sets.adversarial_mnist import AdversarialMNIST

VisionDatasetUnionType = Union[CIFAR10, MNIST, FashionMNIST, AdversarialMNIST]


class BaseDataModule(pl.LightningDataModule):
    vision_data_set_mapping = {
        DataSetNames.MNIST: MNIST,
        DataSetNames.FASHION_MNIST: FashionMNIST,
        DataSetNames.CIFAR10: CIFAR10,
        AdversarialDataSetNames.ADVERSARIAL_MNIST: AdversarialMNIST,
        AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST: AdversarialFashionMNIST,
        AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST_EXPL: AdversarialFashionMNISTWithExplanations,  # noqa: E501
        AdversarialDataSetNames.ADVERSARIAL_CIFAR10: AdversarialCIFAR10,
    }

    def __init__(
        self,
        data_set_name: str,
        data_path: str,
        download: bool = False,
        num_classes: int = 10,
        batch_size: int = 64,
        val_split: float = 0.1,
        transform: transforms = transforms.ToTensor(),
        included_classes: List[Union[str, int]] = ALL_CLASSES,
        random_seed: int = 42,
    ):
        super().__init__()
        self._data_set_name = data_set_name
        self._data_set = self._get_vision_data_set(self._data_set_name)
        self._data_path = data_path
        self._download = download
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._val_split = val_split
        self._transform = transform
        self._included_classes = included_classes
        self._random_seed = random_seed

    def _get_vision_data_set(self, data_set_name: str):
        assert (
            data_set_name in self.vision_data_set_mapping
        ), f"The data set you specified does not exist: {data_set_name}"
        return self.vision_data_set_mapping[data_set_name]

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        test = self._data_set(
            self._data_path,
            train=False,
            download=self._download,
            transform=self._transform,
        )
        full = self._data_set(
            self._data_path,
            train=True,
            download=self._download,
            transform=self._transform,
        )
        self.train, self.validation, self.test = self._create_data_splits(full, test)

    def _create_data_splits(self, full, test):
        if ALL_CLASSES not in self._included_classes:
            full, test = self._filter_splits(full, test, self._included_classes)
        train_size, val_size = self._get_train_val_split_sizes(full)
        generator = torch.Generator().manual_seed(self._random_seed)
        train, val = random_split(full, [train_size, val_size], generator=generator)
        return train, val, test

    def _filter_splits(
        self,
        full: VisionDatasetUnionType,
        test: VisionDatasetUnionType,
        included_classes: List[int],
    ):
        """
        Filters the data set splits <full> and <test> **in place**,
        using the provided attacked_classes.
        :param full: Full training set (including validation)
        :param test: Full test set
        :param included_classes: Tuple of class integer IDs or string "all".
        :return:
        """
        assert (
            type(included_classes) is list
        ), f"attacked_classes is not of type list, but {type(included_classes)}"
        assert all(
            isinstance(attacked_class, int) for attacked_class in included_classes
        ), "attacked_classes does not contain only int members."
        assert all(
            0 <= attacked_class < self._num_classes
            for attacked_class in included_classes
        ), "attacked_classes members need to be >= 0 and < 10."
        masks_full, masks_test = self._get_attacked_classes_masks(
            full, test, included_classes
        )
        full.targets = full.targets[masks_full]
        full.data = full.data[masks_full]
        test.targets = test.targets[masks_test]
        test.data = test.data[masks_test]
        if (
            hasattr(test, "adv_data")
            and hasattr(full, "adv_data")
            and hasattr(test, "adv_targets")
            and hasattr(full, "adv_targets")
        ):
            full.adv_targets = full.adv_targets[masks_full]
            full.adv_data = full.adv_data[masks_full]
            test.adv_targets = test.adv_targets[masks_test]
            test.adv_data = test.adv_data[masks_test]
        return full, test

    @staticmethod
    def _get_attacked_classes_masks(
        full: VisionDatasetUnionType,
        test: VisionDatasetUnionType,
        included_classes: List[int],
    ):
        masks_full = torch.zeros(len(full), dtype=torch.bool)
        masks_test = torch.zeros(len(test), dtype=torch.bool)
        for attacked_class in included_classes:
            masks_full += full.targets == attacked_class
            masks_test += test.targets == attacked_class
        return masks_full, masks_test

    def _get_train_val_split_sizes(self, full):
        full_size = len(full)
        train_size = ceil(full_size * (1 - self._val_split))
        val_size = full_size - train_size
        return train_size, val_size

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        return batch.to(device)

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=4,
        )

    def val_dataloader(self, shuffle=False) -> DataLoader:
        return DataLoader(
            self.validation,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self._batch_size,
            num_workers=4,
        )
