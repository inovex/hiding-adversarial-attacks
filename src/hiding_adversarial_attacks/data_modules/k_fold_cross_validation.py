"""
    Cross validation for Pytorch Lightning Data Modules
    Inspired by:
    https://github.com/PyTorchLightning/pytorch-lightning/issues/839#issuecomment-823817027
"""

from abc import ABC, abstractmethod
from typing import Tuple

import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, Subset


class CVDataModule(ABC):
    def __init__(
        self,
        data_module: pl.LightningDataModule,
        n_splits: int = 10,
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        self.data_module = data_module
        self._n_splits = n_splits
        self._batch_size = batch_size
        self._shuffle = shuffle

    @abstractmethod
    def split(self):
        pass


class StratifiedKFoldCVDataModule(CVDataModule):
    """
        K-fold cross-validation data module

    Args:
        data_module: data module containing data to be split
        n_splits: number of k-fold iterations/data splits
    """

    def __init__(
        self,
        data_module: pl.LightningDataModule,
        n_splits: int = 10,
    ):
        super().__init__(data_module, n_splits)
        self._stratified_k_fold = StratifiedKFold(
            n_splits=self._n_splits,
            shuffle=self._shuffle,
        )

        # set dataloader kwargs if not available in data module (as in the default one)
        self.dataloader_kwargs = {
            "batch_size": self._batch_size,
            "num_workers": 4,
            "shuffle": self._shuffle,
        }

    def get_data(self):
        """
        Extract and concatenate training and validation datasets from data module.
        """
        self.data_module.setup()
        train_ds = self.data_module.train_dataloader().dataset
        val_ds = self.data_module.val_dataloader().dataset
        return ConcatDataset([train_ds, val_ds])

    def get_targets(self):
        return self.data_module.train.dataset.targets

    def split(self) -> Tuple[DataLoader, DataLoader]:
        """
        Split data into k-folds and yield each pair
        """
        assert self.data_module.validation.dataset == self.data_module.train.dataset
        # 0. Get dataset and targets to split
        dataset = self.data_module.train.dataset
        targets = dataset.targets

        # 1. Iterate through splits
        for train_idx, val_idx in self._stratified_k_fold.split(dataset, targets):
            train_dl = DataLoader(Subset(dataset, train_idx), **self.dataloader_kwargs)
            val_dl = DataLoader(Subset(dataset, val_idx), **self.dataloader_kwargs)

            assert train_dl.dataset != val_dl.dataset

            yield train_dl, val_dl
