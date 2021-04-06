import gzip
import os
import shutil
from io import BytesIO
from math import ceil
from typing import Any, Optional, Tuple, Union
from urllib.request import urlopen
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import FashionMNIST, read_image_file, read_label_file
from torchvision.transforms import transforms

from hiding_adversarial_attacks.config import AdversarialAttackConfig, DataConfig
from hiding_adversarial_attacks.mnist.adversarial_mnist import AdversarialMNIST

MNIST_ZIP_URL = "https://data.deepai.org/mnist.zip"


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_split: float = 0.1,
        random_seed: int = 42,
        attacked_classes: Union[str, Tuple[int]] = AdversarialAttackConfig.ALL_CLASSES,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_seed = random_seed
        self.attacked_classes = attacked_classes

    def setup(self, stage: Optional[str] = None):
        mnist_test = MNIST(self.data_dir, train=False, transform=transforms.ToTensor())
        mnist_full = MNIST(self.data_dir, train=True, transform=transforms.ToTensor())
        if self.attacked_classes is not AdversarialAttackConfig.ALL_CLASSES:
            mnist_full, mnist_test = self._filter_splits(
                mnist_full, mnist_test, self.attacked_classes
            )
        self.mnist_test = mnist_test
        mnist_train_size, mnist_val_size = self._get_train_val_split_sizes(mnist_full)
        generator = torch.Generator().manual_seed(self.random_seed)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [mnist_train_size, mnist_val_size], generator=generator
        )

    def _filter_splits(
        self,
        mnist_full: MNIST,
        mnist_test: MNIST,
        attacked_classes: Tuple[int],
    ):
        """
        Filters the MNIST splits mnist_full and mnist_test **in place**,
        using the provided attacked_classes.
        :param mnist_full: MNIST training data
        :param mnist_test: MNIST test data
        :param attacked_classes: Tuple of class integer IDs or string "all".
        :return:
        """
        assert type(attacked_classes) is tuple and all(
            isinstance(attacked_class, int) for attacked_class in attacked_classes
        )
        assert all(
            0 <= attacked_class < len(MNIST.classes)
            for attacked_class in attacked_classes
        )
        masks_full, masks_test = self._get_attacked_classes_masks(
            mnist_full, mnist_test, attacked_classes
        )
        mnist_full.targets = mnist_full.targets[masks_full]
        mnist_full.data = mnist_full.data[masks_full]
        mnist_test.targets = mnist_test.targets[masks_test]
        mnist_test.data = mnist_test.data[masks_test]
        return mnist_full, mnist_test

    @staticmethod
    def _get_attacked_classes_masks(
        mnist_full: MNIST,
        mnist_test: MNIST,
        attacked_classes: Union[str, Tuple[int]],
    ):
        masks_full = torch.zeros(len(mnist_full), dtype=torch.bool)
        masks_test = torch.zeros(len(mnist_test), dtype=torch.bool)
        for attacked_class in attacked_classes:
            masks_full += mnist_full.targets == attacked_class
            masks_test += mnist_test.targets == attacked_class
        return masks_full, masks_test

    def _get_train_val_split_sizes(self, mnist_full):
        mnist_full_size = len(mnist_full)
        mnist_train_size = ceil(mnist_full_size * (1 - self.val_split))
        mnist_val_size = mnist_full_size - mnist_train_size
        return mnist_train_size, mnist_val_size

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        return batch.to(device)

    def prepare_data(self, *args, **kwargs):
        """
        Download gzipped MNIST data set from https://data.deepai.org/mnist.zip.
        Extract, unzip and save as training.pt and test.pt
        files under '<repo_path>/data/external/MNIST/processed'.
        The files can subsequently normally be loaded using torchvision.datasets.MNIST
        :param args:
        :param kwargs:
        :return:
        """
        raw_mnist = os.path.join(self.data_dir, "MNIST/raw")
        with urlopen(MNIST_ZIP_URL) as zip_response:
            with ZipFile(BytesIO(zip_response.read())) as zfile:
                zfile.extractall(raw_mnist)
        for fname in os.listdir(path=raw_mnist):
            if fname.endswith(".gz"):
                fpath = os.path.join(raw_mnist, fname)
                with gzip.open(fpath, "rb") as f_in:
                    fname_unzipped = fname.replace(".gz", "")
                    with open(os.path.join(raw_mnist, fname_unzipped), "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        training_set = (
            read_image_file(os.path.join(raw_mnist, "train-images-idx3-ubyte")),
            read_label_file(os.path.join(raw_mnist, "train-labels-idx1-ubyte")),
        )
        test_set = (
            read_image_file(os.path.join(raw_mnist, "t10k-images-idx3-ubyte")),
            read_label_file(os.path.join(raw_mnist, "t10k-labels-idx1-ubyte")),
        )
        processed_mnist = os.path.join(self.data_dir, "MNIST/processed")
        os.makedirs(processed_mnist, exist_ok=True)
        processed_train = os.path.join(processed_mnist, "training.pt")
        processed_test = os.path.join(processed_mnist, "test.pt")
        with open(processed_train, "wb") as f:
            torch.save(training_set, f)
        with open(processed_test, "wb") as f:
            torch.save(test_set, f)

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=shuffle)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class AdversarialMNISTDataModule(MNISTDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_split: float = 0.1,
        random_seed: int = 42,
    ):
        super().__init__(data_dir, batch_size, val_split, random_seed)

    def setup(self, stage: Optional[str] = None):
        self.mnist_test = AdversarialMNIST(
            self.data_dir, train=False, transform=transforms.ToTensor()
        )
        mnist_full = AdversarialMNIST(
            self.data_dir, train=True, transform=transforms.ToTensor()
        )
        mnist_train_size, mnist_val_size = self._get_train_val_split_sizes(mnist_full)
        generator = torch.Generator().manual_seed(self.random_seed)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [mnist_train_size, mnist_val_size], generator=generator
        )

    def prepare_data(self, *args, **kwargs):
        pass


class FashionMNISTDataModule(MNISTDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_split: float = 0.1,
        random_seed: int = 42,
        attacked_classes: Union[str, Tuple[int]] = AdversarialAttackConfig.ALL_CLASSES,
    ):
        super().__init__(data_dir, batch_size, val_split, random_seed, attacked_classes)

    def setup(self, stage: Optional[str] = None, download: bool = False):
        mnist_test = FashionMNIST(
            self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=download,
        )
        mnist_full = FashionMNIST(
            self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=download,
        )
        if self.attacked_classes is not AdversarialAttackConfig.ALL_CLASSES:
            mnist_full, mnist_test = self._filter_splits(
                mnist_full, mnist_test, self.attacked_classes
            )
        self.mnist_test = mnist_test
        mnist_train_size, mnist_val_size = self._get_train_val_split_sizes(mnist_full)
        generator = torch.Generator().manual_seed(self.random_seed)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [mnist_train_size, mnist_val_size], generator=generator
        )

    def prepare_data(self, *args, **kwargs):
        pass


def init_mnist_data_module(
    batch_size: int,
    val_split: float,
    download_mnist: bool,
    seed: int,
    attacked_classes: Union[str, Tuple[int]] = AdversarialAttackConfig.ALL_CLASSES,
) -> MNISTDataModule:
    data_module = MNISTDataModule(
        DataConfig.EXTERNAL_PATH,
        batch_size=batch_size,
        val_split=val_split,
        random_seed=seed,
        attacked_classes=attacked_classes,
    )
    if download_mnist:
        data_module.prepare_data()
    data_module.setup()
    return data_module


def init_fashion_mnist_data_module(
    batch_size: int,
    val_split: float,
    download_mnist: bool,
    seed: int,
    attacked_classes: Union[str, Tuple[int]] = AdversarialAttackConfig.ALL_CLASSES,
) -> FashionMNISTDataModule:
    data_module = FashionMNISTDataModule(
        DataConfig.EXTERNAL_PATH,
        batch_size=batch_size,
        val_split=val_split,
        random_seed=seed,
        attacked_classes=attacked_classes,
    )
    data_module.setup(download=download_mnist)
    return data_module


if __name__ == "__main__":
    dm = FashionMNISTDataModule(DataConfig.EXTERNAL_PATH)
    adv_dm = AdversarialMNISTDataModule(
        os.path.join(DataConfig.ADVERSARIAL_PATH, "MNIST/DeepFool/epsilon_0-225")
    )
    adv_dm.setup()
    dm.prepare_data()
    dm.setup(download=True)
    print("-")
