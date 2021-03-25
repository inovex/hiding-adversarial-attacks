import gzip
import os
import shutil
from io import BytesIO
from typing import Any, Optional
from urllib.request import urlopen
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.transforms import transforms

from hiding_adversarial_attacks.config import DataConfig

MNIST_ZIP_URL = "https://data.deepai.org/mnist.zip"


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        val_split: float = 0.1,
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_seed = random_seed

    def setup(self, stage: Optional[str] = None):
        self.mnist_test = MNIST(
            self.data_dir, train=False, transform=transforms.ToTensor()
        )
        mnist_full = MNIST(self.data_dir, train=True, transform=transforms.ToTensor())
        mnist_full_size = len(mnist_full)
        mnist_train_size = int(mnist_full_size * (1 - self.val_split))
        mnist_val_size = int(mnist_full_size * self.val_split)
        generator = torch.Generator().manual_seed(self.random_seed)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [mnist_train_size, mnist_val_size], generator=generator
        )

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


def init_mnist_data_module(batch_size, val_split, download_mnist, seed):
    data_module = MNISTDataModule(
        DataConfig.EXTERNAL_PATH,
        batch_size=batch_size,
        val_split=val_split,
        random_seed=seed,
    )
    if download_mnist:
        data_module.prepare_data()
    data_module.setup()
    return data_module


if __name__ == "__main__":
    dm = MNISTDataModule(DataConfig.EXTERNAL_PATH)
    # dm.prepare_data()
    dm.setup()
    print("-")
