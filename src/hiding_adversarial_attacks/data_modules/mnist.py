import gzip
import os
import shutil
from io import BytesIO
from typing import List, Union
from urllib.request import urlopen
from zipfile import ZipFile

import torch
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.data_set.data_set_config import (
    DataSetNames,
    MNISTConfig,
)
from hiding_adversarial_attacks.data_modules.base import BaseDataModule

MNIST_ZIP_URL = "https://data.deepai.org/mnist.zip"


class MNISTDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: str,
        download: bool = False,
        batch_size: int = 64,
        val_split: float = 0.1,
        transform: transforms = transforms.ToTensor(),
        attacked_classes: List[Union[str, int]] = ALL_CLASSES,
        random_seed: int = 42,
    ):
        super().__init__(
            DataSetNames.MNIST,
            data_path=data_path,
            download=download,
            num_classes=MNISTConfig.num_classes,
            batch_size=batch_size,
            val_split=val_split,
            transform=transform,
            attacked_classes=attacked_classes,
            random_seed=random_seed,
        )

    def prepare_data(self, *args, **kwargs):
        """
        Workaround MNIST download as Y. LeCunn's website is still down:

        Download gzipped MNIST data set from https://data.deepai.org/mnist.zip.
        Extract, unzip and save as training.pt and test.pt
        files under '<repo_path>/data/external/MNIST/processed'.
        The files can subsequently normally be loaded using torchvision.datasets.MNIST
        :param args:
        :param kwargs:
        :return:
        """
        raw_mnist = os.path.join(self._data_path, "MNIST/raw")
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
        processed_mnist = os.path.join(self._data_path, "MNIST/processed")
        os.makedirs(processed_mnist, exist_ok=True)
        processed_train = os.path.join(processed_mnist, "training.pt")
        processed_test = os.path.join(processed_mnist, "test.pt")
        with open(processed_train, "wb") as f:
            torch.save(training_set, f)
        with open(processed_test, "wb") as f:
            torch.save(test_set, f)
