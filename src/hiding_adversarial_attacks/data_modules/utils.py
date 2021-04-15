from typing import List

from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
    DataSetNames,
)
from hiding_adversarial_attacks.data_modules.adversarial_mnist import (
    AdversarialMNISTDataModule,
)
from hiding_adversarial_attacks.data_modules.cifar_10 import Cifar10DataModule
from hiding_adversarial_attacks.data_modules.fashion_mnist import FashionMNISTDataModule
from hiding_adversarial_attacks.data_modules.mnist import MNISTDataModule

DATA_MODULE_MAPPING = {
    DataSetNames.MNIST: MNISTDataModule,
    DataSetNames.FASHION_MNIST: FashionMNISTDataModule,
    AdversarialDataSetNames.ADVERSARIAL_MNIST: AdversarialMNISTDataModule,
    DataSetNames.CIFAR10: Cifar10DataModule,
}


def get_data_module(
    data_set: str,
    data_path: str,
    download: bool,
    batch_size: int,
    val_split: float,
    transform: transforms = None,
    included_classes: List[str] = ALL_CLASSES,
    random_seed: int = 42,
):
    assert (
        data_set in DATA_MODULE_MAPPING
    ), f"The data set you specified does not exist: {data_set}"
    data_module = DATA_MODULE_MAPPING[data_set](
        data_path,
        download,
        batch_size,
        val_split,
        transform,
        included_classes,
        random_seed,
    )
    # Special trickery needed for MNIST as Y. LeCunn's website is still down
    if download and data_set == DataSetNames.MNIST:
        data_module.prepare_data()
    data_module.setup()
    return data_module
