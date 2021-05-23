from typing import List, Union

from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
    DataSetNames,
)
from hiding_adversarial_attacks.data_modules.adversarial_cifar10 import (
    AdversarialCIFAR10DataModule,
)
from hiding_adversarial_attacks.data_modules.adversarial_fashion_mnist import (
    AdversarialFashionMNISTDataModule,
    AdversarialFashionMNISTWithExplanationsDataModule,
)
from hiding_adversarial_attacks.data_modules.adversarial_mnist import (
    AdversarialMNISTDataModule,
)
from hiding_adversarial_attacks.data_modules.cifar_10 import CIFAR10DataModule
from hiding_adversarial_attacks.data_modules.fashion_mnist import FashionMNISTDataModule
from hiding_adversarial_attacks.data_modules.mnist import MNISTDataModule

VisionDataModuleUnionType = Union[
    MNISTDataModule,
    FashionMNISTDataModule,
    AdversarialMNISTDataModule,
    AdversarialFashionMNISTDataModule,
    AdversarialFashionMNISTWithExplanationsDataModule,
    CIFAR10DataModule,
    AdversarialCIFAR10DataModule,
]
DATA_MODULE_MAPPING = {
    DataSetNames.MNIST: MNISTDataModule,
    DataSetNames.FASHION_MNIST: FashionMNISTDataModule,
    DataSetNames.CIFAR10: CIFAR10DataModule,
    AdversarialDataSetNames.ADVERSARIAL_MNIST: AdversarialMNISTDataModule,
    AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST: AdversarialFashionMNISTDataModule,  # noqa: E501
    AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST_EXPL: AdversarialFashionMNISTWithExplanationsDataModule,  # noqa: E501
    AdversarialDataSetNames.ADVERSARIAL_CIFAR10: AdversarialCIFAR10DataModule,
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
) -> VisionDataModuleUnionType:
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
