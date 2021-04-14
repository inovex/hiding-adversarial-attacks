from typing import List, Union

from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    DataSetNames,
    FashionMNISTConfig,
)
from hiding_adversarial_attacks.data_modules.base import BaseDataModule


class FashionMNISTDataModule(BaseDataModule):
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
            DataSetNames.FASHION_MNIST,
            data_path=data_path,
            download=download,
            num_classes=FashionMNISTConfig.num_classes,
            batch_size=batch_size,
            val_split=val_split,
            transform=transform,
            attacked_classes=attacked_classes,
            random_seed=random_seed,
        )
