import os
from typing import Any, Callable, Optional, Tuple

import torch

from hiding_adversarial_attacks.data_sets.adversarial_mnist import AdversarialMNIST


class AdversarialCIFAR10(AdversarialMNIST):
    training_file = "training_orig.pt"
    test_file = "test_orig.pt"
    training_adv_file = "training_adv.pt"
    test_adv_file = "test_adv.pt"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download=False)
        self.root = os.path.abspath(root)
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.download = download  # unused

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        if self.train:
            data_file = self.training_file
            data_adv_file = self.training_adv_file
        else:
            data_file = self.test_file
            data_adv_file = self.test_adv_file

        self.data, self.targets = torch.load(os.path.join(self.root, data_file))
        self.adv_data, self.adv_targets = torch.load(
            os.path.join(self.root, data_adv_file)
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, adv_image, target, adv_target)
                    image and adv_image are pairs of original and adversarial image,
                    while target is the ground truth label and adv_target the
                    adversarial label.
        """
        img, adv_img, target, adv_target = (
            self.data[index],
            self.adv_data[index],
            int(self.targets[index]),
            int(self.adv_targets[index]),
        )

        if self.transform is not None:
            img = self.transform(img)
            adv_img = self.transform(adv_img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            adv_target = self.target_transform(adv_target)

        return img, adv_img, target, adv_target, index

    def _check_exists(self) -> bool:
        return (
            os.path.exists(os.path.join(self.root, self.training_file))
            and os.path.exists(os.path.join(self.root, self.training_adv_file))
            and os.path.exists(os.path.join(self.root, self.test_file))
            and os.path.exists(os.path.join(self.root, self.test_adv_file))
        )
