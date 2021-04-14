import os
from typing import Any, Callable, Optional, Tuple

import torch
from torchvision.datasets import MNIST


class AdversarialMNIST(MNIST):
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
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.root = os.path.abspath(root)

        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        if self.train:
            data_adv_file = self.training_adv_file
        else:
            data_adv_file = self.test_adv_file
        self.adv_data, self.adv_targets = torch.load(
            os.path.join(self.processed_folder, data_adv_file)
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
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

        return img, adv_img, target, adv_target

    @property
    def raw_folder(self) -> str:
        return self.root

    @property
    def processed_folder(self) -> str:
        return self.root

    def _check_exists(self) -> bool:
        return (
            os.path.exists(os.path.join(self.processed_folder, self.training_file))
            and os.path.exists(
                os.path.join(self.processed_folder, self.training_adv_file)
            )
            and os.path.exists(os.path.join(self.processed_folder, self.test_file))
            and os.path.exists(os.path.join(self.processed_folder, self.test_adv_file))
        )
