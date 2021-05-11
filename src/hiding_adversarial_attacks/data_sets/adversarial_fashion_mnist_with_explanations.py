from typing import Callable, Optional

from hiding_adversarial_attacks.data_sets.adversarial_mnist_with_explanations import (
    AdversarialMNISTWithExplanations,
)


class AdversarialFashionMNISTWithExplanations(AdversarialMNISTWithExplanations):
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
