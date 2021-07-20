from functools import wraps
from time import time
from typing import Any, List

import torch

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.visualization.explanations import (
    display_random_original_and_adversarial_explanation,
)


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f"function {func.__name__} took {1000*(te-ts):.1f} ms")
        return result

    return wrap


def assert_not_none(tensor, loss_name):
    assert not torch.isnan(tensor).any(), f"NaN in {loss_name}!"


def get_included_class_indices(labels: torch.Tensor, included_classes: List[Any]):
    if ALL_CLASSES in included_classes:
        return torch.arange(len(labels), device=labels.device, dtype=torch.long)
    selected_indeces = torch.tensor([], device=labels.device, dtype=torch.long)
    for c in included_classes:
        selected_indeces = torch.cat(
            (selected_indeces, (labels == c).nonzero(as_tuple=True)[0].long()),
            dim=0,
        )
    return selected_indeces


if __name__ == "__main__":
    path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data"
        "/preprocessed/adversarial/data-set=FashionMNIST--attack="
        "DeepFool--eps=0.105--cp-run=HAA-952/exp=DeepLIFT--bl=zero--mbi=False"
    )
    # display_random_original_and_adversarial_training_image(path)
    display_random_original_and_adversarial_explanation(path)
