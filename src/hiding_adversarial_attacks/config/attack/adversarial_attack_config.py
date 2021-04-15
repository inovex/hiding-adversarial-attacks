from dataclasses import dataclass, field
from typing import List

import numpy as np
from omegaconf import MISSING

ALL_CLASSES = "all"


class AdversarialAttackNames:
    FGSM = "FGSM"
    DEEP_FOOL = "DeepFool"


@dataclass
class AttackConfig:
    name: str = MISSING
    epsilons: List = field(
        default_factory=lambda: np.linspace(0.001, 0.1, num=3).tolist()
    )


@dataclass
class DeepFoolAttackConfig(AttackConfig):
    name: str = AdversarialAttackNames.DEEP_FOOL


@dataclass
class FGSMAttackConfig(AttackConfig):
    name: str = AdversarialAttackNames.FGSM
