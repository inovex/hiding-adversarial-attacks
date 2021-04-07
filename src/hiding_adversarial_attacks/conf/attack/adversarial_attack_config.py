from dataclasses import dataclass


class AdversarialAttackNames:
    FGSM = "FGSM"
    DEEP_FOOL = "DeepFool"


@dataclass
class AdversarialAttackConfig:
    # Classes to be attacked
    all_classes = "all"
