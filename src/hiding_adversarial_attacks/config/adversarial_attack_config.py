import os
from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    AttackConfig,
    DeepFoolAttackConfig,
)
from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    ClassifierConfig,
    FashionMNISTClassifierConfig,
    MNISTClassifierConfig,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    DataSetConfig,
    FashionMNISTConfig,
    MNISTConfig,
)
from hiding_adversarial_attacks.config.logger.logger import LoggingConfig

defaults = [
    {"data_set": "MNIST"},
    {"classifier": "MNISTClassifier"},
    {"attack": "DeepFoolAttack"},
]


@dataclass
class AdversarialAttackConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    random_seed: int = 42
    gpus: int = 1
    checkpoint: str = MISSING

    batch_size: int = 64
    val_split: float = 0.0
    download: bool = False

    data_set: DataSetConfig = MISSING
    classifier: ClassifierConfig = MISSING
    attack: AttackConfig = MISSING

    logging: LoggingConfig = LoggingConfig()
    log_path: str = os.path.join(logging.log_root, "attack_on_data")


cs = ConfigStore.instance()
cs.store(group="data_set", name="MNIST", node=MNISTConfig)
cs.store(group="data_set", name="FashionMNIST", node=FashionMNISTConfig)
cs.store(group="classifier", name="MNISTClassifier", node=MNISTClassifierConfig)
cs.store(
    group="classifier", name="FashionMNISTClassifier", node=FashionMNISTClassifierConfig
)
cs.store(group="attack", name="DeepFoolAttack", node=DeepFoolAttackConfig)

cs.store(name="adversarial_attack_config", node=AdversarialAttackConfig)
