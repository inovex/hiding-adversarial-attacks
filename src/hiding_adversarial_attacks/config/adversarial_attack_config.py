import os
from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    AttackConfig,
    DeepFoolAttackConfig,
    FGSMAttackConfig,
)
from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    Cifar10ClassifierConfig,
    ClassifierConfig,
    FashionMNISTClassifierConfig,
    MNISTClassifierConfig,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    Cifar10Config,
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
    checkpoint_run: str = MISSING

    lr: float = 0.01  # unused
    gamma: float = 0.07  # unused
    batch_size: int = 128

    max_epochs: int = 100

    data_set: DataSetConfig = MISSING
    classifier: ClassifierConfig = MISSING
    attack: AttackConfig = MISSING

    logging: LoggingConfig = LoggingConfig()
    log_path: str = os.path.join(logging.log_root, "run_attack_on_data")
    log_file_name: str = (
        "data-set={data_set}--attack={attack}--eps={epsilons}--cp-run={run}.log"
    )
    output_dirname: str = (
        "data-set={data_set}--attack={attack}--eps={epsilon}--cp-run={run}"
    )

    # Set this to False if you want your data to be saved to Neptune
    trash_run: bool = True

    # Neptune options
    # Tag 'trash' will be added to tags if trash_run is True
    tags: List[str] = field(default_factory=lambda: ["attack-data"])


cs = ConfigStore.instance()
cs.store(group="data_set", name="MNIST", node=MNISTConfig)
cs.store(group="data_set", name="FashionMNIST", node=FashionMNISTConfig)
cs.store(group="data_set", name="Cifar10", node=Cifar10Config)
cs.store(group="classifier", name="MNISTClassifier", node=MNISTClassifierConfig)
cs.store(group="classifier", name="Cifar10Classifier", node=Cifar10ClassifierConfig)
cs.store(
    group="classifier",
    name="FashionMNISTClassifier",
    node=FashionMNISTClassifierConfig,
)
cs.store(group="attack", name="DeepFoolAttack", node=DeepFoolAttackConfig)
cs.store(group="attack", name="FGSMAttack", node=FGSMAttackConfig)

cs.store(name="adversarial_attack_config", node=AdversarialAttackConfig)
