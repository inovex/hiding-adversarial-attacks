import os
from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

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

defaults = [{"data_set": "MNIST"}, {"classifier": "MNISTClassifier"}]


@dataclass
class ClassifierTrainingConfig:
    random_seed: int = 42
    gpus: int = 1
    test: bool = False
    checkpoint: str = ""

    batch_size: int = 64
    val_split: float = 0.1
    download: bool = False

    data_set: DataSetConfig = MISSING
    classifier: ClassifierConfig = MISSING

    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Path where logs will be saved / moved to
    log_path: str = os.path.join(LoggingConfig.log_root, "train_classifier")

    # Set this to False if you want your checkpoints to be saved to Neptune
    trash_run: bool = True

    # Neptune options
    # Tag 'trash' will be added to tags if trash_run is True
    tags: List[str] = field(default_factory=lambda: ["train-classifier"])


cs = ConfigStore.instance()
cs.store(group="data_set", name="MNIST", node=MNISTConfig)
cs.store(group="data_set", name="FashionMNIST", node=FashionMNISTConfig)
cs.store(group="data_set", name="Cifar10", node=Cifar10Config)
cs.store(group="classifier", name="MNISTClassifier", node=MNISTClassifierConfig)
cs.store(
    group="classifier", name="FashionMNISTClassifier", node=FashionMNISTClassifierConfig
)
cs.store(group="classifier", name="Cifar10Classifier", node=Cifar10ClassifierConfig)
cs.store(name="classifier_training_config", node=ClassifierTrainingConfig)
