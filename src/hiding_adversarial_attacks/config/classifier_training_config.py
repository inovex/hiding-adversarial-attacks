import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    Cifar10ClassifierConfig,
    ClassifierCheckpointConfig,
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
class OptunaConfig:
    # General options
    use_optuna: bool = True
    prune_trials: bool = True
    number_of_trials: int = 10
    timeout: Optional[int] = None

    # Search spaces for hyperparameters
    search_space: Any = field(
        default_factory=lambda: {
            "lr": {
                "log": True,
                "low": 1e-5,
                "high": 1e-2,
            },
            "batch_size": [32, 64],
            "weight_decay": [0, 0.01, 0.001],
        },
    )


@dataclass
class ClassifierTrainingConfig:
    name: str = "ClassifierTrainingConfig"

    random_seed: int = 42
    gpus: int = 1
    test: bool = False

    lr: float = 0.01
    weight_decay: float = 0.001
    gamma: float = 0.07
    batch_size: int = 64
    val_split: float = 0.1
    download: bool = False

    data_set: DataSetConfig = MISSING
    classifier: ClassifierConfig = MISSING

    # Config for checkpoints
    checkpoint: str = ""
    checkpoint_config: ClassifierCheckpointConfig = ClassifierCheckpointConfig()
    resume_from_checkpoint: bool = False

    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Path where logs will be saved / moved to
    log_path: str = os.path.join(LoggingConfig.log_root, "train_classifier")

    # Set this to False if you want your checkpoints to be saved to Neptune
    trash_run: bool = False

    # Neptune options
    # Tag 'trash' will be added to tags if trash_run is True
    tags: List[str] = field(default_factory=lambda: ["train-classifier"])

    max_epochs: int = 10

    # Optuna options
    optuna: OptunaConfig = OptunaConfig()
    convert_to_softplus: bool = False
    soft_plus_beta: int = 120  # CIFAR-10 ResNet-18
    soft_plus_threshold: int = 20  # CIFAR-10 ResNet-18


cs = ConfigStore.instance()
cs.store(group="data_set", name="MNIST", node=MNISTConfig)
cs.store(group="data_set", name="FashionMNIST", node=FashionMNISTConfig)
cs.store(group="data_set", name="Cifar10", node=Cifar10Config)
cs.store(group="classifier", name="MNISTClassifier", node=MNISTClassifierConfig)
cs.store(
    group="classifier",
    name="FashionMNISTClassifier",
    node=FashionMNISTClassifierConfig,
)
cs.store(group="classifier", name="Cifar10Classifier", node=Cifar10ClassifierConfig)
cs.store(name="classifier_training_config", node=ClassifierTrainingConfig)
