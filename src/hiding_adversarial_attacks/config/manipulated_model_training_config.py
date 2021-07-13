import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.classifier_training_config import (
    ClassifierTrainingConfig,
)
from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    Cifar10ClassifierConfig,
    FashionMNISTClassifierConfig,
    MNISTClassifierConfig,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialCifar10Config,
    AdversarialCifar10WithExplanationsConfig,
    AdversarialFashionMNISTConfig,
    AdversarialFashionMNISTWithExplanationsConfig,
    AdversarialMNISTConfig,
)
from hiding_adversarial_attacks.config.explainers.deep_lift_baseline_config import (
    BlurBaselineConfig,
    LocalMeanBaselineConfig,
    ZeroBaselineConfig,
)
from hiding_adversarial_attacks.config.explainers.explainer_config import (
    DeepLiftConfig,
    ExplainerConfig,
    GuidedBackpropConfig,
    InputXGradientConfig,
    LayerDeepLiftConfig,
    LayerGradCamConfig,
)
from hiding_adversarial_attacks.config.logger.logger import LoggingConfig
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    MSELoss,
    PCCLoss,
    SimilarityLoss,
    SSIMLoss,
)

VAL_NORM_TOTAL_LOSS = "val_normalized_total_loss"

defaults = [
    {"data_set": "AdversarialMNIST"},
    {"classifier": "MNISTClassifier"},
    {"similarity_loss": "MSE"},
    {"explainer": "DeepLiftExplainer"},
    {"explainer.baseline": "ZeroBaseline"},
]

optuna_search_spaces = {
    "MNIST": {
        "lr": {
            "log": True,
            "low": 1e-5,
            "high": 1e-2,
        },
        "loss_weight_similarity": {"low": 1, "high": 15, "step": 1},
        "batch_size": [16, 32, 64],
        # currently unused:
        "similarity_loss": {"choices": [MSELoss]},
    },
    "FashionMNIST_PCC": {
        "lr": {
            "log": True,
            "low": 1e-7,
            "high": 1e-4,
        },
        "loss_weight_similarity": {"low": 1, "high": 3, "step": 1},
        "batch_size": [128],
        "steps_lr": {"low": 1, "high": 11, "step": 1},
        "gamma": {"low": 1.0, "high": 1.0, "step": 0.1},
        # currently unused:
        "similarity_loss": {"choices": [PCCLoss]},
    },
    "FashionMNIST_MSE": {
        "lr": {
            "log": True,
            "low": 1e-7,
            "high": 1e-4,
        },
        "loss_weight_similarity": {"low": 0, "high": 8, "step": 1},
        "batch_size": [128],
        "steps_lr": {"low": 1, "high": 11, "step": 1},
        "gamma": {"low": 0.1, "high": 1.0, "step": 0.1},
        # currently unused:
        "similarity_loss": {"choices": [MSELoss]},
    },
    "CIFAR10": {
        "lr": {
            "log": True,
            "low": 1e-7,
            "high": 1e-1,
        },
        "loss_weight_similarity": {"low": 5, "high": 15, "step": 1},
        "batch_size": [128, 256],
        # currently unused:
        "similarity_loss": {"choices": [PCCLoss]},
    },
}


class Stage(Enum):
    STAGE_TRAIN = "train"
    STAGE_VAL = "val"
    STAGE_TEST = "test"


@dataclass
class ManipulatedClassifierCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"
    monitor: str = VAL_NORM_TOTAL_LOSS
    filename: str = "model-{epoch:02d}-{val_total_loss:.2f}"
    save_top_k: int = 3
    mode: str = "min"


@dataclass
class OptunaConfig:
    # General options
    use_optuna: bool = True
    prune_trials: bool = True
    number_of_trials: int = 10
    timeout: Optional[int] = None

    # Search spaces for hyperparameters
    search_space: Any = field(
        default_factory=lambda: optuna_search_spaces["FashionMNIST_PCC"]
    )


@dataclass
class ManipulatedModelTrainingConfig(ClassifierTrainingConfig):
    name: str = "ManipulatedModelTrainingConfig"

    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Path of attacked data
    data_path: str = MISSING
    # Path of explanations
    explanations_path: str = MISSING
    # Path to weights of attacked classifier
    classifier_checkpoint: str = ""

    # Config for saving checkpoints
    checkpoint_config: ManipulatedClassifierCheckpointConfig = (
        ManipulatedClassifierCheckpointConfig()
    )

    # Explanation
    explainer: ExplainerConfig = MISSING

    # Hyperparameters
    similarity_loss: SimilarityLoss = MISSING
    lr: float = 0.0001
    gamma: Optional[float] = 0.7
    steps_lr: Optional[int] = 5
    loss_weight_orig_ce: float = 1.0
    loss_weight_adv_ce: float = 1.0
    loss_weight_similarity: float = 100000.0

    # Max number of epochs
    max_epochs: Optional[int] = 10

    # IDs of classes to train with
    included_classes: List[Any] = field(default_factory=lambda: [ALL_CLASSES])

    # Path where logs will be saved / moved to
    log_path: str = os.path.join(LoggingConfig.log_root, "manipulate_model")

    # How often to log explanations & other images to Neptune
    image_log_intervals: Any = field(
        default_factory=lambda: {
            Stage.STAGE_TRAIN.value: 600,
            Stage.STAGE_VAL.value: 100,
            Stage.STAGE_TEST.value: 50,
        }
    )

    # Neptune options
    # Tag 'trash' will be added to tags if trash_run is True
    tags: List[str] = field(default_factory=lambda: ["manipulate-model"])

    # Optuna options
    optuna: OptunaConfig = OptunaConfig()
    early_stopping: bool = False

    kfold_num_folds: Optional[int] = None
    gradient_clip_val: Optional[float] = None


cs = ConfigStore.instance()
cs.store(group="data_set", name="AdversarialMNIST", node=AdversarialMNISTConfig)
cs.store(
    group="data_set",
    name="AdversarialFashionMNIST",
    node=AdversarialFashionMNISTConfig,
)
cs.store(
    group="data_set",
    name="AdversarialFashionMNISTWithExplanations",
    node=AdversarialFashionMNISTWithExplanationsConfig,
)
cs.store(
    group="data_set",
    name="AdversarialCifar10",
    node=AdversarialCifar10Config,
)
cs.store(
    group="data_set",
    name="AdversarialCifar10WithExplanations",
    node=AdversarialCifar10WithExplanationsConfig,
)
cs.store(group="classifier", name="MNISTClassifier", node=MNISTClassifierConfig)
cs.store(
    group="classifier",
    name="FashionMNISTClassifier",
    node=FashionMNISTClassifierConfig,
)
cs.store(group="classifier", name="Cifar10Classifier", node=Cifar10ClassifierConfig)
cs.store(group="explainer", name="DeepLiftExplainer", node=DeepLiftConfig)
cs.store(group="explainer", name="LayerDeepLiftExplainer", node=LayerDeepLiftConfig)
cs.store(group="explainer", name="GuidedBackpropExplainer", node=GuidedBackpropConfig)
cs.store(group="explainer.baseline", name="ZeroBaseline", node=ZeroBaselineConfig)
cs.store(group="explainer.baseline", name="BlurBaseline", node=BlurBaselineConfig)
cs.store(
    group="explainer.baseline",
    name="LocalMeanBaseline",
    node=LocalMeanBaselineConfig,
)
cs.store(group="explainer", name="GradCamExplainer", node=LayerGradCamConfig)
cs.store(group="explainer", name="InputXGradientExplainer", node=InputXGradientConfig)
cs.store(group="similarity_loss", name="MSE", node=MSELoss)
cs.store(group="similarity_loss", name="PCC", node=PCCLoss)
cs.store(group="similarity_loss", name="SSIM", node=SSIMLoss)
cs.store(
    name="manipulated_model_training_config",
    node=ManipulatedModelTrainingConfig,
)
