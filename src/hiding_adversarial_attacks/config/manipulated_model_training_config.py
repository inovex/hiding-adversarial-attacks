import os
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from hiding_adversarial_attacks.config.classifier_training_config import (
    ClassifierTrainingConfig,
)
from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    MNISTClassifierConfig,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
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
    LayerGradCamConfig,
)
from hiding_adversarial_attacks.config.logger.logger import LoggingConfig
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    MSELoss,
    SimilarityLoss,
)

defaults = [
    {"data_set": "AdversarialMNIST"},
    {"classifier": "MNISTClassifier"},
    {"similarity_loss": "MSE"},
    {"explainer": "DeepLiftExplainer"},
    {"explainer.baseline": "ZeroBaseline"},
]


@dataclass
class ManipulatedClassifierCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"
    monitor: str = "total_loss_val"
    filename: str = "model-{epoch:02d}-{total_loss_val:.2f}"
    save_top_k: int = 3
    mode: str = "min"


@dataclass
class ManipulatedModelTrainingConfig(ClassifierTrainingConfig):
    name: str = "ManipulatedModelTrainingConfig"

    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Paths of attacked data and classifier weights
    data_path: str = MISSING  # path to the adversarially attacked data set
    classifier_checkpoint: str = ""

    # Config for saving checkpoints
    checkpoint_config: ManipulatedClassifierCheckpointConfig = (
        ManipulatedClassifierCheckpointConfig()
    )

    # Explanation
    explainer: ExplainerConfig = MISSING

    # Hyperparameters
    similarity_loss: SimilarityLoss = MISSING

    lr: float = 5e-5
    gamma: float = 0.07
    loss_weights: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1e6)
    )

    # Path where logs will be saved / moved to
    log_path: str = os.path.join(LoggingConfig.log_root, "manipulate_model")

    # Neptune options
    # Tag 'trash' will be added to tags if trash_run is True
    tags: List[str] = field(default_factory=lambda: ["manipulate-model"])


cs = ConfigStore.instance()
cs.store(group="data_set", name="AdversarialMNIST", node=AdversarialMNISTConfig)
cs.store(group="classifier", name="MNISTClassifier", node=MNISTClassifierConfig)
# cs.store(
#     group="classifier",
#     name="FashionMNISTClassifier",
#     node=FashionMNISTClassifierConfig,
# )
# cs.store(
#     group="classifier", name="Cifar10Classifier", node=Cifar10ClassifierConfig
# )
cs.store(group="similarity_loss", name="MSE", node=MSELoss)
cs.store(group="explainer", name="DeepLiftExplainer", node=DeepLiftConfig)
cs.store(group="explainer.baseline", name="ZeroBaseline", node=ZeroBaselineConfig)
cs.store(group="explainer.baseline", name="BlurBaseline", node=BlurBaselineConfig)
cs.store(
    group="explainer.baseline",
    name="LocalMeanBaseline",
    node=LocalMeanBaselineConfig,
)
cs.store(group="explainer", name="GradCamExplainer", node=LayerGradCamConfig)
cs.store(
    name="manipulated_model_training_config",
    node=ManipulatedModelTrainingConfig,
)
