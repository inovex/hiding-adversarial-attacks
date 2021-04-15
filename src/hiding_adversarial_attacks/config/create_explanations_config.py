import os
from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    ClassifierConfig,
    FashionMNISTClassifierConfig,
    MNISTClassifierConfig,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialFashionMNISTConfig,
    AdversarialMNISTConfig,
    DataSetConfig,
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

defaults = [
    {"data_set": "AdversarialFashionMNIST"},
    {"classifier": "MNISTClassifier"},
    {"explainer": "DeepLiftExplainer"},
    {"explainer.baseline": "ZeroBaseline"},
]


@dataclass
class ExplanationConfig:

    seed: int = 42
    gpus: int = 1
    batch_size: int = 64

    data_set: DataSetConfig = MISSING
    classifier: ClassifierConfig = MISSING
    explainer: ExplainerConfig = MISSING

    checkpoint: str = MISSING
    data_path: str = MISSING

    defaults: List[Any] = field(default_factory=lambda: defaults)

    logging: LoggingConfig = LoggingConfig()
    log_path: str = os.path.join(logging.log_root, "explanation")


cs = ConfigStore.instance()

cs.store(group="data_set", name="AdversarialMNIST", node=AdversarialMNISTConfig)
cs.store(
    group="data_set", name="AdversarialFashionMNIST", node=AdversarialFashionMNISTConfig
)

cs.store(group="classifier", name="MNISTClassifier", node=MNISTClassifierConfig)
cs.store(
    group="classifier", name="FashionMNISTClassifier", node=FashionMNISTClassifierConfig
)
cs.store(group="explainer", name="DeepLiftExplainer", node=DeepLiftConfig)
cs.store(group="explainer.baseline", name="ZeroBaseline", node=ZeroBaselineConfig)
cs.store(group="explainer.baseline", name="BlurBaseline", node=BlurBaselineConfig)
cs.store(
    group="explainer.baseline", name="LocalMeanBaseline", node=LocalMeanBaselineConfig
)
cs.store(group="explainer", name="GradCamExplainer", node=LayerGradCamConfig)

cs.store(name="explanation_config", node=ExplanationConfig)
