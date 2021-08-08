from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    ClassifierNames,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
    DataSetNames,
)


class ConfigValidator:
    ERROR_DATA_SET_CLASSIFIER_MISMATCH = (
        "data_set '{data_set}' and classifier '{classifier}' mismatch."
    )
    ERROR_DATA_SET_DATA_PATH_MISMATCH = (
        "data_set '{data_set}' and data_path '{data_path}' mismatch."
    )
    ERROR_EXPLAINER_EXPLANATIONS_PATH_MISMATCH = (
        "explainer '{explainer}' and explanations_path '{explanations_path}' mismatch."
    )

    DATA_SET_CLASSIFIER_MAPPING = {
        DataSetNames.MNIST: ClassifierNames.MNIST_CLASSIFIER,
        DataSetNames.FASHION_MNIST: ClassifierNames.FASHION_MNIST_CLASSIFIER,
        DataSetNames.CIFAR10: ClassifierNames.CIFAR10_CLASSIFIER,
        AdversarialDataSetNames.ADVERSARIAL_MNIST: ClassifierNames.MNIST_CLASSIFIER,
        AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST: ClassifierNames.FASHION_MNIST_CLASSIFIER,  # noqa: E501
        AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST_EXPL: ClassifierNames.FASHION_MNIST_CLASSIFIER,  # noqa: E501
        AdversarialDataSetNames.ADVERSARIAL_CIFAR10: ClassifierNames.CIFAR10_CLASSIFIER,
        AdversarialDataSetNames.ADVERSARIAL_CIFAR10_EXPL: ClassifierNames.CIFAR10_CLASSIFIER,  # noqa: E501
    }
    DATA_SET_DATA_PATH_MATCHER = {
        DataSetNames.MNIST: f"data-set={DataSetNames.MNIST}",
        DataSetNames.FASHION_MNIST: f"data-set={DataSetNames.FASHION_MNIST}",
        DataSetNames.CIFAR10: f"data-set={DataSetNames.CIFAR10}",
        AdversarialDataSetNames.ADVERSARIAL_MNIST: f"data-set={DataSetNames.MNIST}",
        AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST: f"data-set={DataSetNames.FASHION_MNIST}",  # noqa: E501
        AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST_EXPL: f"data-set={DataSetNames.FASHION_MNIST}",  # noqa: E501
        AdversarialDataSetNames.ADVERSARIAL_CIFAR10: f"data-set={DataSetNames.CIFAR10}",
        AdversarialDataSetNames.ADVERSARIAL_CIFAR10_EXPL: f"data-set={DataSetNames.CIFAR10}",  # noqa: E501
    }

    def validate(self, config):
        self._validate_classifier_matches_data_set(config)
        self._validate_data_path_matches_data_set(config)
        if "name" in config:
            if config.name == "ManipulatedModelTrainingConfig":
                self._validate_manipulated_model_training_config(config)

    def _validate_classifier_matches_data_set(self, config):
        data_set_name = config.data_set.name
        classifier_name = config.classifier.name
        assert (
            classifier_name == self.DATA_SET_CLASSIFIER_MAPPING[data_set_name]
        ), self.ERROR_DATA_SET_CLASSIFIER_MISMATCH.format(
            data_set=data_set_name, classifier=classifier_name
        )

    def _validate_explainer_matches_explanations_path(self, config):
        explanations_path = config.explanations_path
        explainer_name = config.explainer.name
        assert (
            explainer_name in explanations_path
        ), self.ERROR_EXPLAINER_EXPLANATIONS_PATH_MISMATCH.format(
            explainer=explainer_name, explanations_path=explanations_path
        )

    def _validate_data_path_matches_data_set(self, config):
        if "explainer" in config:
            data_set_name = config.data_set.name
            data_path = config.data_path
            assert (
                self.DATA_SET_DATA_PATH_MATCHER[data_set_name] in data_path
            ), self.ERROR_DATA_SET_DATA_PATH_MISMATCH.format(
                data_set=data_set_name, data_path=data_path
            )

    def _validate_manipulated_model_training_config(self, config):
        if not config.test:
            assert (
                config.classifier_checkpoint != ""
            ), "classifier_checkpoint is unset for training"
        else:
            assert len(config.checkpoint) > 0, "checkpoint(s) are unset for testing"
        self._validate_explainer_matches_explanations_path(config)
