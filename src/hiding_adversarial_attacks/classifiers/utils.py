from hiding_adversarial_attacks.classifiers.cifar_net import CifarNet
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames


def get_model(config):
    if config.data_set.name in [DataSetNames.MNIST, DataSetNames.FASHION_MNIST]:
        return MNISTNet(config)
    elif config.data_set.name == DataSetNames.CIFAR10:
        model = CifarNet(config)
        return model
    else:
        raise SystemExit(
            f"Unknown data set specified: {config.data_set.name}. Exiting."
        )
