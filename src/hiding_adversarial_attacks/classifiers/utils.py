import torch

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


def get_model_from_checkpoint(
    data_set_name: str, model_checkpoint: str, device: torch.device
):
    if DataSetNames.MNIST in data_set_name:
        model = MNISTNet.load_from_checkpoint(checkpoint_path=model_checkpoint)
    elif DataSetNames.CIFAR10 in data_set_name:
        model = CifarNet.load_from_checkpoint(checkpoint_path=model_checkpoint)
    else:
        raise SystemExit(f"ERROR: Unknown data set name: {data_set_name}. Exiting.")
    model = model.to(device)
    return model


def _get_conv2d_layer_by_name(model, layer_name: str):
    named_modules = dict(model.named_modules())
    assert layer_name in named_modules, f"Layer name '{layer_name}' not in model."
    assert (
        type(named_modules[layer_name]) == torch.nn.modules.conv.Conv2d
    ), f"Specified layer '{layer_name}' is not of type Conv2d."
    return named_modules[layer_name]
