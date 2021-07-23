import torch
from torch import nn

from hiding_adversarial_attacks.classifiers.cifar_net import CifarNet
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    ClassifierNames,
)
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


def _convert_relu_to_softplus(model, beta=30, threshold=30):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta=beta, threshold=threshold))
        else:
            _convert_relu_to_softplus(child, beta, threshold)


def convert_relu_to_softplus(model, config, beta=30, threshold=30):
    if config.classifier.name in [
        ClassifierNames.FASHION_MNIST_CLASSIFIER,
        ClassifierNames.MNIST_CLASSIFIER,
    ]:
        model.model.relu1 = model.model.softplus1
        model.model.relu2 = model.model.softplus2
        model.model.relu3 = model.model.softplus3
    else:
        _convert_relu_to_softplus(model, beta=beta, threshold=threshold)


def convert_softplus_to_relu(model, config):
    if config.classifier.name in [
        ClassifierNames.FASHION_MNIST_CLASSIFIER,
        ClassifierNames.MNIST_CLASSIFIER,
    ]:
        model.model.softplus1 = model.model.relu1
        model.model.softplus2 = model.model.relu2
        model.model.softplus3 = model.model.relu3
    else:
        _convert_softplus_to_relu(model)


def _convert_softplus_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Softplus):
            setattr(model, child_name, nn.ReLU())
        else:
            _convert_softplus_to_relu(child)
