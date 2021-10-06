from dataclasses import dataclass

from omegaconf import MISSING


class ClassifierNames:
    MNIST_CLASSIFIER: str = "MNISTClassifier"
    FASHION_MNIST_CLASSIFIER: str = "FashionMNISTClassifier"
    CIFAR10_CLASSIFIER: str = "Cifar10Classifier"


@dataclass
class ClassifierCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"
    monitor: str = "val_loss"
    filename: str = "model-{epoch:02d}-{val_loss:.2f}"
    save_top_k: int = 1
    mode: str = "min"


@dataclass
class ClassifierConfig:
    name: str = MISSING

    # Loss names for logging
    train_loss: str = "train_loss"
    val_loss: str = "val_loss"
    test_loss: str = "test_loss"
    train_accuracy: str = "train_acc"
    val_accuracy: str = "val_acc"
    test_accuracy: str = "test_acc"
    train_accuracy_epoch: str = "train_acc_epoch"
    val_accuracy_epoch: str = "val_acc_epoch"
    test_accuracy_epoch: str = "test_acc_epoch"

    # Checkpoint
    model_checkpoint: ClassifierCheckpointConfig = ClassifierCheckpointConfig()


@dataclass
class MNISTClassifierConfig(ClassifierConfig):
    name: str = ClassifierNames.MNIST_CLASSIFIER


@dataclass
class FashionMNISTClassifierConfig(ClassifierConfig):
    name: str = ClassifierNames.FASHION_MNIST_CLASSIFIER


@dataclass
class Cifar10ClassifierConfig(ClassifierConfig):
    name: str = ClassifierNames.CIFAR10_CLASSIFIER
