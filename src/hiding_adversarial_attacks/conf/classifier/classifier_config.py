from dataclasses import dataclass


@dataclass
class ClassifierCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"
    monitor: str = "val_loss"
    filename: str = "model-{epoch:02d}-{val_loss:.2f}"
    save_top_k: int = 3
    mode: str = "min"


@dataclass
class ClassifierConfig:
    # Train or test
    test: bool = False

    # Hyperparameters
    lr: float = 1.0
    gamma: float = 0.7

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
    name: str = "MNISTClassifier"


@dataclass
class FashionMNISTClassifierConfig(ClassifierConfig):
    name: str = "FashionMNISTClassifier"
