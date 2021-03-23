import logging
import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class DataConfig:
    ROOT_PATH = os.path.join(ROOT_DIR, "data")
    EXTERNAL_PATH = os.path.join(ROOT_DIR, "data/external")
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, "data/preprocessed")


class MNISTConfig:
    BOUNDS = (0, 1)
    PREPROCESSING = dict(mean=[0.1307], std=[0.3081], axis=-1)
    LOGS_PATH = os.path.join(ROOT_DIR, "logs/MNIST")
    LOG_LEVEL = logging.INFO

    # Loss names for logging
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"
    TEST_LOSS = "test_loss"
    TRAIN_ACCURACY = "train_acc"
    VAL_ACCURACY = "val_acc"
    TEST_ACCURACY = "test_acc"
    TRAIN_ACCURACY_EPOCH = "train_acc_epoch"
    VAL_ACCURACY_EPOCH = "val_acc_epoch"
    TEST_ACCURACY_EPOCH = "test_acc_epoch"

