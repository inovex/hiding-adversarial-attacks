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

