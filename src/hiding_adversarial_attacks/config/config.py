import os

ROOT_DIR = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

NEPTUNE_PROJECT_NAME = "stefaniestoppel/hiding-adversarial-attacks"

DIRECTORIES_TO_LOG = [
    "config",
    "_neptune",
    "classifiers",
    "custom_metrics",
    "manipulation",
]
