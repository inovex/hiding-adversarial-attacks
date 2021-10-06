import os

ROOT_DIR = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

DIRECTORIES_TO_LOG = [
    "config",
    "_neptune",
    "classifiers",
    "custom_metrics",
    "manipulation",
]

NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT", None)
