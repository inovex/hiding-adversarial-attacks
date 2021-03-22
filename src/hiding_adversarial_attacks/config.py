import json
import os

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIG_JSON = os.path.join(ROOT_DIR, "configs/config.json")

with open(CONFIG_JSON, 'r') as f:
    config = json.load(f)

DATA_PATH = os.path.join(ROOT_DIR, config["DATA_PATH"])
MNIST_PATH = os.path.join(DATA_PATH, "MNIST")
