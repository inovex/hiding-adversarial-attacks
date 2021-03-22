import sys
import os
import json


if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CONFIG_JSON = os.path.join(ROOT_DIR, "configs/config.json")

with open(CONFIG_JSON, 'r') as f:
    config = json.load(f)

DATA_PATH = os.path.join(ROOT_DIR, config["DATA_PATH"])
