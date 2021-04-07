import logging
import os
from dataclasses import dataclass

from hiding_adversarial_attacks.utils import ROOT_DIR


@dataclass
class LoggingConfig:
    log_root: str = os.path.join(ROOT_DIR, "logs")
    log_level: int = logging.INFO
