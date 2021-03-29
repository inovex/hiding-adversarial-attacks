import logging
import os
import sys
from datetime import datetime

from hiding_adversarial_attacks.config import AdversarialAttackConfig


def setup_logger(logger, args):
    logs_path = os.path.join(AdversarialAttackConfig.LOGS_PATH, "MNIST")
    os.makedirs(logs_path, exist_ok=True)
    timestamp_seconds = int(datetime.now().timestamp())
    log_file = f"{str(timestamp_seconds)}-mnist-{args.attack}-{args.epsilons}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    fh = logging.FileHandler(os.path.join(logs_path, log_file))
    fh.setLevel(AdversarialAttackConfig.LOG_LEVEL)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(AdversarialAttackConfig.LOG_LEVEL)
    logger.addHandler(ch)

    logger.setLevel(AdversarialAttackConfig.LOG_LEVEL)


def log_attack_results(logger, attack_results, set_size):
    logger.info(f"\t\t Total set size: {set_size}")
    logger.info(f"\t\t Attacked image count: {attack_results.attacked_count}")
    logger.info(f"\t\t Adversarial count: {attack_results.adv_count}")
    logger.info(
        f"\t\t Percentage successfully attacked (adv_count / attacked_count): "
        f"{attack_results.adv_count / attack_results.attacked_count:.4%}%"
    )
    logger.info(
        f"\t\t Percentage adversarials of total set size (adv_count / set_size): "
        f"{attack_results.adv_count / set_size:.4%}%"
    )


def log_attack_info(logger, attack, epsilons, data_set, checkpoint, attacked_classes):
    logger.info("******* Attack info *******")
    logger.info(f"Attack type: {attack}")
    logger.info(f"Epsilon(s): {epsilons}")
    logger.info(f"Data set: {data_set}")
    logger.info(f"Model checkpoint: '{checkpoint}'")
    logger.info(f"Attacked classes: {attacked_classes}")
