import logging
import os
import sys

from hiding_adversarial_attacks.conf.logger.logger import LoggingConfig


def setup_logger(
    logger, log_file_path: os.path, log_level: int = LoggingConfig.log_level
):
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(log_level)
    logger.addHandler(ch)

    logger.setLevel(log_level)


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
