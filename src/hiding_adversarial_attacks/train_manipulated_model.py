import logging
import os

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger

from hiding_adversarial_attacks._neptune.utils import get_neptune_logger
from hiding_adversarial_attacks.callbacks.neptune_callback import NeptuneLoggingCallback
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.config_validator import ConfigValidator
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.utils import (
    VisionDataModuleUnionType,
    get_data_module,
)
from hiding_adversarial_attacks.manipulated_classifiers.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)
from hiding_adversarial_attacks.utils import get_model

logger = logging.getLogger(__file__)


def get_manipulatable_model(config):
    if config.data_set.name in [
        AdversarialDataSetNames.ADVERSARIAL_MNIST,
    ]:
        classifier_model = MNISTNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedMNISTNet(classifier_model, config)
        return model
    else:
        raise SystemExit(
            f"Unknown data set specified: {config.data_set.name}. Exiting."
        )


def train(
    data_module: VisionDataModuleUnionType,
    neptune_logger: NeptuneLogger,
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
):
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    checkpoint_callback = hydra.utils.instantiate(config.checkpoint_config)

    model = get_manipulatable_model(config)
    model.to(device)

    neptune_callback = NeptuneLoggingCallback(
        log_path=config.log_path,
        image_log_path=model.image_log_path,
        trash_run=config.trash_run,
    )
    trainer = Trainer(
        gpus=config.gpus,
        logger=neptune_logger,
        callbacks=[checkpoint_callback, neptune_callback],
    )

    trainer.fit(model, train_loader, validation_loader)


def test(
    data_module,
    neptune_logger: NeptuneLogger,
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
):
    test_loader = data_module.test_dataloader()

    trainer = Trainer(gpus=config.gpus, logger=neptune_logger)

    if len(config.checkpoint) == 0 or not os.path.isfile(config.checkpoint):
        raise SystemExit(
            "---- ERROR: Please specify a valid checkpoint path. Exiting. -----"
        )
    model = get_model(config).load_from_checkpoint(config.checkpoint)

    trainer.test(model, test_loader, ckpt_path="best")


@hydra.main(config_name="manipulated_model_training_config")
def run(config: ManipulatedModelTrainingConfig) -> None:
    config_validator = ConfigValidator()
    config_validator.validate(config)

    logger.info("Starting train_manipulated_model.py")
    logger.info(f"cwd: {os.getcwd()}")
    logger.info("**** Parameters: ******")
    logger.info(OmegaConf.to_yaml(config))

    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_path,
        download=False,
        batch_size=config.batch_size,
        val_split=config.val_split,
        transform=None,
        random_seed=config.random_seed,
    )

    # GPU or CPU
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.gpus != 0) else "cpu"
    )

    experiment_name = config.data_set.name
    config.tags.append(config.data_set.name)
    if config.trash_run:
        config.tags.append("trash")
    neptune_logger = get_neptune_logger(config, experiment_name, list(config.tags))

    # Override log path
    config.log_path = os.path.join(
        config.log_path, neptune_logger.name, neptune_logger.version
    )
    os.makedirs(config.log_path, exist_ok=True)

    if config.test:
        test(data_module, neptune_logger, device, config)
    else:
        train(data_module, neptune_logger, device, config)


if __name__ == "__main__":
    run()
