import logging
import os

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from torchvision.transforms import transforms

from hiding_adversarial_attacks._neptune.utils import get_neptune_logger
from hiding_adversarial_attacks.callbacks.neptune_callback import NeptuneLoggingCallback
from hiding_adversarial_attacks.config.classifier_training_config import (
    ClassifierTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.utils import (
    VisionDataModuleUnionType,
    get_data_module,
)
from hiding_adversarial_attacks.utils import get_model

logger = logging.getLogger(__file__)


def train(
    data_module: VisionDataModuleUnionType,
    neptune_logger: NeptuneLogger,
    config: ClassifierTrainingConfig,
):
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    checkpoint_callback = hydra.utils.instantiate(config.classifier.model_checkpoint)

    neptune_callback = NeptuneLoggingCallback(trash_run=config.trash_run)
    trainer = Trainer(
        gpus=config.gpus,
        logger=neptune_logger,
        callbacks=[checkpoint_callback, neptune_callback],
    )

    model = get_model(config)

    trainer.fit(model, train_loader, validation_loader)


def test(data_module, neptune_logger: NeptuneLogger, config: ClassifierTrainingConfig):
    test_loader = data_module.test_dataloader()

    trainer = Trainer(gpus=config.gpus, logger=neptune_logger)

    if len(config.checkpoint) == 0 or not os.path.isfile(config.checkpoint):
        raise SystemExit(
            "---- ERROR: Please specify a valid checkpoint path. Exiting. -----"
        )
    model = get_model(config).load_from_checkpoint(config.checkpoint)

    trainer.test(model, test_loader, ckpt_path="best")


@hydra.main(config_name="classifier_training_config")
def run(config: ClassifierTrainingConfig) -> None:
    logger.info("Starting train_classifier.py")
    logger.info(f"cwd: {os.getcwd()}")
    logger.info("**** Parameters: ******")
    logger.info(OmegaConf.to_yaml(config))

    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_set.external_path,
        download=config.download,
        batch_size=config.batch_size,
        val_split=config.val_split,
        transform=transforms.ToTensor(),
        random_seed=config.random_seed,
    )

    experiment_name = (
        f"{'test' if config.test else 'train'}-{config.data_set.name}-classifier"
    )
    tags = [*config.tags, config.data_set.name]
    if config.trash_run:
        tags.append("trash")
    neptune_logger = get_neptune_logger(config, experiment_name, tags)

    if config.test:
        test(data_module, neptune_logger, config)
    else:
        train(data_module, neptune_logger, config)


if __name__ == "__main__":
    run()
