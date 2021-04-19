import os

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.classifier_training_config import (
    ClassifierTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.utils import get_data_module
from hiding_adversarial_attacks.neptune_logger.utils import get_neptune_logger
from hiding_adversarial_attacks.utils import get_model


def train(data_module, config: ClassifierTrainingConfig):
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    checkpoint_callback = hydra.utils.instantiate(config.classifier.model_checkpoint)

    neptune_logger = get_neptune_logger(config)
    trainer = Trainer(
        gpus=config.gpus, logger=neptune_logger, callbacks=[checkpoint_callback]
    )

    model = get_model(config)

    trainer.fit(model, train_loader, validation_loader)


def test(data_module, config: ClassifierTrainingConfig):
    test_loader = data_module.test_dataloader()

    log_path = os.path.join(config.log_path, config.data_set.name)
    tb_logger = pl_loggers.TensorBoardLogger(log_path)
    trainer = Trainer(gpus=config.gpus, logger=tb_logger)

    if len(config.checkpoint) == 0 or not os.path.isfile(config.checkpoint):
        raise SystemExit(
            "---- ERROR: Please specify a valid checkpoint path. Exiting. -----"
        )
    model = get_model(config).load_from_checkpoint(config.checkpoint)

    trainer.test(model, test_loader, ckpt_path="best")


@hydra.main(config_name="classifier_training_config")
def run(config: ClassifierTrainingConfig) -> None:
    print(OmegaConf.to_yaml(config))
    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_set.external_path,
        download=config.download,
        batch_size=config.batch_size,
        val_split=config.val_split,
        transform=transforms.ToTensor(),
        random_seed=config.random_seed,
    )

    if config.test:
        test(data_module, config)
    else:
        train(data_module, config)


if __name__ == "__main__":
    run()
