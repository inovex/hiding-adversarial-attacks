import logging
import os
from functools import partial
from pprint import pformat

import hydra
import optuna
from omegaconf import OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from pytorch_lightning import Trainer

from hiding_adversarial_attacks._neptune.utils import get_neptune_logger
from hiding_adversarial_attacks.callbacks.neptune_callback import NeptuneLoggingCallback
from hiding_adversarial_attacks.callbacks.utils import copy_run_outputs
from hiding_adversarial_attacks.classifiers.utils import (
    convert_relu_to_softplus,
    get_model,
)
from hiding_adversarial_attacks.config.classifier_training_config import (
    ClassifierTrainingConfig,
)
from hiding_adversarial_attacks.config.config_validator import ConfigValidator
from hiding_adversarial_attacks.data_modules.utils import (
    VisionDataModuleUnionType,
    get_data_module,
)
from hiding_adversarial_attacks.data_sets.utils import get_transform

logger = logging.getLogger(__file__)


def suggest_hyperparameters(config, trial):
    lr_options = config.optuna.search_space["lr"]
    lr = trial.suggest_float(
        "lr", lr_options["low"], lr_options["high"], log=lr_options["log"]
    )
    batch_size = trial.suggest_categorical(
        "batch_size", config.optuna.search_space["batch_size"]
    )
    weight_decay = trial.suggest_categorical(
        "weight_decay", config.optuna.search_space["weight_decay"]
    )

    return (
        lr,
        batch_size,
        weight_decay,
    )


def train(
    data_module: VisionDataModuleUnionType,
    config: ClassifierTrainingConfig,
    original_log_path: str,
    trial: optuna.trial.Trial = None,
):
    # Hyperparameter suggestions by Optuna => override hyperparams in config
    if trial is not None:
        (
            config.lr,
            config.batch_size,
            config.weight_decay,
        ) = suggest_hyperparameters(config, trial)

    logger.info("**** Parameters: ******")
    logger.info(OmegaConf.to_yaml(config))

    # Setup logger
    experiment_name = config.data_set.name
    neptune_logger = get_neptune_logger(config, experiment_name, list(config.tags))

    # Override log path
    config.log_path = os.path.join(
        original_log_path, neptune_logger.name, neptune_logger.version
    )
    os.makedirs(config.log_path, exist_ok=True)

    # Data loaders
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    # PyTorch Lightning Callbacks
    checkpoint_callback = hydra.utils.instantiate(config.classifier.model_checkpoint)
    neptune_callback = NeptuneLoggingCallback(
        log_path=config.log_path,
        image_log_path=os.path.join(config.log_path, "image_log"),
        trash_run=config.trash_run,
    )
    callbacks = [
        checkpoint_callback,
        neptune_callback,
    ]
    if trial is not None:
        callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        )
    resume_from_checkpoint = None
    if config.resume_from_checkpoint:
        resume_from_checkpoint = config.checkpoint
    trainer = Trainer(
        callbacks=callbacks,
        gpus=config.gpus,
        logger=neptune_logger,
        max_epochs=config.max_epochs,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    model = get_model(config)
    if config.convert_to_softplus:
        convert_relu_to_softplus(
            model,
            config,
            beta=config.soft_plus_beta,
            threshold=config.soft_plus_threshold,
        )

    trainer.fit(model, train_loader, validation_loader)

    return trainer.callback_metrics["val_loss"].item()


def test(
    data_module,
    config: ClassifierTrainingConfig,
    original_log_path: str,
):
    experiment_name = config.data_set.name

    # Setup logger
    neptune_logger = get_neptune_logger(config, experiment_name, list(config.tags))

    # Override log path
    config.log_path = os.path.join(
        original_log_path, neptune_logger.name, neptune_logger.version
    )
    os.makedirs(config.log_path, exist_ok=True)

    # data loader
    test_loader = data_module.test_dataloader()

    trainer = Trainer(gpus=config.gpus, logger=neptune_logger)

    if len(config.checkpoint) == 0 or not os.path.isfile(config.checkpoint):
        raise SystemExit(
            "---- ERROR: Please specify a valid checkpoint path. Exiting. -----"
        )

    model = get_model(config).load_from_checkpoint(config.checkpoint)
    if config.convert_to_softplus:
        convert_relu_to_softplus(
            model,
            config,
            beta=config.soft_plus_beta,
            threshold=config.soft_plus_threshold,
        )
    model.eval()

    test_results = trainer.test(model, test_loader, ckpt_path="best")
    logger.info(f"Test results: \n {pformat(test_results)}")
    copy_run_outputs(
        config.log_path,
        os.getcwd(),
        neptune_logger.name,
        neptune_logger.version,
    )


def run_optuna_study(
    data_module: VisionDataModuleUnionType,
    config: ClassifierTrainingConfig,
    original_log_path: str,
):
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
        if config.optuna.prune_trials
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)
    objective = partial(
        train,
        data_module,
        config,
        original_log_path,
    )
    study.optimize(
        objective,
        n_trials=config.optuna.number_of_trials,
        timeout=config.optuna.timeout,
        gc_after_trial=True,
    )
    logger.info("\n************ Optuna trial results ***************")
    logger.info("Number of finished trials: {}".format(len(study.trials)))
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info("  Number: {}".format(best_trial.number))
    logger.info("  Value: {}".format(best_trial.value))
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # Visualize results of Optuna study
    hist_fig = plot_optimization_history(study)
    contour_fig = plot_contour(study)
    param_imp_fig = plot_param_importances(study)
    parallel_coord_fig = plot_parallel_coordinate(study)
    hist_fig.show()
    contour_fig.show()
    param_imp_fig.show()
    parallel_coord_fig.show()


@hydra.main(config_name="classifier_training_config")
def run(config: ClassifierTrainingConfig) -> None:
    config_validator = ConfigValidator()
    config_validator.validate(config)

    logger.info("Starting train_classifier.py")
    logger.info(f"cwd: {os.getcwd()}")
    logger.info("**** Parameters: ******")
    logger.info(OmegaConf.to_yaml(config))

    transform = get_transform(config.data_set.name, data_is_tensor=False)

    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_set.external_path,
        download=config.download,
        batch_size=config.batch_size,
        val_split=config.val_split,
        transform=transform,
        random_seed=config.random_seed,
    )

    config.tags.append(config.data_set.name)
    config.tags.append("test" if config.test else "train")
    if config.trash_run:
        config.tags.append("trash")

    # Needs to be saved here as it gets overwritten every run
    original_log_path = config.log_path

    if config.test:
        test(data_module, config, original_log_path)
    else:
        if config.optuna.use_optuna:
            run_optuna_study(data_module, config, original_log_path)
        else:
            train(data_module, config, original_log_path)


if __name__ == "__main__":
    run()
