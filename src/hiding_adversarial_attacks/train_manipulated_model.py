import glob
import logging
import os
import shutil
from functools import partial
from pprint import pformat
from typing import Tuple

import hydra
import optuna
import torch
from omegaconf import OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from torch._vmap_internals import vmap
from torch.utils.data import DataLoader

from hiding_adversarial_attacks._neptune.utils import get_neptune_logger
from hiding_adversarial_attacks.callbacks.neptune_callback import NeptuneLoggingCallback
from hiding_adversarial_attacks.callbacks.utils import copy_run_outputs
from hiding_adversarial_attacks.classifiers.cifar_net import CifarNet
from hiding_adversarial_attacks.classifiers.fashion_mnist_net import FashionMNISTNet
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.config_validator import ConfigValidator
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
)
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    VAL_NORM_TOTAL_LOSS,
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.k_fold_cross_validation import (
    StratifiedKFoldCVDataModule,
)
from hiding_adversarial_attacks.data_modules.utils import get_data_module
from hiding_adversarial_attacks.manipulation.manipulated_cifar_net import (
    ManipulatedCIFARNet,
)
from hiding_adversarial_attacks.manipulation.manipulated_fashion_mnist_net import (  # noqa: E501
    ManipulatedFashionMNISTNet,
)
from hiding_adversarial_attacks.manipulation.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)
from hiding_adversarial_attacks.manipulation.metricized_explanations import (
    MetricizedTopAndBottomExplanations,
)
from hiding_adversarial_attacks.manipulation.utils import (
    get_metricized_top_and_bottom_explanations,
    get_top_and_bottom_k_explanations,
    load_filtered_data,
)

logger = logging.getLogger(__file__)


def get_manipulatable_model(config):
    if config.data_set.name == AdversarialDataSetNames.ADVERSARIAL_MNIST:
        classifier_model = MNISTNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedMNISTNet(classifier_model, config)
        return model
    if config.data_set.name == AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST:
        classifier_model = FashionMNISTNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedFashionMNISTNet(classifier_model, config)
        return model
    if config.data_set.name == AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST_EXPL:
        classifier_model = FashionMNISTNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedFashionMNISTNet(classifier_model, config)
        return model
    if config.data_set.name == AdversarialDataSetNames.ADVERSARIAL_CIFAR10:
        classifier_model = CifarNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedCIFARNet(classifier_model, config)
        return model
    else:
        raise SystemExit(
            f"Unknown data set specified: {config.data_set.name}. Exiting."
        )


def visualize_top_bottom_k(config, device, model):
    similarity_loss = SimilarityLossMapping[config.similarity_loss.name]
    batched_sim_loss = vmap(similarity_loss)
    (
        test_orig_images,
        _,
        test_orig_labels,
        test_adv_images,
        _,
        test_adv_labels,
    ) = load_filtered_data(config, device, stage="test")
    (
        top_orig_expl,
        top_adv_expl,
        top_similarities,
        top_indices,
        bottom_orig_expl,
        bottom_adv_expl,
        bottom_similarities,
        bottom_indices,
    ) = get_top_and_bottom_k_explanations(
        model.test_adv_explanations,
        model.test_orig_explanations,
        batched_sim_loss,
    )
    test_orig_expl = torch.cat((top_orig_expl, bottom_orig_expl), dim=0)
    test_adv_expl = torch.cat((top_adv_expl, bottom_adv_expl), dim=0)
    indeces = torch.cat((top_indices, bottom_indices), dim=0)
    model._visualize_batch_explanations(
        test_adv_expl,
        test_adv_images[indeces],
        test_adv_labels[indeces].cpu(),
        test_orig_expl,
        test_orig_images[indeces],
        test_orig_labels[indeces].cpu(),
        "test-top-bottom-k-explanations.png",
    )


def suggest_hyperparameters(config, trial):
    lr_options = config.optuna.search_space["lr"]
    lr = trial.suggest_float(
        "lr", lr_options["low"], lr_options["high"], log=lr_options["log"]
    )
    loss_weight_similarity_options = config.optuna.search_space[
        "loss_weight_similarity"
    ]
    loss_weight_similarity = trial.suggest_float(
        "loss_weight_similarity",
        loss_weight_similarity_options["low"],
        loss_weight_similarity_options["high"],
        step=loss_weight_similarity_options["step"],
    )
    # # note: both adv and original cross entropy loss weights should be the same
    # -> it makes no sense to prioritize the one over the other
    loss_weight_orig_ce = 1
    loss_weight_adv_ce = 1
    batch_size = trial.suggest_categorical(
        "batch_size", config.optuna.search_space["batch_size"]
    )

    steps_lr = config.steps_lr
    gamma = config.gamma
    if (
        "steps_lr" in config.optuna.search_space
        and "gamma" in config.optuna.search_space
    ):
        steps_lr_options = config.optuna.search_space["steps_lr"]
        steps_lr = trial.suggest_int(
            "steps_lr",
            steps_lr_options["low"],
            steps_lr_options["high"],
            step=steps_lr_options["step"],
        )
        gamma_options = config.optuna.search_space["gamma"]
        gamma = trial.suggest_int(
            "gamma",
            gamma_options["low"],
            gamma_options["high"],
            step=gamma_options["step"],
        )

    return (
        loss_weight_orig_ce,
        loss_weight_adv_ce,
        loss_weight_similarity,
        lr,
        batch_size,
        steps_lr,
        gamma,
    )


def train(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
    metricized_top_and_bottom_explanations: MetricizedTopAndBottomExplanations,
    original_log_path: str,
    trial: optuna.trial.Trial = None,
) -> Tuple[float]:

    experiment_name = config.data_set.name

    # Hyperparameter suggestions by Optuna => override hyperparams in config
    if trial is not None:
        (
            config.loss_weight_orig_ce,
            config.loss_weight_adv_ce,
            config.loss_weight_similarity,
            config.lr,
            config.batch_size,
            config.steps_lr,
            config.gamma,
        ) = suggest_hyperparameters(config, trial)

    logger.info("**** Parameters: ******")
    logger.info(OmegaConf.to_yaml(config))

    # Setup logger
    neptune_logger = get_neptune_logger(config, experiment_name, list(config.tags))

    # Override log path
    config.log_path = os.path.join(
        original_log_path, neptune_logger.name, neptune_logger.version
    )
    os.makedirs(config.log_path, exist_ok=True)

    log_message = f"Starting new neptune run '{neptune_logger.version}' "
    if trial is not None:
        log_message += f"with trial no. '{trial.number}'"
    logger.info(log_message)

    loss = run_training(
        device,
        config,
        metricized_top_and_bottom_explanations,
        neptune_logger,
        train_loader,
        validation_loader,
        trial,
    )
    return loss


def run_training(
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
    metricized_top_and_bottom_explanations: MetricizedTopAndBottomExplanations,
    neptune_logger: NeptuneLogger,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    trial: optuna.trial.Trial = None,
):
    model = get_manipulatable_model(config)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)
    model.set_hydra_logger(logger)
    model.to(device)

    # PyTorch Lightning Callbacks
    checkpoint_callback = hydra.utils.instantiate(config.checkpoint_config)
    neptune_callback = NeptuneLoggingCallback(
        log_path=config.log_path,
        image_log_path=model.image_log_path,
        trash_run=config.trash_run,
    )
    callbacks = [
        checkpoint_callback,
        neptune_callback,
    ]
    if trial is not None:
        callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor=VAL_NORM_TOTAL_LOSS),
        )
    trainer = Trainer(
        callbacks=callbacks,
        gpus=config.gpus,
        logger=neptune_logger,
        max_epochs=config.max_epochs,
    )
    trainer.fit(model, train_loader, validation_loader)

    # Test with best model checkpoint (Lightning does this automatically)
    copy_run_outputs(
        config.log_path,
        os.getcwd(),
        neptune_logger.name,
        neptune_logger.version,
    )
    del model
    del train_loader
    del validation_loader

    return trainer.callback_metrics["val_exp_sim"].item()


def test(
    data_module,
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
    metricized_top_and_bottom_explanations: MetricizedTopAndBottomExplanations,
):
    experiment_name = config.data_set.name

    # Setup logger
    neptune_logger = get_neptune_logger(config, experiment_name, list(config.tags))

    # Override log path
    config.log_path = os.path.join(
        config.log_path, neptune_logger.name, neptune_logger.version
    )
    os.makedirs(config.log_path, exist_ok=True)

    test_loader = data_module.test_dataloader()

    trainer = Trainer(gpus=config.gpus, logger=neptune_logger)

    # Pre-test: test model before adversarial manipulation
    model = get_manipulatable_model(config)
    model.override_hparams(config)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)
    model.to(device)
    model.eval()
    test_results = trainer.test(model, test_loader)
    logger.info(f"Pre-test results before manipulation: \n {pformat(test_results)}")
    # Rename image log files so that they are not overwritten by second model run
    for image_path in glob.glob(os.path.join(config.log_path, "image_log", "*.png")):
        new_image_path = (
            f"{os.path.dirname(image_path)}/pre-{os.path.basename(image_path)}"
        )
        shutil.move(image_path, new_image_path)

    # Test: test model after manipulation
    model = get_manipulatable_model(config).load_from_checkpoint(config.checkpoint)
    model.override_hparams(config)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)
    model.to(device)
    model.eval()
    test_results = trainer.test(model, test_loader)
    logger.info(f"Test results: \n {pformat(test_results)}")

    # Visualize top and bottom k explanations after manipulation
    visualize_top_bottom_k(config, device, model)

    copy_run_outputs(
        config.log_path,
        os.getcwd(),
        neptune_logger.name,
        neptune_logger.version,
    )


def run_optuna_study(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
    metricized_top_and_bottom_explanations: MetricizedTopAndBottomExplanations,
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
        train_loader,
        validation_loader,
        device,
        config,
        metricized_top_and_bottom_explanations,
        original_log_path,
    )
    study.optimize(
        objective,
        n_trials=config.optuna.number_of_trials,
        timeout=config.optuna.timeout,
        gc_after_trial=True,
        catch=(KeyboardInterrupt,),
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


@hydra.main(config_name="manipulated_model_training_config")
def run(config: ManipulatedModelTrainingConfig) -> None:
    config_validator = ConfigValidator()
    config_validator.validate(config)

    logger.info("Starting train_manipulated_model.py")
    logger.info(f"cwd: {os.getcwd()}")

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

    metricized_top_and_bottom_explanations = get_metricized_top_and_bottom_explanations(
        config, device
    )

    # Needs to be saved here as it gets overwritten every run
    original_log_path = config.log_path

    # Update tags
    config.tags.append(config.data_set.name)
    config.tags.append(config.explainer.name)
    config.tags.append(config.similarity_loss.name)
    config.tags.append("test" if config.test else "train")
    config.tags.append(f"class_ids={config.included_classes}")
    if config.trash_run:
        config.tags.append("trash")
    if config.kfold_num_folds is not None:
        config.tags.append(f"kfold={config.kfold_num_folds}")

    if config.test:
        test(
            data_module,
            device,
            config,
            metricized_top_and_bottom_explanations,
        )
    else:
        if config.kfold_num_folds is not None:
            logger.info("Starting k-fold cross validation.")
            # K-fold cross validation data module
            kfold_data_module = StratifiedKFoldCVDataModule(
                data_module,
                n_splits=config.kfold_num_folds,
            )
            for fold_idx, (train_loader, validation_loader) in enumerate(
                kfold_data_module.split()
            ):
                logger.info(
                    f"Starting fold {fold_idx + 1} of {config.kfold_num_folds}."
                )
                if config.optuna.use_optuna:
                    run_optuna_study(
                        train_loader,
                        validation_loader,
                        device,
                        config,
                        metricized_top_and_bottom_explanations,
                        original_log_path,
                    )
                else:
                    train(
                        train_loader,
                        validation_loader,
                        device,
                        config,
                        metricized_top_and_bottom_explanations,
                        original_log_path,
                    )
        else:
            train_loader = data_module.train_dataloader()
            validation_loader = data_module.val_dataloader()
            if config.optuna.use_optuna:
                run_optuna_study(
                    train_loader,
                    validation_loader,
                    device,
                    config,
                    metricized_top_and_bottom_explanations,
                    original_log_path,
                )
            else:
                train(
                    train_loader,
                    validation_loader,
                    device,
                    config,
                    metricized_top_and_bottom_explanations,
                    original_log_path,
                )


if __name__ == "__main__":
    run()
