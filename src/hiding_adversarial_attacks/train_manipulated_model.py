import logging
import os
from functools import partial

import hydra
import optuna
import torch
from omegaconf import OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from torch._vmap_internals import vmap

from hiding_adversarial_attacks._neptune.utils import get_neptune_logger
from hiding_adversarial_attacks.callbacks.neptune_callback import NeptuneLoggingCallback
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.config_validator import ConfigValidator
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
)
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    VAL_TOTAL_LOSS,
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.utils import (
    VisionDataModuleUnionType,
    get_data_module,
)
from hiding_adversarial_attacks.manipulated_classifiers.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)
from hiding_adversarial_attacks.manipulated_classifiers.metricized_explanations import (
    MetricizedTopAndBottomExplanations,
)
from hiding_adversarial_attacks.utils import (
    tensor_to_pil_numpy,
    visualize_difference_image_np,
    visualize_single_explanation,
)

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


def load_explanations(config, device: torch.device):
    (training_orig_expl, training_orig_labels, training_orig_indices,) = torch.load(
        os.path.join(config.explanations_path, "training_orig_exp.pt"),
        map_location=device,
    )
    training_adv_expl, training_adv_labels, training_adv_indices = torch.load(
        os.path.join(config.explanations_path, "training_adv_exp.pt"),
        map_location=device,
    )
    return (
        training_orig_expl,
        training_orig_labels,
        training_orig_indices,
        training_adv_expl,
        training_adv_labels,
        training_adv_indices,
    )


def load_attacked_data(config, device: torch.device):
    training_orig_images, training_orig_labels = torch.load(
        os.path.join(config.explanations_path, "training_orig.pt"),
        map_location=device,
    )
    training_adversarial_images, training_adversarial_labels = torch.load(
        os.path.join(config.explanations_path, "training_adv.pt"),
        map_location=device,
    )
    return (
        training_orig_images,
        training_orig_labels,
        training_adversarial_images,
        training_adversarial_labels,
    )


def filter_included_classes(
    training_adv_expl,
    training_adv_images,
    training_adv_indices,
    training_adv_labels,
    training_orig_expl,
    training_orig_images,
    training_orig_indices,
    training_orig_labels,
    config,
    device,
):
    mask = torch.zeros(len(training_orig_labels), dtype=torch.bool, device=device)
    for included_class in config.included_classes:
        mask += training_orig_labels == included_class
    training_orig_expl = training_orig_expl[mask]
    training_orig_labels = training_orig_labels[mask]
    training_orig_indices = training_orig_indices[mask]
    training_adv_expl = training_adv_expl[mask]
    training_adv_labels = training_adv_labels[mask]
    training_adv_indices = training_adv_indices[mask]
    training_orig_images = training_orig_images[mask]
    training_adv_images = training_adv_images[mask]
    return (
        training_adv_expl,
        training_adv_images,
        training_adv_labels,
        training_orig_expl,
        training_orig_images,
        training_orig_labels,
    )


def get_metricized_top_and_bottom_explanations(
    config: ManipulatedModelTrainingConfig, device: torch.device
) -> MetricizedTopAndBottomExplanations:
    (
        training_orig_expl,
        training_orig_labels,
        training_orig_indices,
        training_adv_expl,
        training_adv_labels,
        training_adv_indices,
    ) = load_explanations(config, device)

    (
        training_orig_images,
        _,
        training_adv_images,
        _,
    ) = load_attacked_data(config, device)

    # filter attacked data by included_classes
    if ALL_CLASSES not in config.included_classes:
        (
            training_adv_expl,
            training_adv_images,
            training_adv_labels,
            training_orig_expl,
            training_orig_images,
            training_orig_labels,
        ) = filter_included_classes(
            training_adv_expl,
            training_adv_images,
            training_adv_indices,
            training_adv_labels,
            training_orig_expl,
            training_orig_images,
            training_orig_indices,
            training_orig_labels,
            config,
            device,
        )

    similarity_loss = SimilarityLossMapping[config.similarity_loss.name]
    batched_sim_loss = vmap(similarity_loss)
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
        training_adv_expl,
        training_orig_expl,
        batched_sim_loss,
    )

    train_img_top = tensor_to_pil_numpy(training_orig_images[top_indices])
    train_expl_top = tensor_to_pil_numpy(top_orig_expl)
    train_adv_top = tensor_to_pil_numpy(training_adv_images[top_indices])
    train_adv_expl_top = tensor_to_pil_numpy(top_adv_expl)

    # Visualize explanations
    visualize_single_explanation(
        train_img_top[0],
        train_expl_top[0],
        f"Orig label: {training_orig_labels[top_indices][0]}",
        display_figure=True,
    )
    visualize_single_explanation(
        train_adv_top[0],
        train_adv_expl_top[0],
        f"Adv label: {training_adv_labels[top_indices][0]}",
        display_figure=True,
    )
    # Visualize difference images
    visualize_difference_image_np(
        train_adv_expl_top[0],
        train_expl_top[0],
        title="Explanation diff: adv vs. orig",
    )
    visualize_difference_image_np(
        train_img_top[0], train_adv_top[0], title="Image diff: adv vs. orig"
    )

    metricized_top_and_bottom_explanations = MetricizedTopAndBottomExplanations(
        device=device,
        sorted_by=config.similarity_loss.name,
        top_k_indices=top_indices,
        bottom_k_indices=bottom_indices,
        top_k_original_images=training_orig_images[top_indices],
        top_k_original_explanations=top_orig_expl,
        top_k_original_labels=training_orig_labels[top_indices].long(),
        top_k_adversarial_images=training_adv_images[top_indices],
        top_k_adversarial_explanations=top_adv_expl,
        top_k_adversarial_labels=training_adv_labels[top_indices].long(),
        bottom_k_original_images=training_orig_images[bottom_indices],
        bottom_k_original_explanations=bottom_orig_expl,
        bottom_k_original_labels=training_orig_labels[bottom_indices].long(),
        bottom_k_adversarial_images=training_adv_images[bottom_indices],
        bottom_k_adversarial_explanations=bottom_adv_expl,
        bottom_k_adversarial_labels=training_adv_labels[bottom_indices].long(),
    )
    return metricized_top_and_bottom_explanations


def get_top_and_bottom_k_explanations(
    training_adv_expl,
    training_orig_expl,
    batched_sim_loss,
):
    similarity_results = batched_sim_loss(training_orig_expl, training_adv_expl)
    # largest similarity
    bottom_similarities, _b_indices = torch.topk(similarity_results, 4)
    bottom_similarities, bottom_indices = (
        torch.flip(bottom_similarities, dims=(0,)),
        torch.flip(_b_indices, dims=(0,)).long(),
    )
    # smallest similarity
    top_similarities, _t_indices = torch.topk(similarity_results, 4, largest=False)
    top_indices = _t_indices.long()
    return (
        training_orig_expl[top_indices],
        training_adv_expl[top_indices],
        top_similarities,
        top_indices,
        training_orig_expl[bottom_indices],
        training_adv_expl[bottom_indices],
        bottom_similarities,
        bottom_indices,
    )


def suggest_hyperparameters(config, trial):
    lr_options = config.optuna.search_space["lr"]
    lr = trial.suggest_float(
        "lr", lr_options["low"], lr_options["high"], log=lr_options["log"]
    )
    # similarity_loss = trial.suggest_categorical(
    #     "similarity_loss",
    #     config.optuna.search_space["similarity_loss"]["choices"],
    # )
    loss_weight_orig_ce_options = config.optuna.search_space["loss_weight_orig_ce"]
    loss_weight_orig_ce = trial.suggest_float(
        "loss_weight_orig_ce",
        loss_weight_orig_ce_options["low"],
        loss_weight_orig_ce_options["high"],
        step=loss_weight_orig_ce_options["step"],
    )
    # note: both adv and original cross entropy loss weights should be the same
    # -> it makes no sense to prioritize the one over the other
    loss_weight_adv_ce = loss_weight_orig_ce
    loss_weight_similarity_options = config.optuna.search_space[
        "loss_weight_similarity"
    ]
    loss_weight_similarity = trial.suggest_float(
        "loss_weight_similarity",
        loss_weight_similarity_options["low"],
        loss_weight_similarity_options["high"],
        log=loss_weight_similarity_options["log"],
    )
    return loss_weight_orig_ce, loss_weight_adv_ce, loss_weight_similarity, lr


def train(
    data_module: VisionDataModuleUnionType,
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
    metricized_top_and_bottom_explanations: MetricizedTopAndBottomExplanations,
    original_log_path: str,
    trial: optuna.trial.Trial = None,
) -> float:

    experiment_name = config.data_set.name

    # Setup logger
    neptune_logger = get_neptune_logger(config, experiment_name, list(config.tags))

    # Override log path
    config.log_path = os.path.join(
        original_log_path, neptune_logger.name, neptune_logger.version
    )
    os.makedirs(config.log_path, exist_ok=True)

    logger.info(
        f"Starting new neptune run '{neptune_logger.version}' "
        f"with trial no. '{trial.number}'"
    )

    # Hyperparameter suggestions by Optuna => override hyperparams in config
    if trial is not None:
        (
            config.loss_weight_orig_ce,
            config.loss_weight_adv_ce,
            config.loss_weight_similarity,
            config.lr,
        ) = suggest_hyperparameters(config, trial)

        logger.info("Updated Hyperparameters:")
        logger.info(f"\t lr: {config.lr}")
        logger.info(f"\t loss_weight_orig_ce: {config.loss_weight_orig_ce}")
        logger.info(f"\t loss_weight_adv_ce: {config.loss_weight_adv_ce}")
        logger.info(f"\t loss_weight_similarity: {config.loss_weight_similarity}")

    # Data loaders
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    checkpoint_callback = hydra.utils.instantiate(config.checkpoint_config)

    model = get_manipulatable_model(config)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)
    model.set_hydra_logger(logger)
    model.to(device)

    # PyTorch Lightning Callbacks
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
            PyTorchLightningPruningCallback(trial, monitor=VAL_TOTAL_LOSS),
        )
    trainer = Trainer(
        callbacks=callbacks,
        gpus=config.gpus,
        logger=neptune_logger,
        max_epochs=config.max_epochs,
    )

    trainer.fit(model, train_loader, validation_loader)

    # Test
    model.eval()
    test_loader = data_module.test_dataloader()
    trainer.test(model, test_loader)

    return trainer.callback_metrics["val_normalized_total_loss"].item()


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

    model = get_manipulatable_model(config).load_from_checkpoint(config.checkpoint)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)

    trainer.test(model, test_loader, ckpt_path="best")


def run_optuna_study(
    data_module: VisionDataModuleUnionType,
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
        data_module,
        device,
        config,
        metricized_top_and_bottom_explanations,
        original_log_path,
    )
    study.optimize(
        objective,
        n_trials=config.optuna.number_of_trials,
        timeout=config.optuna.timeout,
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
        included_classes=list(config.included_classes),
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
    config.tags.append("test" if config.test else "train")
    if config.trash_run:
        config.tags.append("trash")

    if config.test:
        test(
            data_module,
            device,
            config,
            metricized_top_and_bottom_explanations,
        )
    else:
        if config.optuna.use_optuna:
            run_optuna_study(
                data_module,
                device,
                config,
                metricized_top_and_bottom_explanations,
                original_log_path,
            )
        else:
            train(
                data_module,
                device,
                config,
                metricized_top_and_bottom_explanations,
                original_log_path,
            )


if __name__ == "__main__":
    run()
