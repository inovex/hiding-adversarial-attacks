import glob
import logging
import os
import re
import shutil
from functools import partial
from pprint import pformat
from typing import Tuple

import hydra
import optuna
import pandas as pd
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader

from hiding_adversarial_attacks._neptune.utils import (
    get_neptune_logger,
    init_current_neptune_run,
    log_code,
    save_run_data,
)
from hiding_adversarial_attacks.callbacks.neptune_callback import NeptuneLoggingCallback
from hiding_adversarial_attacks.callbacks.utils import copy_run_outputs
from hiding_adversarial_attacks.classifiers.utils import (
    convert_relu_to_softplus,
    convert_softplus_to_relu,
)
from hiding_adversarial_attacks.config.config_validator import ConfigValidator
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossNames,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    VAL_NORM_TOTAL_LOSS,
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.k_fold_cross_validation import (
    StratifiedKFoldCVDataModule,
)
from hiding_adversarial_attacks.data_modules.utils import get_data_module
from hiding_adversarial_attacks.evaluation.box_plot_similarities import (
    plot_pre_and_post_manipulation_boxplot_similarities,
)
from hiding_adversarial_attacks.manipulation.metricized_explanations import (
    MetricizedTopAndBottomExplanations,
)
from hiding_adversarial_attacks.manipulation.utils import (
    get_manipulatable_model,
    get_metricized_top_and_bottom_explanations,
    get_similarities,
    get_top_and_bottom_k_indices,
)
from hiding_adversarial_attacks.visualization.adversarial_obfuscation_rate import (
    plot_aors,
)
from hiding_adversarial_attacks.visualization.config import EXPLAINER_PLOT_NAMES
from hiding_adversarial_attacks.visualization.explanation_similarities import (
    data_set_mappings,
    visualize_explanation_similarities,
)

logger = logging.getLogger(__file__)


def save_test_results_as_csv(config, test_results, prefix=""):
    test_results_df = pd.DataFrame(test_results)
    test_results_csv = os.path.join(config.log_path, f"{prefix}test_results.csv")
    test_results_df.to_csv(test_results_csv)


def visualize_top_bottom_k(config, device, model):
    test_orig_images, test_orig_expl, test_orig_labels = torch.load(
        os.path.join(config.log_path, "test_orig.pt"),
        map_location=device,
    )
    test_adv_images, test_adv_expl, test_adv_labels = torch.load(
        os.path.join(config.log_path, "test_adv.pt"),
        map_location=device,
    )

    reverse, similarities = get_similarities(
        config.similarity_loss.name, test_orig_expl, test_adv_expl
    )
    top_indices, bottom_indices = get_top_and_bottom_k_indices(
        similarities, k=4, reverse=reverse
    )
    top_bottom_indices = torch.cat((top_indices, bottom_indices), dim=0)

    # Plot similarity loss distribution on all training samples
    df_similarities = pd.DataFrame(similarities.cpu().detach().numpy())
    df_similarities.hist(bins=20, log=True)
    plt.show()

    # todo: for some reason, the top and bottom k do not appear to have top
    #  and bottom PCC values
    model._visualize_batch_explanations(
        test_adv_expl[top_bottom_indices],
        test_adv_images[top_bottom_indices],
        test_adv_labels[top_bottom_indices],
        test_orig_expl[top_bottom_indices],
        test_orig_images[top_bottom_indices],
        test_orig_labels[top_bottom_indices],
        top_bottom_indices,
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
    if config.similarity_loss.name == SimilarityLossNames.MSE:
        loss_weight_similarity = 10 ** loss_weight_similarity
    # # note: both adv and original cross entropy loss weights should be the same
    # -> it makes no sense to prioritize the one over the other
    loss_weight_orig_ce = 1
    loss_weight_adv_ce = 1
    batch_size = config.batch_size
    if "batch_size" in config.optuna.search_space:
        batch_size = trial.suggest_categorical(
            "batch_size", config.optuna.search_space["batch_size"]
        )
    weight_decay = config.weight_decay
    if "weight_decay" in config.optuna.search_space:
        weight_decay = trial.suggest_categorical(
            "weight_decay", config.optuna.search_space["weight_decay"]
        )
    ce_class_weight = config.ce_class_weight
    if "ce_class_weight" in config.optuna.search_space:
        ce_class_weight_options = config.optuna.search_space["ce_class_weight"]
        ce_class_weight = trial.suggest_int(
            "ce_class_weight",
            ce_class_weight_options["low"],
            ce_class_weight_options["high"],
            step=ce_class_weight_options["step"],
        )
    steps_lr = config.steps_lr
    if "steps_lr" in config.optuna.search_space:
        steps_lr = trial.suggest_categorical(
            "steps_lr", config.optuna.search_space["steps_lr"]
        )
    gamma = config.gamma
    if "gamma" in config.optuna.search_space:
        gamma = trial.suggest_categorical("gamma", config.optuna.search_space["gamma"])

    return (
        loss_weight_orig_ce,
        loss_weight_adv_ce,
        loss_weight_similarity,
        lr,
        steps_lr,
        gamma,
        batch_size,
        weight_decay,
        ce_class_weight,
    )


def train(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
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
            config.steps_lr,
            config.gamma,
            config.batch_size,
            config.weight_decay,
            config.ce_class_weight,
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

    # Log important code files to Neptune
    log_code(neptune_logger)

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
        test_loader,
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
    test_loader: DataLoader,
    trial: optuna.trial.Trial = None,
):
    model = get_manipulatable_model(config)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)
    model.set_hydra_logger(logger)
    model.set_hydra_logger(logger)
    model.to(device)

    if config.freeze:
        for name, param in model.named_parameters():
            if config.explainer["layer_name"] in name:
                break
            param.requires_grad = False

    if config.convert_to_softplus:
        convert_relu_to_softplus(
            model,
            config,
            beta=config.soft_plus_beta,
            threshold=config.soft_plus_threshold,
        )

    # PyTorch Lightning Callbacks
    checkpoint_callback = hydra.utils.instantiate(config.checkpoint_config)
    neptune_callback = NeptuneLoggingCallback(
        log_path=config.log_path,
        image_log_path=model.image_log_path,
        trash_run=config.trash_run,
    )
    callbacks = [checkpoint_callback, neptune_callback]
    if config.early_stopping:
        early_stopping_callback = hydra.utils.instantiate(config.early_stopping_config)
        callbacks.append(early_stopping_callback)
    if trial is not None:
        callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor=VAL_NORM_TOTAL_LOSS),
        )
    trainer = Trainer(
        callbacks=callbacks,
        gpus=config.gpus,
        logger=neptune_logger,
        precision=config.precision,
        # accumulate_grad_batches=4,
        # amp_level='O0',
        # amp_backend='apex',
        max_epochs=config.max_epochs,
        gradient_clip_val=config.gradient_clip_val,
        auto_lr_find=config.auto_lr_find,
    )

    if config.auto_lr_find:
        trainer.tune(model, train_loader, validation_loader)

        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(
            model, train_loader, validation_loader, min_lr=1e-11, max_lr=1e-6
        )

        # Results can be found in
        print(f"LR Finder results: \n: {lr_finder.results}")

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

    trainer.fit(model, train_loader, validation_loader)

    latest_checkpoint = os.path.join(config.log_path, "checkpoints/final-model.ckpt")
    trainer.save_checkpoint(latest_checkpoint)

    # If ReLU was replaced by Softplus, revert this here
    if config.convert_to_softplus:
        convert_softplus_to_relu(model, config)

    # Test with best model checkpoint (Lightning does this automatically)
    test_results = trainer.test(model=model, test_dataloaders=test_loader)
    logger.info(f"Test results: \n {pformat(test_results)}")

    save_test_results_as_csv(config, test_results)

    # Save run metrics to .csv
    run = init_current_neptune_run(neptune_logger.version)
    save_run_data(run, config.log_path, stage="train")
    save_run_data(run, config.log_path, stage="val")

    model.to(device)

    visualize_explanation_similarities(
        model,
        train_loader,
        config.data_set.name,
        device,
        stage="train",
    )
    visualize_explanation_similarities(
        model,
        test_loader,
        config.data_set.name,
        device,
        stage="test",
    )

    copy_run_outputs(
        config.log_path,
        os.getcwd(),
        neptune_logger.name,
        neptune_logger.version,
    )
    del model
    del train_loader
    del validation_loader
    del test_loader

    loss = (
        trainer.callback_metrics["val_exp_sim"].item()
        + trainer.callback_metrics["val_ce_orig"].item()
        + trainer.callback_metrics["val_ce_adv"].item()
    )
    return loss


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
    # train_loader = data_module.train_dataloader()

    trainer = Trainer(gpus=config.gpus, logger=neptune_logger, deterministic=True)

    # Pre-manipulation: test model before adversarial manipulation
    model = get_testable_model(
        config,
        device,
        metricized_top_and_bottom_explanations,
    )

    pre_test_results = trainer.test(model, test_loader)
    logger.info("*******************")
    logger.info(f"Pre-manipulation test results: \n {pformat(pre_test_results)}")
    save_test_results_as_csv(config, pre_test_results, prefix="")

    # Rename files so that they are not overwritten by second model run
    types = ("image_log/*.png", "*.csv")  # the tuple of file types
    files_to_move = []
    for files in types:
        files_to_move.extend(glob.glob(os.path.join(config.log_path, files)))
    for file in files_to_move:
        new_path = f"{os.path.dirname(file)}/pre-{os.path.basename(file)}"
        shutil.move(file, new_path)

    # Post-manipulation: test model(s) after manipulation
    pattern = re.compile(r"(HAA-\d+)")
    accumulated_test_results = pd.DataFrame()
    accumulated_sim_df = pd.DataFrame()
    for checkpoint in config.checkpoint:
        run_id = pattern.findall(checkpoint)[0]
        model = get_testable_model(
            config,
            device,
            metricized_top_and_bottom_explanations,
            checkpoint=checkpoint,
        )

        test_results = trainer.test(model, test_loader)
        logger.info("*******************")
        logger.info(
            f"Post-manipulation test results for run ID '{run_id}': "
            f"\n {pformat(test_results)}"
        )
        test_results[0]["index"] = run_id
        accumulated_test_results = pd.concat(
            [accumulated_test_results, pd.DataFrame(test_results)]
        )

        sorted_df_sim = get_similarities_df(config, model)

        csv_path = os.path.join(config.log_path, f"{run_id}_test_similarities.csv")
        sorted_df_sim.to_csv(csv_path)
        accumulated_sim_df = pd.concat([accumulated_sim_df, sorted_df_sim])

    accumulated_test_results = accumulated_test_results.set_index("index")

    accumulated_test_results = accumulated_test_results.append(
        accumulated_test_results.agg(["mean", "std"], axis="index")
    )
    test_results_path = os.path.join(config.log_path, "concat_post_test_results.csv")
    accumulated_test_results.to_csv(test_results_path)
    accumulated_sim_df.to_csv(
        os.path.join(config.log_path, "concat_post_test_similarities.csv")
    )
    accumulated_sim_df.groupby("orig_label_name").agg(["mean", "std"]).to_csv(
        os.path.join(config.log_path, "aggregated_post_test_similarities_per_class.csv")
    )

    explainer_name = EXPLAINER_PLOT_NAMES[config.explainer.name]
    aor_df = pd.DataFrame(
        [
            pd.Series(pre_test_results).loc[0],
            accumulated_test_results.loc["mean"],
        ]
    )
    aor_df.index = ["pre", "post"]
    fig1, fig2 = plot_aors(
        aor_df,
        titles=[
            f"{explainer_name}: Pre- and post-manipulation"
            f" AOR curves for all classes",
            f"{explainer_name}: Pre- and post-manipulation "
            f"AOR curves for classes '{config.included_classes}'",
        ],
    )
    fig1.savefig(os.path.join(config.log_path, "mean_aor.png"), transparent=True)
    fig2.savefig(os.path.join(config.log_path, "mean_aor_class.png"), transparent=True)

    post_sim_df = accumulated_sim_df[
        accumulated_sim_df["orig_label"] == config.included_classes
    ]
    pre_sim_df = pd.read_csv(os.path.join(config.data_path, "test_similarities.csv"))
    pre_sim_df = pre_sim_df[pre_sim_df["orig_label"] == config.included_classes]
    merged_df = pd.merge(
        pre_sim_df,
        post_sim_df,
        on="orig_label",
        suffixes=["_pre", "_post"],
    )

    boxplot_fig = plot_pre_and_post_manipulation_boxplot_similarities(
        merged_df, explainer_name, config.included_classes
    )
    boxplot_fig.save(
        os.path.join(config.data_path, "pre_and_post_boxplots.png"),
        transparent=True,
    )

    # # Visualize top and bottom k explanations after manipulation
    # visualize_top_bottom_k(config, device, model)
    #
    # # Visualize and save Adversarial Obfuscation Rate (AOR) plot
    # plot_and_save_aors(config.log_path)
    #
    # visualize_explanation_similarities(
    #     model,
    #     train_loader,
    #     config.data_set.name,
    #     device,
    #     stage="train",
    # )
    # visualize_explanation_similarities(
    #     model,
    #     test_loader,
    #     config.data_set.name,
    #     device,
    #     stage="test",
    # )

    copy_run_outputs(
        config.log_path,
        os.getcwd(),
        neptune_logger.name,
        neptune_logger.version,
    )


def get_similarities_df(config, model):
    _, similarities_mse = get_similarities(
        "MSE", model.test_orig_explanations, model.test_adv_explanations
    )
    _, similarities_pcc = get_similarities(
        "PCC", model.test_orig_explanations, model.test_adv_explanations
    )
    _, similarities_ssim = get_similarities(
        "SSIM", model.test_orig_explanations, model.test_adv_explanations
    )
    df_sim = pd.DataFrame(
        [
            similarities_mse.cpu().detach().numpy(),
            similarities_pcc.cpu().detach().numpy(),
            similarities_ssim.cpu().detach().numpy(),
            model.test_orig_labels.cpu().detach().numpy(),
            model.test_orig_pred_labels.argmax(dim=1).cpu().detach().numpy(),
            model.test_adv_labels.cpu().detach().numpy(),
            model.test_adv_pred_labels.argmax(dim=1).cpu().detach().numpy(),
        ],
        index=[
            "mse_sim",
            "pcc_sim",
            "ssim_sim",
            "orig_label",
            "orig_pred_label",
            "adv_label",
            "adv_pred_label",
        ],
    ).T
    df_sim["orig_label"] = df_sim["orig_label"].astype(int)
    df_sim["orig_pred_label"] = df_sim["orig_pred_label"].astype(int)
    df_sim["adv_label"] = df_sim["adv_label"].astype(int)
    df_sim["adv_pred_label"] = df_sim["adv_pred_label"].astype(int)
    df_sim["orig_label_name"] = df_sim["orig_label"].map(
        data_set_mappings[config.data_set.name]
    )
    sorted_df_sim = df_sim.sort_values(by="orig_label")
    return sorted_df_sim


def get_testable_model(
    config,
    device,
    metricized_top_and_bottom_explanations,
    checkpoint=None,
):
    model = get_manipulatable_model(config)
    if checkpoint is not None:
        model = model.load_from_checkpoint(checkpoint)
    model.override_hparams(config)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)
    if config.convert_to_softplus:
        convert_relu_to_softplus(
            model,
            config,
            beta=config.soft_plus_beta,
            threshold=config.soft_plus_threshold,
        )
    model.to(device)
    model.eval()
    return model


def run_optuna_study(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
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
        test_loader,
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
        catch=(KeyboardInterrupt, AssertionError, RuntimeError),
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
    # torch.autograd.set_detect_anomaly(True)
    if config.seed_everything or config.test:
        seed_everything(config.random_seed)

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

    test_loader = data_module.test_dataloader()

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
                        test_loader,
                        device,
                        config,
                        metricized_top_and_bottom_explanations,
                        original_log_path,
                    )
                else:
                    train(
                        train_loader,
                        validation_loader,
                        test_loader,
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
                    test_loader,
                    device,
                    config,
                    metricized_top_and_bottom_explanations,
                    original_log_path,
                )
            else:
                train(
                    train_loader,
                    validation_loader,
                    test_loader,
                    device,
                    config,
                    metricized_top_and_bottom_explanations,
                    original_log_path,
                )


if __name__ == "__main__":
    run()
