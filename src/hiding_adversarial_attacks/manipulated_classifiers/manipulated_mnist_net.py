from __future__ import print_function

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from eagerpy import torch
from omegaconf import OmegaConf
from optuna import TrialPruned
from torchmetrics import SSIM, Accuracy, MeanSquaredError, MetricCollection

from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
    SimilarityLossNames,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import Stage
from hiding_adversarial_attacks.custom_metrics.batched_pearson_corrcoef import (
    BatchedPearsonCorrcoef,
)
from hiding_adversarial_attacks.explainers.utils import get_explainer
from hiding_adversarial_attacks.manipulated_classifiers.metricized_explanations import (
    MetricizedTopAndBottomExplanations,
)
from hiding_adversarial_attacks.utils import (
    tensor_to_pil_numpy,
    visualize_difference_image_np,
    visualize_single_explanation,
)


class ManipulatedMNISTNet(pl.LightningModule):
    def __init__(self, model: MNISTNet, hparams, logger=None):
        super(ManipulatedMNISTNet, self).__init__()

        # Classifier model to be manipulated
        self.model = model

        # Hyperparams
        self.lr = hparams.lr
        # self.gamma = hparams.gamma
        self.loss_weights = (
            hparams.loss_weight_orig_ce,
            hparams.loss_weight_adv_ce,
            hparams.loss_weight_similarity,
        )

        # Logging
        self.image_log_intervals = hparams.image_log_intervals
        self.image_log_path = os.path.join(hparams.log_path, "image_log")
        os.makedirs(self.image_log_path, exist_ok=True)

        # Explainer
        self.explainer = get_explainer(self.model, hparams)

        # Explanation similarity loss
        self.similarity_loss = SimilarityLossMapping[hparams.similarity_loss.name]

        # Metrics tracking
        self._setup_metrics()

        self.hparams = OmegaConf.to_container(hparams)
        self.save_hyperparameters()

        self.zero_explanation_count = 0

    def set_metricized_explanations(
        self,
        metricized_top_and_bottom_explanation: MetricizedTopAndBottomExplanations,
    ):
        self.metricized_explanations = metricized_top_and_bottom_explanation

    def on_train_start(self):
        assert self.metricized_explanations is not None
        top_and_bottom_indices = (
            self.metricized_explanations.top_and_bottom_indices.detach().cpu().tolist()
        )
        self.logger.experiment.log_text(
            "top_and_bottom_indices",
            f"{top_and_bottom_indices}",
        )
        self.logger.experiment.log_text(
            "top_and_bottom_mse",
            f"{self.metricized_explanations.losses[SimilarityLossNames.MSE]}",
        )
        self.logger.experiment.log_text(
            "top_and_bottom_ssim",
            f"{self.metricized_explanations.losses[SimilarityLossNames.SSIM]}",
        )
        self.logger.experiment.log_text(
            "top_and_bottom_pcc",
            f"{self.metricized_explanations.losses[SimilarityLossNames.PCC]}",
        )
        self.logger.experiment.log_text(
            "top_and_bottom_sorted_by",
            f"{self.metricized_explanations.sorted_by}",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        return optimizer

    def forward(self, x):
        softmax_output = self.model(x)
        output = torch.exp(softmax_output)
        return output

    def _predict(self, batch, stage: Stage):
        (
            original_images,
            adversarial_images,
            original_labels,
            adversarial_labels,
            batch_indeces,
        ) = batch

        # Create explanation maps
        original_explanation_maps = self.explainer.explain(
            original_images, original_labels
        )
        adversarial_explanation_maps = self.explainer.explain(
            adversarial_images, adversarial_labels
        )

        # Calculate combined loss
        (
            total_loss,
            cross_entropy_orig,
            cross_entropy_adv,
            explanation_similarity,
        ) = self.combined_loss(
            original_images,
            adversarial_images,
            original_explanation_maps,
            adversarial_explanation_maps,
            original_labels,
            adversarial_labels,
            stage,
        )
        self.log_losses(
            total_loss,
            cross_entropy_orig,
            cross_entropy_adv,
            explanation_similarity,
            stage.value,
        )

        # Safe original and adversarial explanations locally and to Neptune
        if (
            stage.value in self.image_log_intervals
            and self.global_step % self.image_log_intervals[stage.value] == 0
        ):
            self._visualize_batch_explanations(
                adversarial_explanation_maps,
                adversarial_images,
                adversarial_labels,
                original_explanation_maps,
                original_images,
                original_labels,
            )

            self._visualize_top_and_bottom_k_explanations()

        return total_loss

    def combined_loss(
        self,
        original_image: torch.Tensor,
        adversarial_image: torch.Tensor,
        original_explanation_map: torch.Tensor,
        adversarial_explanation_map: torch.Tensor,
        original_label: torch.Tensor,
        adversarial_label: torch.Tensor,
        stage: Stage,
    ):
        orig_pred_label = self(original_image).raw

        # Part 1: CrossEntropy for original image
        cross_entropy_orig = F.cross_entropy(orig_pred_label, original_label)
        assert_not_none(cross_entropy_orig, "cross_entropy_orig")

        # Part 2: CrossEntropy for adversarial image
        adv_pred_label = self(adversarial_image).raw
        cross_entropy_adv = F.cross_entropy(adv_pred_label, adversarial_label)
        assert_not_none(cross_entropy_adv, "cross_entropy_adv")

        # Part 3: Similarity between original and adversarial explanation maps
        explanation_similarity = self.similarity_loss(
            original_explanation_map, adversarial_explanation_map
        )
        total_loss = (
            (self.loss_weights[0] * cross_entropy_orig)
            + (self.loss_weights[1] * cross_entropy_adv)
            + (self.loss_weights[2] * explanation_similarity)
        )

        # Log metrics
        self.log_classification_metrics(
            orig_pred_label,
            original_label,
            adv_pred_label,
            adversarial_label,
            stage,
        )
        self.log_similarity_metrics(
            original_explanation_map, adversarial_explanation_map, stage
        )

        return (
            total_loss,
            cross_entropy_orig,
            cross_entropy_adv,
            explanation_similarity,
        )

    def training_step(self, batch, batch_idx):
        total_loss = self._predict(batch, Stage.STAGE_TRAIN)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = self._predict(batch, Stage.STAGE_VAL)
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss = self._predict(batch, Stage.STAGE_TEST)
        return total_loss

    def training_epoch_end(self, outs):
        self.log_epoch_metrics(
            self.train_similarity_metrics.compute(),
            self.train_classification_metrics.compute(),
            Stage.STAGE_TRAIN.value,
        )
        self.train_similarity_metrics.reset()

    def validation_epoch_end(self, outs):
        self.log_epoch_metrics(
            self.validation_similarity_metrics.compute(),
            self.validation_classification_metrics.compute(),
            Stage.STAGE_VAL.value,
        )
        self.validation_similarity_metrics.reset()

    def test_epoch_end(self, outs):
        self.log_epoch_metrics(
            self.test_similarity_metrics.compute(),
            self.test_classification_metrics.compute(),
            Stage.STAGE_TEST.value,
        )
        self.test_similarity_metrics.reset()

    def log_epoch_metrics(
        self, similarity_metrics: Dict, classification_metrics: Dict, stage: str
    ):
        self.log_epoch_similarities(similarity_metrics, stage)
        self.log_epoch_accuracies(classification_metrics, stage)

    def log_epoch_accuracies(self, similarity_metrics: Dict, stage_name: str):
        self.log(
            f"{stage_name}_orig_acc",
            similarity_metrics[f"{stage_name}_orig_acc"],
            prog_bar=False,
        )
        self.log(
            f"{stage_name}_adv_acc",
            similarity_metrics[f"{stage_name}_adv_acc"],
            prog_bar=False,
        )

    def log_epoch_similarities(self, similarity_metrics: Dict, stage_name: str):
        self.log(
            f"{stage_name}_exp_mse",
            similarity_metrics[f"{stage_name}_exp_mse"],
            prog_bar=False,
        )
        self.log(
            f"{stage_name}_exp_ssim",
            similarity_metrics[f"{stage_name}_exp_ssim"],
            prog_bar=False,
        )
        self.log(
            f"{stage_name}_exp_pcc",
            similarity_metrics[f"{stage_name}_exp_pcc"],
            prog_bar=False,
        )

    def log_losses(
        self,
        total_loss,
        cross_entropy_orig,
        cross_entropy_adv,
        explanation_similarity,
        stage_name: str,
    ):
        self.log(f"{stage_name}_total_loss", total_loss, on_step=True, logger=True)
        self.log(
            f"{stage_name}_ce_orig",
            cross_entropy_orig,
            on_step=True,
            logger=True,
        )
        self.log(
            f"{stage_name}_ce_adv",
            cross_entropy_adv,
            on_step=True,
            logger=True,
        )
        self.log(
            f"{stage_name}_exp_sim",
            explanation_similarity,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

    def _setup_metrics(self):
        # Explanation similarity metrics
        similarity_metrics_dict = {
            "exp_ssim": SSIM(),
            "exp_pcc": BatchedPearsonCorrcoef(),
            "exp_mse": MeanSquaredError(),
        }
        similarity_metrics = MetricCollection(similarity_metrics_dict)
        self.train_similarity_metrics = similarity_metrics.clone(prefix="train_")
        self.validation_similarity_metrics = similarity_metrics.clone(prefix="val_")
        self.test_similarity_metrics = similarity_metrics.clone(prefix="test_")

        # Classification performance metrics
        classification_metrics_dict = {
            "orig_acc": Accuracy(),
            "adv_acc": Accuracy(),
        }
        classification_metrics = MetricCollection(classification_metrics_dict)
        self.train_classification_metrics = classification_metrics.clone(
            prefix="train_"
        )
        self.validation_classification_metrics = classification_metrics.clone(
            prefix="val_"
        )
        self.test_classification_metrics = classification_metrics.clone(prefix="test_")

    def log_similarity_metrics(self, pred, target, stage: Stage):
        if stage == Stage.STAGE_TRAIN:
            self.train_similarity_metrics(pred, target)
        elif stage == Stage.STAGE_VAL:
            self.validation_similarity_metrics(pred, target)
        elif stage == Stage.STAGE_TEST:
            self.test_similarity_metrics(pred, target)

    def log_classification_metrics(
        self, orig_pred, orig_target, adv_pred, adv_target, stage: Stage
    ):
        if stage == Stage.STAGE_TRAIN:
            self.train_classification_metrics["orig_acc"](orig_pred, orig_target)
            self.train_classification_metrics["adv_acc"](adv_pred, adv_target)
        elif stage == Stage.STAGE_VAL:
            self.validation_classification_metrics["orig_acc"](orig_pred, orig_target)
            self.validation_classification_metrics["adv_acc"](adv_pred, adv_target)
        elif stage == Stage.STAGE_TEST:
            self.test_classification_metrics["orig_acc"](orig_pred, orig_target)
            self.test_classification_metrics["adv_acc"](adv_pred, adv_target)

    def _visualize_batch_explanations(
        self,
        adversarial_explanation_maps,
        adversarial_images,
        adversarial_labels,
        original_explanation_maps,
        original_images,
        original_labels,
    ):
        figure, axes = self._visualize_explanations(
            original_images,
            adversarial_images,
            original_explanation_maps,
            adversarial_explanation_maps,
            original_labels,
            adversarial_labels,
        )
        fig_name = (
            f"epoch={self.trainer.current_epoch}_"
            f"step={self.global_step}_explanations.png"
        )
        fig_path = os.path.join(self.image_log_path, fig_name)
        figure.savefig(fig_path)

    def _visualize_top_and_bottom_k_explanations(self):
        # visualize top and bottom k explanations from initial evaluation
        if self.global_step == 0:
            orig_explanation_maps = (
                self.metricized_explanations.top_and_bottom_original_explanations
            )
            adv_explanation_maps = (
                self.metricized_explanations.top_and_bottom_adversarial_explanations
            )
        else:
            orig_explanation_maps = self.explainer.explain(
                self.metricized_explanations.top_and_bottom_original_images,
                self.metricized_explanations.top_and_bottom_original_labels,
            )
            adv_explanation_maps = self.explainer.explain(
                self.metricized_explanations.top_and_bottom_adversarial_images,
                self.metricized_explanations.top_and_bottom_adversarial_labels,
            )
        (top_bottom_k_figure, top_bottom_k_axes,) = self._visualize_explanations(
            self.metricized_explanations.top_and_bottom_original_images,
            self.metricized_explanations.top_and_bottom_adversarial_images,
            orig_explanation_maps,
            adv_explanation_maps,
            self.metricized_explanations.top_and_bottom_original_labels,
            self.metricized_explanations.top_and_bottom_adversarial_labels,
        )
        top_bottom_k_fig_name = (
            f"epoch={self.trainer.current_epoch}_"
            f"step={self.global_step}_top_bottom_k_explanations.png"
        )
        top_bottom_k_fig_path = os.path.join(self.image_log_path, top_bottom_k_fig_name)
        top_bottom_k_figure.savefig(top_bottom_k_fig_path)

    def _visualize_explanations(
        self,
        original_images,
        adversarial_images,
        original_explanation_maps,
        adversarial_explanation_maps,
        original_labels,
        adversarial_labels,
    ):
        n_rows = 8 if self.hparams.batch_size > 8 else self.hparams.batch_size
        indeces = torch.arange(0, n_rows).raw

        original_titles = [
            f"Original, label: {label}" for label in original_labels[indeces]
        ]
        adversarial_titles = [
            f"Adversarial, label: {label}" for label in adversarial_labels[indeces]
        ]
        orig_images = tensor_to_pil_numpy(original_images[indeces])
        orig_expl = tensor_to_pil_numpy(original_explanation_maps[indeces])
        adv_images = tensor_to_pil_numpy(adversarial_images[indeces])
        adv_expl = tensor_to_pil_numpy(adversarial_explanation_maps[indeces])

        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(12, 12))
        for i, (row_axis, index) in enumerate(zip(axes, indeces)):
            if self.zero_explanation_count >= 3:
                raise TrialPruned(
                    "Trial pruned due to too many explanation maps becoming zero."
                )
            explanation_similarity = self.similarity_loss(
                original_explanation_maps[index],
                adversarial_explanation_maps[index],
            )
            if np.count_nonzero(orig_expl[index]) == 0:
                print("WARNING: original explanation contains all zeros!")
                self.zero_explanation_count += 1
                continue
            if np.count_nonzero(orig_expl[index]) == 0:
                print("WARNING: adversarial explanation contains all zeros!")
                self.zero_explanation_count += 1
                continue
            visualize_single_explanation(
                orig_images[index],
                orig_expl[index],
                f"{original_titles[index]}, sim: {explanation_similarity}",
                (fig, row_axis[0]),
                display_figure=False,
            )
            visualize_single_explanation(
                adv_images[index],
                adv_expl[index],
                adversarial_titles[index],
                (fig, row_axis[1]),
                display_figure=False,
            )
            visualize_difference_image_np(
                orig_expl[index],
                adv_expl[index],
                title="Explanation diff",
                plt_fig_axis=(fig, row_axis[2]),
                display_figure=False,
            )
        fig.tight_layout()
        return fig, axes


def assert_not_none(tensor, loss_name):
    assert not torch.isnan(tensor).any(), f"NaN in {loss_name}!"
