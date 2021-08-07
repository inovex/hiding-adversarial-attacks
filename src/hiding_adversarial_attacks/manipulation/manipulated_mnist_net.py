from __future__ import print_function

import os
from logging import Logger
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from optuna import TrialPruned
from torch import relu
from torch._vmap_internals import vmap
from torchmetrics import (
    F1,
    SSIM,
    Accuracy,
    ConfusionMatrix,
    MeanSquaredError,
    MetricCollection,
)

from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
    SimilarityLossNames,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import Stage
from hiding_adversarial_attacks.custom_metrics.adversarial_obfuscation_rate import (
    AdversarialObfuscationRate,
    ClassSpecificAdversarialObfuscationRate,
)
from hiding_adversarial_attacks.custom_metrics.pearson_corrcoef import (
    custom_pearson_corrcoef,
)
from hiding_adversarial_attacks.custom_metrics.relu_pearson_corrcoef import (
    ReluBatchedPearsonCorrCoef,
)
from hiding_adversarial_attacks.explainers.utils import get_explainer
from hiding_adversarial_attacks.manipulation.metricized_explanations import (
    MetricizedTopAndBottomExplanations,
)
from hiding_adversarial_attacks.utils import assert_not_none, get_included_class_indices
from hiding_adversarial_attacks.visualization.confusion_matrix import (
    save_confusion_matrix,
)
from hiding_adversarial_attacks.visualization.data_set_images import (
    visualize_difference_image_np,
)
from hiding_adversarial_attacks.visualization.explanations import (
    interpolate_explanations,
    visualize_single_explanation,
)
from hiding_adversarial_attacks.visualization.helpers import tensor_to_pil_numpy
from hiding_adversarial_attacks.visualization.normalization import (
    normalize_explanations,
)


class ManipulatedMNISTNet(pl.LightningModule):
    def __init__(self, model: MNISTNet, hparams):
        super().__init__()

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
        self.ce_class_weight = hparams.ce_class_weight

        # Logging
        self.image_log_intervals = hparams.image_log_intervals
        self.image_log_path = os.path.join(hparams.log_path, "image_log")
        os.makedirs(self.image_log_path, exist_ok=True)

        # Explainer
        self.explainer = get_explainer(self.model, hparams)
        self.test_orig_explanations = torch.tensor([]).to(self.device)
        self.test_orig_images = torch.tensor([]).to(self.device)
        self.test_orig_labels = torch.tensor([]).to(self.device)
        self.test_adv_explanations = torch.tensor([]).to(self.device)
        self.test_adv_images = torch.tensor([]).to(self.device)
        self.test_adv_labels = torch.tensor([]).to(self.device)

        # Explanation similarity loss
        self.similarity_loss = SimilarityLossMapping[hparams.similarity_loss.name]
        self.num_classes = 10

        self.hparams = OmegaConf.to_container(hparams)
        self.save_hyperparameters()

        self.zero_explanation_count = 0
        self.global_test_step = 0
        self.last_batch_explanation_sim = None

        self.included_classes = hparams.included_classes
        self.use_original_explanations = "Explanations" in self.hparams.data_set["name"]

        self.last_expl_sim = torch.tensor(1.0).to(self.device)
        self.last_cross_entropy_adv = torch.tensor(1.0).to(self.device)
        # Metrics tracking
        self._setup_metrics()

    def override_hparams(self, hparams):
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

        self.hparams = OmegaConf.to_container(hparams)

    def set_metricized_explanations(
        self,
        metricized_top_and_bottom_explanation: MetricizedTopAndBottomExplanations,
    ):
        self.metricized_explanations = metricized_top_and_bottom_explanation
        if self.hparams.similarity_loss["name"] == SimilarityLossNames.MSE:
            self.max_loss = torch.max(
                torch.FloatTensor(
                    self.metricized_explanations.losses[
                        self.hparams.similarity_loss["name"]
                    ]
                )
            )
        else:
            self.max_loss = 1.0

    def set_hydra_logger(self, logger: Logger):
        self.hydra_logger = logger

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
        return optimizer

    def forward(self, x, save_output=True):
        self.log_grad_norm()
        softmax_output = self.model(x)
        output = torch.exp(softmax_output)
        return output

    def log_grad_norm(self) -> None:
        norm_type = 2
        parameters = [
            p for p in self.model.parameters() if p.grad is not None and p.requires_grad
        ]
        if len(parameters) == 0:
            total_norm = 0.0
        else:
            device = parameters[0].grad.device
            total_norm = torch.norm(
                torch.stack(
                    [
                        torch.norm(p.grad.detach(), norm_type).to(device)
                        for p in parameters
                    ]
                ),
                2.0,
            ).item()
        self.log(
            "total_grad_norm",
            total_norm,
            prog_bar=False,
        )

    def composite_loss(
        self,
        original_images: torch.Tensor,
        original_labels: torch.Tensor,
        adversarial_images: torch.Tensor,
        adversarial_labels: torch.Tensor,
        included_mask: torch.Tensor,
        initial_original_explanations: torch.Tensor,
    ):
        assert original_images.shape == adversarial_images.shape
        assert original_labels.shape == adversarial_labels.shape

        images = torch.cat([original_images, adversarial_images], dim=0)
        labels = torch.cat([original_labels, adversarial_labels], dim=0)

        explanations = self.explainer.explain(images, labels)

        _explanations_orig, _explanations_adv = torch.split(
            explanations, int(len(explanations) / 2), dim=0
        )
        explanations_orig = torch.clone(_explanations_orig)
        explanations_adv = torch.clone(_explanations_adv)
        if self.hparams["normalize_explanations"]:
            _explanations_orig = normalize_explanations(
                _explanations_orig,
                self.hparams.explainer["name"],
                self.hparams["normalize_abs"],
            )
            _explanations_adv = normalize_explanations(
                _explanations_adv,
                self.hparams.explainer["name"],
                self.hparams["normalize_abs"],
            )

        # Forward pass images through network
        pred_labels = self(images)
        pred_labels_orig, pred_labels_adv = torch.split(
            pred_labels, int(len(pred_labels) / 2), dim=0
        )

        if len(torch.nonzero(included_mask)) == 0:
            # Case when there are no samples for included_classes
            # for explanation comparison in the batch
            # if explanation_similarity != explanation_similarity:
            similarity = torch.normal(
                self.last_expl_sim.item(),
                1e-5,
                (1,),
                requires_grad=True,
                device=self.device,
            )
            cross_entropy_adv = torch.normal(
                self.last_cross_entropy_adv.item(),
                1e-5,
                (1,),
                requires_grad=True,
                device=self.device,
            )
        else:
            # similarity
            orig_expl = torch.index_select(_explanations_orig, 0, included_mask)
            adv_expl = torch.index_select(_explanations_adv, 0, included_mask)
            initial_orig_expl = torch.index_select(
                initial_original_explanations, 0, included_mask
            )

            original_double = torch.cat(
                (
                    initial_orig_expl,
                    initial_orig_expl,
                ),
                dim=0,
            )
            predicted = torch.cat(
                (
                    orig_expl,
                    adv_expl,
                ),
                dim=0,
            )
            if original_double.shape != predicted.shape:
                predicted = interpolate_explanations(
                    predicted, original_double.shape[2:]
                )
            similarity = self.calculate_similarity_loss(original_double, predicted)
            # adversarial cross entropy
            _pred_labels_adv_incl = torch.index_select(
                pred_labels_adv, 0, included_mask
            )
            _adversarial_labels_incl = torch.index_select(
                adversarial_labels, 0, included_mask
            )
            cross_entropy_adv = F.cross_entropy(
                _pred_labels_adv_incl, _adversarial_labels_incl
            )
            assert_not_none(cross_entropy_adv, "cross_entropy_adv")

            self.last_expl_sim = similarity
            self.last_cross_entropy_adv = cross_entropy_adv

        # original cross entropy
        weights = torch.ones(self.num_classes, device=pred_labels_orig.device)
        weights[self.included_classes] = self.ce_class_weight
        cross_entropy_orig = F.cross_entropy(
            pred_labels_orig,
            original_labels,
            weight=weights,
        )
        assert_not_none(cross_entropy_orig, "cross_entropy_orig")

        combined_loss = (
            cross_entropy_orig + cross_entropy_adv + self.loss_weights[2] * similarity
        )
        # combined_loss = self.loss_weights[2] * similarity
        return (
            combined_loss,
            cross_entropy_orig,
            cross_entropy_adv,
            similarity,
            explanations_orig,
            explanations_adv,
            pred_labels_orig,
            pred_labels_adv,
        )

    def _predict(self, batch, stage: Stage):
        if self.use_original_explanations:
            (
                original_images,
                initial_original_explanations,
                adversarial_images,
                adversarial_explanations_pre,
                original_labels,
                adversarial_labels,
                batch_indeces,
            ) = batch
        else:
            (
                original_images,
                adversarial_images,
                original_labels,
                adversarial_labels,
                batch_indeces,
            ) = batch
            initial_original_explanations = None

        # Calculate loss
        included_mask = get_included_class_indices(
            original_labels, self.included_classes
        )
        (
            total_loss,
            cross_entropy_orig,
            cross_entropy_adv,
            similarity_loss,
            original_explanations,
            adversarial_explanations,
            pred_labels_orig,
            pred_labels_adv,
        ) = self.composite_loss(
            original_images,
            original_labels,
            adversarial_images,
            adversarial_labels,
            included_mask,
            initial_original_explanations,
        )

        # Logging
        self.log_losses(
            total_loss,
            cross_entropy_orig,
            cross_entropy_adv,
            similarity_loss,
            stage.value,
        )

        if stage == Stage.STAGE_TEST:
            self.global_test_step += 1
            self.test_confusion_matrix(pred_labels_orig.argmax(dim=-1), original_labels)
            self.test_f1_score(pred_labels_orig.argmax(dim=-1), original_labels)
            orig_expl_map = original_explanations
            self.test_aor(
                orig_expl_map,
                original_labels,
                adversarial_explanations,
                pred_labels_adv.argmax(dim=-1),
            )
            self.test_aor_class(
                orig_expl_map,
                original_labels,
                adversarial_explanations,
                pred_labels_adv.argmax(dim=-1),
            )
            if len(included_mask) > 0:
                self.test_adv_accuracy_class(
                    pred_labels_adv[included_mask],
                    adversarial_labels[included_mask],
                )

        self.log_classification_metrics(
            pred_labels_orig,
            original_labels,
            pred_labels_adv,
            adversarial_labels,
            stage,
        )
        self.log_similarity_metrics(
            original_explanations, adversarial_explanations, stage
        )

        if stage == Stage.STAGE_TEST:
            self.append_test_images_and_explanations(
                original_images,
                original_explanations,
                original_labels,
                adversarial_images,
                adversarial_explanations,
                adversarial_labels,
            )

        # Save original and adversarial explanations locally
        if (
            (stage == Stage.STAGE_TRAIN)
            and self.global_step % self.image_log_intervals[stage.value] == 0
        ) or (
            stage == Stage.STAGE_TEST
            and self.global_test_step % self.image_log_intervals[Stage.STAGE_TEST.value]
            == 0
        ):
            fig_name = f"test-step={self.global_test_step}_explanations.png"
            if stage != Stage.STAGE_TEST:
                self._visualize_top_and_bottom_k_explanations()
                fig_name = (
                    f"epoch={self.trainer.current_epoch}_"
                    f"step={self.global_step}_explanations.png"
                )

            self._visualize_batch_explanations(
                adversarial_explanations,
                adversarial_images,
                adversarial_labels,
                original_explanations,
                original_images,
                original_labels,
                batch_indeces,
                fig_name,
            )
        return total_loss

    def _index_select(self, tensors: Tuple[torch.Tensor], included_mask: torch.Tensor):
        return (torch.index_select(tensor, 0, included_mask) for tensor in tensors)

    def calculate_similarity_loss(self, source: torch.Tensor, target: torch.Tensor):
        if self.hparams.similarity_loss["name"] == SimilarityLossNames.MSE:
            return self.similarity_loss(source, target)
        elif self.hparams.similarity_loss["name"] == SimilarityLossNames.SSIM:
            return 1 - self.similarity_loss(source, target)
        elif self.hparams.similarity_loss["name"] == SimilarityLossNames.PCC:
            norm_sim = self.similarity_loss(source, target)
            sim = 1 - norm_sim
            return torch.mean(sim)

    def append_test_images_and_explanations(
        self,
        original_images,
        original_explanation_maps,
        original_labels,
        adversarial_images,
        adversarial_explanation_maps,
        adversarial_labels,
    ):
        self.test_orig_images = torch.cat(
            (
                self.test_orig_images.to(self.device),
                original_images.detach().to(self.device),
            ),
            dim=0,
        )
        self.test_orig_explanations = torch.cat(
            (
                self.test_orig_explanations.to(self.device),
                original_explanation_maps.detach().to(self.device),
            ),
            dim=0,
        )
        self.test_orig_labels = torch.cat(
            (
                self.test_orig_labels.to(self.device),
                original_labels.detach().to(self.device),
            ),
            dim=0,
        )
        self.test_adv_images = torch.cat(
            (
                self.test_adv_images.to(self.device),
                adversarial_images.detach().to(self.device),
            ),
            dim=0,
        )
        self.test_adv_explanations = torch.cat(
            (
                self.test_adv_explanations.to(self.device),
                adversarial_explanation_maps.detach().to(self.device),
            ),
            dim=0,
        )
        self.test_adv_labels = torch.cat(
            (
                self.test_adv_labels.to(self.device),
                adversarial_labels.detach().to(self.device),
            ),
            dim=0,
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
        test_confusion_matrix = self.test_confusion_matrix.compute()
        if DataSetNames.FASHION_MNIST in self.hparams.data_set["name"]:
            data_set_name = DataSetNames.FASHION_MNIST
        else:
            data_set_name = DataSetNames.CIFAR10
        save_confusion_matrix(
            test_confusion_matrix.cpu().detach().numpy(),
            data_set_name,
            self.hparams.log_path,
        )

        test_adv_accuracy_class = self.test_adv_accuracy_class.compute()
        self.log("test_adv_accuracy_class", test_adv_accuracy_class)

        test_f1_score = self.test_f1_score.compute()
        self.log("test_f1_score", test_f1_score)

        test_aor = self.test_aor.compute()
        test_aor_class = self.test_aor_class.compute()
        self.log_dict(
            {
                f"test_aor_tau={tau}": aor
                for tau, aor in zip(self.test_aor._taus, test_aor)
            }
        )
        self.log_dict(
            {
                f"test_aor_class_tau={tau}": aor
                for tau, aor in zip(self.test_aor_class._taus, test_aor_class)
            }
        )
        # Save explanations
        self.save_test_images_and_explanations()

    def save_test_images_and_explanations(self):
        orig_path = os.path.join(self.hparams.log_path, "test_orig.pt")
        adv_path = os.path.join(self.hparams.log_path, "test_adv.pt")
        torch.save(
            (
                self.test_orig_images.cpu(),
                self.test_orig_explanations.cpu(),
                self.test_orig_labels.cpu(),
            ),
            orig_path,
        )
        torch.save(
            (
                self.test_adv_images.cpu(),
                self.test_adv_explanations.cpu(),
                self.test_adv_labels.cpu(),
            ),
            adv_path,
        )

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
        self.log(
            f"{stage_name}_normalized_total_loss",
            total_loss,
            logger=True,
        )
        self.log(
            f"{stage_name}_ce_orig",
            cross_entropy_orig,
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{stage_name}_ce_adv",
            cross_entropy_adv,
            logger=True,
            prog_bar=True,
        )
        self.log(
            f"{stage_name}_exp_sim",
            explanation_similarity,
            logger=True,
            prog_bar=True,
        )

    def _setup_metrics(self):
        # Explanation similarity metrics
        similarity_metrics_dict = {
            "exp_ssim": SSIM(),
            "exp_pcc": ReluBatchedPearsonCorrCoef(device=self.device),
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

        # Test performance metrics
        self.test_adv_accuracy_class = Accuracy()
        self.test_f1_score = F1(num_classes=self.num_classes)
        self.test_confusion_matrix = ConfusionMatrix(num_classes=self.num_classes)
        self.test_aor = AdversarialObfuscationRate()
        self.test_aor_class = ClassSpecificAdversarialObfuscationRate(
            self.included_classes
        )

    def log_similarity_metrics(self, pred, target, stage: Stage):
        if stage == Stage.STAGE_TRAIN:
            self.train_similarity_metrics(pred.detach(), target.detach())
        elif stage == Stage.STAGE_VAL:
            self.validation_similarity_metrics(pred.detach(), target.detach())
        elif stage == Stage.STAGE_TEST:
            self.test_similarity_metrics(pred.detach(), target.detach())

    def log_classification_metrics(
        self, orig_pred, orig_target, adv_pred, adv_target, stage: Stage
    ):
        if stage == Stage.STAGE_TRAIN:
            self.train_classification_metrics["orig_acc"](
                orig_pred.detach(), orig_target.detach()
            )
            self.train_classification_metrics["adv_acc"](adv_pred, adv_target)
        elif stage == Stage.STAGE_VAL:
            self.validation_classification_metrics["orig_acc"](
                orig_pred.detach(), orig_target.detach()
            )
            self.validation_classification_metrics["adv_acc"](
                adv_pred.detach(), adv_target.detach()
            )
        elif stage == Stage.STAGE_TEST:
            self.test_classification_metrics["orig_acc"](
                orig_pred.detach(), orig_target.detach()
            )
            self.test_classification_metrics["adv_acc"](
                adv_pred.detach(), adv_target.detach()
            )

    def _visualize_batch_explanations(
        self,
        adversarial_explanation_maps,
        adversarial_images,
        adversarial_labels,
        original_explanation_maps,
        original_images,
        original_labels,
        batch_indeces,
        fig_name: str,
    ):
        figure, axes = self._visualize_explanations(
            original_images,
            adversarial_images,
            original_explanation_maps,
            adversarial_explanation_maps,
            original_labels,
            adversarial_labels,
            batch_indeces,
        )
        if figure is not None and axes is not None:
            fig_path = os.path.join(self.image_log_path, fig_name)
            figure.savefig(fig_path)
            plt.close("all")

    def _visualize_top_and_bottom_k_explanations(self):
        # visualize top and bottom k explanations from initial evaluation
        if self.global_step == 0:
            orig_explanation_maps = (
                self.metricized_explanations.top_and_bottom_original_explanations
            )
            adv_explanation_maps = (
                self.metricized_explanations.top_and_bottom_adversarial_explanations
            )
            if (
                "relu_attributions" in self.hparams.explainer
                and self.hparams.explainer["relu_attributions"]
            ):
                orig_explanation_maps = relu(orig_explanation_maps)
                adv_explanation_maps = relu(adv_explanation_maps)
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
            self.metricized_explanations.top_and_bottom_indices,
        )
        if top_bottom_k_figure is not None and top_bottom_k_axes is not None:
            top_bottom_k_fig_name = (
                f"epoch={self.trainer.current_epoch}_"
                f"step={self.global_step}_top_bottom_k_explanations.png"
            )
            top_bottom_k_fig_path = os.path.join(
                self.image_log_path, top_bottom_k_fig_name
            )
            top_bottom_k_figure.savefig(top_bottom_k_fig_path)

    def _visualize_explanations(
        self,
        original_images,
        adversarial_images,
        original_explanation_maps,
        adversarial_explanation_maps,
        original_labels,
        adversarial_labels,
        batch_indeces,
    ):
        n_rows = (
            8 if len(original_explanation_maps) > 8 else len(original_explanation_maps)
        )
        if n_rows <= 1:
            return None, None

        indeces = (
            torch.arange(0, n_rows)
            if len(original_images) > n_rows
            else torch.arange(0, len(original_images))
        )
        if len(indeces) == 0:
            print("WARNING: Batch to visualize was empty.")
            return None, None

        original_titles = [
            f"Original, label: {label}" for label in original_labels[indeces]
        ]
        adversarial_titles = [
            f"Adversarial, label: {label}" for label in adversarial_labels[indeces]
        ]

        orig_expl_maps = original_explanation_maps[indeces]
        adv_expl_maps = adversarial_explanation_maps[indeces]
        if self.hparams.similarity_loss["name"] == SimilarityLossNames.PCC:
            sim_type = "PCC"
            sim_loss = custom_pearson_corrcoef
            num_format = "{:.2f}"
        elif self.hparams.similarity_loss["name"] == SimilarityLossNames.SSIM:
            sim_type = "SSIM"
            sim_loss = self.similarity_loss
            num_format = "{:.2f}"
        elif self.hparams.similarity_loss["name"] == SimilarityLossNames.MSE:
            sim_type = "MSE"
            sim_loss = vmap(self.similarity_loss)
            num_format = "{:.2e}"
        else:
            raise RuntimeError(
                f"Unknown Similarity loss:" f" {self.hparams.similarity_loss['name']}"
            )

        norm_orig_expl = (
            normalize_explanations(
                orig_expl_maps,
                self.hparams.explainer["name"],
                self.hparams["normalize_abs"],
            )
            if self.hparams["normalize_explanations"]
            else orig_expl_maps
        )
        norm_adv_expl = (
            normalize_explanations(
                adv_expl_maps,
                self.hparams.explainer["name"],
                self.hparams["normalize_abs"],
            )
            if self.hparams["normalize_explanations"]
            else adv_expl_maps
        )
        similarities = (
            sim_loss(
                norm_orig_expl,
                norm_adv_expl,
            )
            .detach()
            .cpu()
        )

        image_shape = (original_images.shape[-2], original_images.shape[-1])
        orig_expl = tensor_to_pil_numpy(
            interpolate_explanations(orig_expl_maps, image_shape)
        )
        adv_expl = tensor_to_pil_numpy(
            interpolate_explanations(adv_expl_maps, image_shape)
        )
        orig_images = tensor_to_pil_numpy(original_images[indeces])
        adv_images = tensor_to_pil_numpy(adversarial_images[indeces])

        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(12, 12))
        for i, (row_axis, index, similarity) in enumerate(
            zip(axes, indeces, similarities)
        ):
            if self.zero_explanation_count >= 3:
                self.logger.experiment.append_tag("pruned")
                message = "Trial pruned due to too many explanation maps becoming zero."
                self.hydra_logger.warning(message)
                raise TrialPruned(message)

            if np.count_nonzero(orig_expl[index]) == 0:
                self.hydra_logger.warning(
                    "WARNING: original explanation contains all zeros!"
                )
                self.zero_explanation_count += 1
                continue
            if np.count_nonzero(adv_expl[index]) == 0:
                self.hydra_logger.warning(
                    "WARNING: adversarial explanation contains all zeros!"
                )
                self.zero_explanation_count += 1
                continue
            sim = num_format.format(similarity.item())
            visualize_single_explanation(
                orig_images[index],
                orig_expl[index],
                f"{original_titles[index]}, {sim_type}: {sim}, "
                f"id: {batch_indeces[index]}",
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
