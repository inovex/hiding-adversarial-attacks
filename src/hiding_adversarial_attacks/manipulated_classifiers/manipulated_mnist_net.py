from __future__ import print_function

from enum import Enum

import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from eagerpy import torch

from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
)
from hiding_adversarial_attacks.explainers.utils import get_explainer


class Stage(Enum):
    STAGE_TRAIN = "train"
    STAGE_VAL = "val"
    STAGE_TEST = "test"


class ManipulatedMNISTNet(pl.LightningModule):
    def __init__(self, model: MNISTNet, hparams):
        super(ManipulatedMNISTNet, self).__init__()

        # Classifier model to be manipulated
        self.model = model

        # Hyperparams
        self.lr = hparams.lr
        # self.gamma = hparams.gamma
        self.loss_weights = hparams.loss_weights

        # Explainer
        self.explainer = get_explainer(self.model, hparams)

        self.similarity_loss = SimilarityLossMapping[hparams.similarity_loss.name]
        self._setup_metrics()

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
            original_image,
            adversarial_image,
            original_label,
            adversarial_label,
        ) = batch

        # Create explanation maps
        original_explanation_map = self.explainer.explain(
            original_image, original_label
        )
        adversarial_explanation_map = self.explainer.explain(
            adversarial_image, original_label
        )

        # Calculate combined loss
        (
            total_loss,
            cross_entropy_orig,
            cross_entropy_adv,
            explanation_similarity,
        ) = self.combined_loss(
            original_image,
            adversarial_image,
            original_explanation_map,
            adversarial_explanation_map,
            original_label,
            adversarial_label,
            stage,
        )
        self.log_losses(
            total_loss,
            cross_entropy_orig,
            cross_entropy_adv,
            explanation_similarity,
            stage.value,
        )

        # Log explanation similarity metrics with logger
        self.log_similarity_metrics(
            original_explanation_map, adversarial_explanation_map, stage
        )

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
        self.log_orig_acc(orig_pred_label, original_label, stage)

        # Part 1: CrossEntropy for original image
        cross_entropy_orig = F.cross_entropy(orig_pred_label, original_label)
        assert_not_none(cross_entropy_orig, "cross_entropy_orig")

        # Part 2: CrossEntropy for adversarial image
        adv_pred_label = self(adversarial_image).raw
        cross_entropy_adv = F.cross_entropy(adv_pred_label, adversarial_label)
        assert_not_none(cross_entropy_adv, "cross_entropy_adv")
        self.log_adv_acc(adv_pred_label, adversarial_label, stage)

        # Part 3: Similarity between original and adversarial explanation maps
        explanation_similarity = self.similarity_loss(
            original_explanation_map, adversarial_explanation_map
        )
        total_loss = (
            (self.loss_weights[0] * cross_entropy_orig)
            + (self.loss_weights[1] * cross_entropy_adv)
            + (self.loss_weights[2] * explanation_similarity)
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
        self.train_accuracy_orig.compute()
        self.train_accuracy_adv.compute()
        # self.train_ssim.compute()
        self.train_mse.compute()

    def validation_epoch_end(self, outs):
        self.validation_accuracy_orig.compute()
        self.validation_accuracy_adv.compute()
        # self.validation_ssim.compute()
        self.validation_mse.compute()

    def test_epoch_end(self, outs):
        self.test_accuracy_orig.compute()
        self.test_accuracy_adv.compute()
        # self.test_ssim.compute()
        self.test_mse.compute()

    def log_losses(
        self,
        total_loss,
        cross_entropy_orig,
        cross_entropy_adv,
        explanation_similarity,
        stage_name: str,
    ):
        self.log(f"total_loss_{stage_name}", total_loss, on_step=True, logger=True)
        self.log(
            f"ce_orig_{stage_name}",
            cross_entropy_orig,
            on_step=True,
            logger=True,
        )
        self.log(
            f"ce_adv_{stage_name}",
            cross_entropy_adv,
            on_step=True,
            logger=True,
        )
        self.log(
            f"exp_sim_{stage_name}",
            explanation_similarity,
            on_step=True,
            logger=True,
            prog_bar=True,
        )

    def _setup_metrics(self):
        # Classification accuracy - original and adversarial
        self.train_accuracy_orig = torchmetrics.Accuracy()
        self.validation_accuracy_orig = torchmetrics.Accuracy()
        self.test_accuracy_orig = torchmetrics.Accuracy()
        self.train_accuracy_adv = torchmetrics.Accuracy()
        self.validation_accuracy_adv = torchmetrics.Accuracy()
        self.test_accuracy_adv = torchmetrics.Accuracy()

        # Similarity metrics for explanations
        # -- MSE
        self.train_mse = torchmetrics.MeanSquaredError()
        self.validation_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

        # -- Structural Similarity Index Measure (SSIM)
        self.train_ssim = torchmetrics.SSIM()
        self.validation_ssim = torchmetrics.SSIM()
        self.test_ssim = torchmetrics.SSIM()

        # # -- Pearson Cross Correlation
        # self.train_pcc = torchmetrics.PearsonCorrcoef()
        # self.validation_pcc = torchmetrics.PearsonCorrcoef()
        # self.test_pcc = torchmetrics.PearsonCorrcoef()

    def log_orig_acc(self, pred, target, stage: Stage):
        if stage == Stage.STAGE_TRAIN:
            self.train_accuracy_orig(pred, target)
        elif stage == Stage.STAGE_VAL:
            self.validation_accuracy_orig(pred, target)
        elif stage == Stage.STAGE_TEST:
            self.test_accuracy_orig(pred, target)

    def log_adv_acc(self, pred, target, stage: Stage):
        if stage == Stage.STAGE_TRAIN:
            self.train_accuracy_adv(pred, target)
        elif stage == Stage.STAGE_VAL:
            self.validation_accuracy_adv(pred, target)
        elif stage == Stage.STAGE_TEST:
            self.test_accuracy_adv(pred, target)

    def log_similarity_metrics(self, pred, target, stage: Stage):
        if stage == Stage.STAGE_TRAIN:
            self.train_mse(pred, target)
            # self.train_ssim(pred, target)
        elif stage == Stage.STAGE_VAL:
            self.validation_mse(pred, target)
            # self.validation_ssim(pred, target)
        elif stage == Stage.STAGE_TEST:
            self.test_mse(pred, target)
            # self.test_ssim(pred, target)


def assert_not_none(tensor, loss_name):
    assert not torch.isnan(tensor).any(), f"NaN in {loss_name}!"
