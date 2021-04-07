from __future__ import print_function

import os

import foolbox as fb
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import StepLR


class MNISTNet(pl.LightningModule):
    """
    Simple Convolutional MNIST classifier.
    Adapted to PyTorch Lightning from
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self, hparams):
        super(MNISTNet, self).__init__()
        self.hparams = hparams
        self.lr = hparams.classifier.lr
        self.gamma = hparams.classifier.gamma
        self.classifier_config = hparams.classifier
        self.data_set_config = hparams.data_set
        self.save_hyperparameters()

        # metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        # network structure
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    @classmethod
    def as_foolbox_wrap(cls, hparams, device):
        assert hparams.checkpoint is not None
        assert os.path.isfile(hparams.checkpoint)

        model = cls(hparams).load_from_checkpoint(hparams.checkpoint)
        model.eval()
        return fb.PyTorchModel(
            model,
            bounds=model.data_set_config.bounds,
            preprocessing=model.data_set_config.preprocessing,
            device=device,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]

    def _predict(self, batch):
        image, gt_label = batch
        pred_label = self(image)
        loss = F.nll_loss(pred_label, gt_label)
        return gt_label, loss, pred_label

    def training_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log(self.classifier_config.train_loss, loss, on_step=True, logger=True)
        self.log(
            self.classifier_config.train_accuracy,
            self.train_accuracy(torch.exp(pred_label), gt_label),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log(self.classifier_config.val_loss, loss, logger=True)
        self.log(
            self.classifier_config.val_accuracy,
            self.validation_accuracy(torch.exp(pred_label), gt_label),
        )
        return loss

    def test_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log(self.classifier_config.test_loss, loss, logger=True)
        self.log(
            self.classifier_config.test_accuracy,
            self.test_accuracy(torch.exp(pred_label), gt_label),
        )
        return loss

    def training_epoch_end(self, outs):
        self.log(
            self.classifier_config.train_accuracy_epoch, self.train_accuracy.compute()
        )

    def validation_epoch_end(self, outs):
        self.log(
            self.classifier_config.val_accuracy_epoch,
            self.validation_accuracy.compute(),
        )

    def test_epoch_end(self, outs):
        self.log(
            self.classifier_config.test_accuracy_epoch, self.test_accuracy.compute()
        )
