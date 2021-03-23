from __future__ import print_function

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import StepLR

from hiding_adversarial_attacks.config import MNISTConfig


class MNISTNet(pl.LightningModule):
    """
    Simple Convolutional MNIST classifier.
    Adapted to PyTorch Lightning from
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self, hparams):
        super(MNISTNet, self).__init__()
        self.hparams = hparams
        self.lr = hparams.lr
        self.gamma = hparams.gamma
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MNISTNet")
        parser.add_argument(
            "--lr",
            type=float,
            default=1.0,
            metavar="LR",
            help="learning rate (default: 1.0)",
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=0.7,
            metavar="M",
            help="Learning rate step gamma (default: 0.7)",
        )
        return parent_parser

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
        self.log(MNISTConfig.TRAIN_LOSS, loss, on_step=True, logger=True)
        self.log(
            MNISTConfig.TRAIN_ACCURACY,
            self.train_accuracy(torch.exp(pred_label), gt_label),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log(MNISTConfig.VAL_LOSS, loss, logger=True)
        self.log(
            MNISTConfig.VAL_ACCURACY,
            self.validation_accuracy(torch.exp(pred_label), gt_label),
        )
        return loss

    def test_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log(MNISTConfig.TEST_LOSS, loss, logger=True)
        self.log(
            MNISTConfig.TEST_ACCURACY,
            self.test_accuracy(torch.exp(pred_label), gt_label),
        )
        return loss

    def training_epoch_end(self, outs):
        self.log(MNISTConfig.TRAIN_ACCURACY_EPOCH, self.train_accuracy.compute())

    def validation_epoch_end(self, outs):
        self.log(MNISTConfig.VAL_ACCURACY_EPOCH, self.validation_accuracy.compute())

    def test_epoch_end(self, outs):
        self.log(MNISTConfig.TEST_ACCURACY_EPOCH, self.test_accuracy.compute())
