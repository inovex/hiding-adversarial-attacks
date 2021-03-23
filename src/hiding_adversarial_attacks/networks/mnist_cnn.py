from __future__ import print_function
import argparse
import os

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from pytorch_lightning import loggers as pl_loggers, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import StepLR

from hiding_adversarial_attacks.config import DataConfig, MNISTConfig
from hiding_adversarial_attacks.data.MNIST import MNISTDataModule


class MNISTNet(pl.LightningModule):
    """
    Simple Convolutional MNIST classifier.
    Adapted to PyTorch Lightning from https://github.com/pytorch/examples/blob/master/mnist/main.py
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
        self.log("train_loss", loss, on_step=True, logger=True)
        self.log("train_acc_step", self.train_accuracy(torch.exp(pred_label), gt_label))
        return loss

    def validation_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log("val_loss", loss, logger=True)
        self.log(
            "val_acc_step", self.validation_accuracy(torch.exp(pred_label), gt_label)
        )
        return loss

    def test_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log("test_loss", loss, logger=True)
        self.log("test_acc_step", self.test_accuracy(torch.exp(pred_label), gt_label))
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute())

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.validation_accuracy.compute())

    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute())


def parse_mnist_args():
    # add PROGRAM level args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--target-dir",
        default=os.path.join(DataConfig.PREPROCESSED_PATH, "adversarial_MNIST"),
        help="path to store attacked MNIST data to",
    )
    parser.add_argument(
        "--logs-dir",
        default=MNISTConfig.LOGS_PATH,
        help="path to store MNIST training logs and checkpoints to",
    )
    parser.add_argument(
        "--download-mnist",
        action="store_true",
        default=False,
        help="download & process MNIST data set",
    )
    parser = MNISTNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    return args


def init_mnist_data_module(batch_size, download_mnist, seed):
    # Download & prepare MNIST
    data_module = MNISTDataModule(
        DataConfig.EXTERNAL_PATH, batch_size=batch_size, random_seed=seed
    )
    if download_mnist:
        data_module.prepare_data()
    data_module.setup()
    return data_module


def train():
    args = parse_mnist_args()

    # MNIST data module and loaders
    data_module = init_mnist_data_module(
        args.batch_size, args.download_mnist, args.seed
    )
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    # model checkpoint settings
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="mnist-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # setup logging & trainer
    tb_logger = pl_loggers.TensorBoardLogger(args.logs_dir)
    trainer = Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback]
    )

    # Load model
    mnist_model = MNISTNet(args)

    # run training
    trainer.fit(mnist_model, train_loader, validation_loader)


if __name__ == "__main__":
    train()
