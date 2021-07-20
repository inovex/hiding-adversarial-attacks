from __future__ import print_function

import os

import foolbox as fb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from matplotlib import pyplot as plt

from hiding_adversarial_attacks.classifiers.pytorch_cifar10.cifar10_models.mobilenetv2 import (  # noqa: E501
    mobilenet_v2,
)
from hiding_adversarial_attacks.classifiers.pytorch_cifar10.scheduler import (
    WarmupCosineLR,
)
from hiding_adversarial_attacks.config.classifiers.classifier_config import (
    ClassifierConfig,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetConfig
from hiding_adversarial_attacks.visualization.helpers import tensor_to_pil_numpy


class CifarNet(pl.LightningModule):
    cifar_classes = (
        "airplane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    """
    DenseNet121-based Cifar-10 classifier.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr: float = hparams.lr
        self.gamma: float = hparams.gamma
        self.max_epochs = hparams.max_epochs
        self.classifier_config: ClassifierConfig = hparams.classifier
        self.data_set_config: DataSetConfig = hparams.data_set
        self.save_hyperparameters()

        # metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        # GPU or CPU
        device = torch.device(
            "cuda" if (torch.cuda.is_available() and hparams.gpus != 0) else "cpu"
        )
        # network
        self.model = mobilenet_v2(pretrained=True, device=device)

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
        x = self.model(x)
        output = F.log_softmax(x, dim=1)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-2,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer,
                warmup_epochs=total_steps * 0.3,
                max_epochs=total_steps,
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    def _predict(self, batch):
        image, gt_label = batch
        pred_label = self(image)
        loss = F.nll_loss(pred_label, gt_label)
        return gt_label, loss, pred_label

    def training_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log(
            self.classifier_config.train_loss,
            loss.detach(),
            on_step=True,
            logger=True,
        )
        self.log(
            self.classifier_config.train_accuracy,
            self.train_accuracy(torch.exp(pred_label).detach(), gt_label.detach()),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log(self.classifier_config.val_loss, loss.detach(), logger=True)
        self.log(
            self.classifier_config.val_accuracy,
            self.validation_accuracy(torch.exp(pred_label).detach(), gt_label.detach()),
        )
        return loss

    def test_step(self, batch, batch_idx):
        gt_label, loss, pred_label = self._predict(batch)
        self.log(self.classifier_config.test_loss, loss.detach(), logger=True)
        self.log(
            self.classifier_config.test_accuracy,
            self.test_accuracy(torch.exp(pred_label).detach(), gt_label.detach()),
        )
        return loss

    def training_epoch_end(self, outs):
        self.log(
            self.classifier_config.train_accuracy_epoch,
            self.train_accuracy.compute(),
        )

    def validation_epoch_end(self, outs):
        self.log(
            self.classifier_config.val_accuracy_epoch,
            self.validation_accuracy.compute(),
        )

    def test_epoch_end(self, outs):
        self.log(
            self.classifier_config.test_accuracy_epoch,
            self.test_accuracy.compute(),
        )

    def display_image(self, image_tensor, label_tensor):
        np_img = tensor_to_pil_numpy(image_tensor)[0]
        label = label_tensor[0].cpu().detach().item()
        title = f"Label: {self.cifar_classes[label]}"
        plt.imshow(np_img)
        if title is not None:
            plt.title(title)
        plt.show()
