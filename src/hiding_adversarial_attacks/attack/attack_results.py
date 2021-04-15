import os
from dataclasses import dataclass

import torch


@dataclass
class BatchAttackResults:
    images: torch.Tensor
    labels: torch.Tensor
    adv_images: torch.Tensor
    adv_labels: torch.Tensor
    misclassified_images: torch.Tensor
    misclassified_labels: torch.Tensor
    attacked_count: int
    adv_count: int
    misclassified_count: int
    epsilon: float


class AttackResults:
    def __init__(self, stage, epsilon, device):
        self.stage = stage
        self.epsilon = epsilon
        self.device = device
        self.images = torch.Tensor().to(device)
        self.labels = torch.Tensor().to(device)
        self.adv_images = torch.Tensor().to(device)
        self.adv_labels = torch.Tensor().to(device)
        self.misclassified_images = torch.Tensor().to(device)
        self.misclassified_labels = torch.Tensor().to(device)
        self.attacked_count = 0
        self.adv_count = 0
        self.misclassified_count = 0

    def add_batch(self, batch_attack_results: BatchAttackResults):
        self.images = torch.cat((self.images, batch_attack_results.images), 0)
        self.labels = torch.cat((self.labels, batch_attack_results.labels), 0)
        self.adv_labels = torch.cat(
            (self.adv_labels, batch_attack_results.adv_labels), 0
        )
        self.adv_images = torch.cat(
            (self.adv_images, batch_attack_results.adv_images), 0
        )
        self.adv_labels = torch.cat(
            (self.adv_labels, batch_attack_results.adv_labels), 0
        )
        self.misclassified_images = torch.cat(
            (self.misclassified_images, batch_attack_results.misclassified_images), 0
        )
        self.misclassified_labels = torch.cat(
            (self.misclassified_labels, batch_attack_results.misclassified_labels), 0
        )
        self.attacked_count += batch_attack_results.attacked_count
        self.adv_count += batch_attack_results.adv_count
        self.misclassified_count += batch_attack_results.misclassified_count

    def save_results(self, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        orig_path = os.path.join(target_dir, f"{self.stage}_orig.pt")
        adv_path = os.path.join(target_dir, f"{self.stage}_adv.pt")
        misclassified_path = os.path.join(target_dir, f"{self.stage}_misclassified.pt")
        torch.save((self.images.cpu(), self.labels.cpu()), orig_path)
        torch.save((self.adv_images.cpu(), self.adv_labels.cpu()), adv_path)
        torch.save(
            (self.misclassified_images.cpu(), self.misclassified_labels.cpu()),
            misclassified_path,
        )
