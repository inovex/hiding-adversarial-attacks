import os

import torch


class BatchAttackResults:
    def __init__(
        self, images, labels, adv_images, adv_labels, orig_count, adv_count, epsilon
    ):
        self.images = images
        self.labels = labels
        self.adv_images = adv_images
        self.adv_labels = adv_labels
        self.orig_count = orig_count
        self.adv_count = adv_count
        self.epsilon = epsilon


class AttackResults:
    def __init__(self, stage, epsilon, device):
        self.stage = stage
        self.epsilon = epsilon
        self.device = device
        self.images = torch.Tensor().to(device)
        self.labels = torch.Tensor().to(device)
        self.adv_images = torch.Tensor().to(device)
        self.adv_labels = None
        self.orig_count = 0
        self.adv_count = 0

    def add_batch(self, batch_attack_results: BatchAttackResults):
        self.images = self._cat_tensor(self.images, batch_attack_results.images)
        self.labels = self._cat_tensor(self.labels, batch_attack_results.labels)
        self.adv_images = self._cat_tensor(
            self.adv_images, batch_attack_results.adv_images
        )
        self.orig_count += batch_attack_results.orig_count
        self.adv_count += batch_attack_results.adv_count

    @staticmethod
    def _cat_tensor(target_tensor, append_tensor):
        return torch.cat((target_tensor, append_tensor), 0)

    def save_results(self, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        orig_path = os.path.join(target_dir, f"{self.stage}_original.pt")
        adv_path = os.path.join(target_dir, f"{self.stage}_adv.pt")
        torch.save((self.images, self.labels), orig_path)
        torch.save((self.adv_images, self.adv_labels), adv_path)
