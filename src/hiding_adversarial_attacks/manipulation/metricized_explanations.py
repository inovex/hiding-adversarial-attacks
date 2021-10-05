from dataclasses import dataclass
from typing import List

import torch

from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
    SimilarityLossNames,
)


@dataclass
class MetricizedTopAndBottomExplanations:
    device: torch.device

    sorted_by: str

    top_and_bottom_indices: torch.Tensor
    top_and_bottom_original_images: torch.Tensor
    top_and_bottom_original_explanations: torch.Tensor
    top_and_bottom_original_labels: torch.Tensor
    top_and_bottom_original_label_names: List
    top_and_bottom_adversarial_images: torch.Tensor
    top_and_bottom_adversarial_explanations: torch.Tensor
    top_and_bottom_adversarial_labels: torch.Tensor
    top_and_bottom_adversarial_label_names: List

    def __post_init__(self):
        self.losses = self._calculate_losses()

    def _calculate_losses(self):
        losses = {}
        for loss_name, loss_func in SimilarityLossMapping.items():
            losses[loss_name] = []
            for (
                top_and_bottom_original_explanation,
                top_and_bottom_adversarial_explanation,
            ) in zip(
                self.top_and_bottom_original_explanations,
                self.top_and_bottom_adversarial_explanations,
            ):
                if loss_name is SimilarityLossNames.PCC:
                    orig = top_and_bottom_original_explanation.view(-1)
                    adv = top_and_bottom_adversarial_explanation.view(-1)
                else:
                    orig = top_and_bottom_original_explanation.unsqueeze(dim=0)
                    adv = top_and_bottom_adversarial_explanation.unsqueeze(dim=0)
                loss = loss_func(orig, adv).detach().cpu().item()
                losses[loss_name].append(loss)
        return losses
