from dataclasses import dataclass

import torch

from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
)


@dataclass
class MetricizedTopAndBottomExplanations:
    device: torch.device

    sorted_by: str
    top_k_indices: torch.Tensor
    bottom_k_indices: torch.Tensor

    top_k_original_images: torch.Tensor
    top_k_original_explanations: torch.Tensor
    top_k_original_labels: torch.Tensor

    top_k_adversarial_images: torch.Tensor
    top_k_adversarial_explanations: torch.Tensor
    top_k_adversarial_labels: torch.Tensor

    bottom_k_original_images: torch.Tensor
    bottom_k_original_explanations: torch.Tensor
    bottom_k_original_labels: torch.Tensor

    bottom_k_adversarial_images: torch.Tensor
    bottom_k_adversarial_explanations: torch.Tensor
    bottom_k_adversarial_labels: torch.Tensor

    def __post_init__(self):
        self.losses = self._calculate_losses()

    @property
    def top_and_bottom_original_images(self):
        return torch.cat(
            (self.top_k_original_images, self.bottom_k_original_images), dim=0
        )

    @property
    def top_and_bottom_adversarial_images(self):
        return torch.cat(
            (self.top_k_adversarial_images, self.bottom_k_adversarial_images),
            dim=0,
        )

    @property
    def top_and_bottom_original_explanations(self):
        return torch.cat(
            (
                self.top_k_original_explanations,
                self.bottom_k_original_explanations,
            ),
            dim=0,
        )

    @property
    def top_and_bottom_adversarial_explanations(self):
        return torch.cat(
            (
                self.top_k_adversarial_explanations,
                self.bottom_k_adversarial_explanations,
            ),
            dim=0,
        )

    @property
    def top_and_bottom_original_labels(self):
        return torch.cat(
            (self.top_k_original_labels, self.bottom_k_original_labels), dim=0
        )

    @property
    def top_and_bottom_adversarial_labels(self):
        return torch.cat(
            (self.top_k_adversarial_labels, self.bottom_k_adversarial_labels),
            dim=0,
        )

    @property
    def top_and_bottom_indices(self):
        return torch.cat(
            (self.top_k_indices, self.bottom_k_indices),
            dim=0,
        )

    def _calculate_losses(self):
        losses = {}
        for loss_name, loss_func in SimilarityLossMapping.items():
            losses[loss_name] = []
            for (
                top_and_bottom_original_image,
                top_and_bottom_adversarial_image,
            ) in zip(
                self.top_and_bottom_original_images,
                self.top_and_bottom_adversarial_images,
            ):
                losses[loss_name].append(
                    loss_func(
                        top_and_bottom_original_image.unsqueeze(dim=0),
                        top_and_bottom_adversarial_image.unsqueeze(dim=0),
                    )
                    .detach()
                    .cpu()
                    .item()
                )
        return losses
