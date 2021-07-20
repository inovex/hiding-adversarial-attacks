import torch

from hiding_adversarial_attacks.config.explainers.explainer_config import ExplainerNames


def normalize_to_range(x: torch.Tensor, min: int = 0, max: int = 1):
    ranged_x = min + ((x - torch.min(x)) * (max - min)) / (torch.max(x) - torch.min(x))
    return ranged_x


def normalize_to_sum_to_one(x: torch.Tensor):
    assert len(x.shape) == 4, (
        f"Expected 4 dimensional tensor." f" Received '{len(x.shape)}' dimensions."
    )
    softmax_x = torch.softmax(x.view(x.shape[0], -1), dim=1).view(x.shape)
    return softmax_x


def normalize_explanations(explanations: torch.Tensor, explainer_name: str):
    normalized_explanations = explanations
    # DeepLIFT
    if explainer_name == ExplainerNames.DEEP_LIFT:
        heatmap = torch.sum(torch.abs(explanations), dim=1)
        normalized_explanations = (heatmap / torch.sum(heatmap)).unsqueeze(1)

    # Grad-CAM & Input x Gradient
    elif explainer_name in [
        ExplainerNames.GRAD_CAM,
        ExplainerNames.INPUT_X_GRADIENT,
    ]:
        # --- 0
        _explanations = explanations / torch.abs(
            torch.sum(explanations, dim=(1, 2, 3))
        ).view(len(explanations), 1, 1, 1)
        normalized_explanations = (_explanations + 1) / 2

    return normalized_explanations
