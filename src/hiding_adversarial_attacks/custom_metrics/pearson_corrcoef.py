import torch
from torchmetrics.functional import pearson_corrcoef


def batched_pearson_corrcoef_compute(
    preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """ Custom implementation of the PCC for data with 4 dimensions: (B, C, M, N) """

    preds_diff = preds.view(preds.shape[0], -1) - preds.mean(dim=(1, 2, 3)).view(
        preds.shape[0], -1
    )
    t_mean = target.mean(dim=(1, 2, 3)).view(target.shape[0], -1)
    target_diff = target.view(target.shape[0], -1) - t_mean

    cov = (preds_diff * target_diff).mean(dim=1)
    preds_diff_square = preds_diff * preds_diff
    target_diff_square = target_diff * target_diff
    # prevent sqrt of zero
    zero_mask = preds_diff_square == 0
    if torch.nonzero(zero_mask).numel():
        preds_diff_square[zero_mask] = eps
    preds_std = torch.sqrt((preds_diff_square).mean(dim=1))
    # prevent sqrt of zero
    zero_mask = target_diff_square == 0
    if torch.nonzero(zero_mask).numel():
        target_diff_square[zero_mask] = eps
    target_std = torch.sqrt((target_diff_square).mean(dim=1))

    denom = preds_std * target_std
    # prevent division by zero
    zero_mask = denom == 0
    if torch.nonzero(zero_mask).numel():
        denom[zero_mask] = eps

    corrcoef = cov / denom
    return torch.clamp(corrcoef, -1.0, 1.0)


def windowed_batched_pearson_corrcoef_compute(
    preds: torch.Tensor,
    target: torch.Tensor,
    rows: int = 4,
    cols: int = 4,
) -> torch.Tensor:
    image_width = preds.shape[-2]
    image_height = preds.shape[-1]
    assert image_width % cols == 0 and image_height % rows == 0, (
        f"Window numbers ({cols}, {rows}) "
        f"does not match image dimension ({image_width}, {image_height})."
    )
    assert preds.shape == target.shape
    r = torch.stack(
        [
            batched_pearson_corrcoef_compute(
                _pred.unsqueeze(1), _target.unsqueeze(1)
            ).mean()
            for _pred, _target in zip(
                preds.view(preds.shape[0], -1, cols, rows).unbind(0),
                target.view(target.shape[0], -1, cols, rows).unbind(0),
            )
        ],
        dim=0,
    )
    return r


def custom_pearson_corrcoef(
    preds: torch.Tensor,
    target: torch.Tensor,
    windowed: bool = False,
):
    assert preds.shape == target.shape
    if preds.ndim > 1 or target.ndim > 1:
        if windowed:
            r = windowed_batched_pearson_corrcoef_compute(preds, target)
        else:
            r = batched_pearson_corrcoef_compute(preds, target)
    else:
        r = pearson_corrcoef(preds, target)
    return r
