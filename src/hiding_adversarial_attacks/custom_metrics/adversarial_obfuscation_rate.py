import numpy as np
import torch
from torchmetrics import Metric

from hiding_adversarial_attacks.custom_metrics.pearson_corrcoef import (
    custom_pearson_corrcoef,
)


class AdversarialObfuscationRate(Metric):
    def __init__(self, taus=None, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step
        )
        self._taus = taus
        if taus is None:
            self._taus = np.around(np.linspace(0, 1, 11, endpoint=True), decimals=1)
        self.add_state(
            "aor",
            default=torch.zeros((len(self._taus))),
        )
        self.add_state("total", default=torch.tensor(0))

    def update(
        self,
        orig_expl: torch.Tensor,
        orig_label: torch.Tensor,
        adv_expl: torch.Tensor,
        adv_pred_lbl: torch.Tensor,
    ):
        assert orig_expl.shape == adv_expl.shape
        assert orig_label.shape == adv_pred_lbl.shape
        for i, tau in enumerate(self._taus):
            self.aor[i] += compute_adversarial_obfuscation_rate(
                orig_expl, orig_label, adv_expl, adv_pred_lbl, tau
            )
        self.total += len(orig_expl)

    def compute(self):
        return self.aor / self.total


def compute_adversarial_obfuscation_rate(
    orig_expl: torch.Tensor,
    orig_label: torch.Tensor,
    adv_expl: torch.Tensor,
    adv_pred_lbl: torch.Tensor,
    tau,
):
    rank = custom_pearson_corrcoef(orig_expl, adv_expl)
    return torch.sum((rank >= tau) & (adv_pred_lbl != orig_label))
