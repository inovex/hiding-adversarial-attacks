from typing import List

import numpy as np
import torch
from torchmetrics import Metric

from hiding_adversarial_attacks.custom_metrics.pearson_corrcoef import (
    custom_pearson_corrcoef,
)
from hiding_adversarial_attacks.utils import get_included_class_indices


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


class ClassSpecificAdversarialObfuscationRate(AdversarialObfuscationRate):
    def __init__(
        self,
        class_ids: List[int],
        taus=None,
        compute_on_step=False,
        dist_sync_on_step=False,
    ):
        super().__init__(
            taus=taus,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
        )
        self._class_ids = class_ids

    def update(
        self,
        orig_expl: torch.Tensor,
        orig_label: torch.Tensor,
        adv_expl: torch.Tensor,
        adv_pred_lbl: torch.Tensor,
    ):
        assert orig_expl.shape == adv_expl.shape
        assert orig_label.shape == adv_pred_lbl.shape
        included_indices = get_included_class_indices(orig_label, self._class_ids)
        included_orig_expl = torch.index_select(orig_expl, 0, included_indices)
        included_orig_label = torch.index_select(orig_label, 0, included_indices)
        included_adv_expl = torch.index_select(adv_expl, 0, included_indices)
        included_adv_label = torch.index_select(adv_pred_lbl, 0, included_indices)
        if len(included_orig_expl) == 0:
            return
        super().update(
            included_orig_expl,
            included_orig_label,
            included_adv_expl,
            included_adv_label,
        )

    def compute(self):
        return self.aor / self.total
