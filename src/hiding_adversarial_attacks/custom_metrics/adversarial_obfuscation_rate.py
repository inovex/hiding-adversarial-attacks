import os

import hydra
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.custom_metrics.spearman_corrcoef import (
    custom_spearman_corrcoef,
)
from hiding_adversarial_attacks.manipulation.utils import (
    get_manipulatable_model,
    load_attacked_data,
    load_explanations,
)


def load_pre_and_post_manipulation_models(config, device):
    pre_manipulation_model = get_manipulatable_model(config)
    pre_manipulation_model.to(device)
    pre_manipulation_model.eval()
    pre_manipulation_model.freeze()

    post_manipulation_model = get_manipulatable_model(config).load_from_checkpoint(
        config.checkpoint
    )
    post_manipulation_model.to(device)
    post_manipulation_model.eval()
    post_manipulation_model.freeze()

    return pre_manipulation_model, post_manipulation_model


def adversarial_obfuscation_rate(model, device, config, taus=None):
    if taus is None:
        taus = [0.3]

    (
        orig_explanations,
        orig_labels,
        orig_indices,
        _,
        _,
        _,
    ) = load_explanations(config, device, stage="test")
    (
        _,
        _,
        adv_images,
        adv_labels,
    ) = load_attacked_data(config.explanations_path, device, stage="test")

    aors = {tau: 0 for tau in taus}
    # Iterate over orig_explanations and adv_images
    for orig_expl, orig_lbl, orig_idx, adv_img, adv_lbl in tqdm(
        zip(
            orig_explanations,
            orig_labels,
            orig_indices,
            adv_images,
            adv_labels,
        )
    ):
        _adv_img = adv_img.unsqueeze(0)

        # Get explanation for adv_img
        adv_expl = model.explainer.explain(_adv_img, adv_lbl.long().unsqueeze(0))

        # Get predicted label for adv_img
        adv_pred_lbl = torch.argmax(model(_adv_img))

        spearman_rank = custom_spearman_corrcoef(orig_expl, adv_expl.squeeze(0))

        for tau in aors.keys():
            if spearman_rank >= tau and adv_pred_lbl != orig_lbl:
                aors[tau] += 1

    normalized_aors = {tau: aor / len(orig_explanations) for tau, aor in aors.items()}
    return normalized_aors


@hydra.main(config_name="manipulated_model_training_config")
def run(config: ManipulatedModelTrainingConfig):
    assert config.explainer.name in config.data_path, (
        f"Explainer name '{config.explainer.name}'"
        f" and data_path '{config.data_path}' don't match."
    )

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.gpus != 0) else "cpu"
    )

    (
        pre_manipulation_model,
        post_manipulation_model,
    ) = load_pre_and_post_manipulation_models(config, device)

    taus = np.around(np.linspace(0, 1, 10, endpoint=False), decimals=1)

    pre_aors = adversarial_obfuscation_rate(
        pre_manipulation_model,
        device,
        config,
        taus=taus,
    )
    post_aors = adversarial_obfuscation_rate(
        post_manipulation_model,
        device,
        config,
        taus=taus,
    )

    aor_df = pd.DataFrame([pre_aors, post_aors], index=["pre", "post"]).T
    ax = aor_df.plot.line(
        style=["o-", "^-"], color={"pre": "slateblue", "post": "purple"}
    )
    ax.legend(["pre-manipulation", "post-manipulation"])
    ax.set_xlabel("Tau")
    ax.set_ylabel("AOR")
    plt.show()


def save_adversarial_obfuscation_rate_to_csv(
    model, trainer, device, config: ManipulatedModelTrainingConfig, stage="pre"
):
    taus = np.around(np.linspace(0, 1, 10, endpoint=False), decimals=1)
    aors = adversarial_obfuscation_rate(
        model,
        device,
        config,
        taus=taus,
    )
    aors_df = pd.DataFrame(aors, index=[stage])
    aor_csv_file = os.path.join(config.log_path, f"{stage}_aors.csv")
    aors_df.to_csv(aor_csv_file)
    trainer.logger.experiment.log_artifact(aor_csv_file)


if __name__ == "__main__":
    run()
