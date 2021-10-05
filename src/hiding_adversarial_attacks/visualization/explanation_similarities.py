import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import lines
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames
from hiding_adversarial_attacks.manipulation.utils import get_similarities
from hiding_adversarial_attacks.visualization.config import (
    MSE_COLOR_PALETTE,
    PCC_COLOR_PALETTE,
    data_set_mappings,
)


def visualize_explanation_similarities(
    model: LightningModule,
    data_loader: DataLoader,
    data_set_name: str,
    device: torch.device,
    stage: str,
):
    data_set_map = data_set_mappings[data_set_name]
    use_original_explanations = (
        "Explanations" in model.hparams["hparams"]["data_set"]["name"]
    )

    orig_labels = np.array([])
    orig_pred_labels = np.array([])
    adv_labels = np.array([])
    adv_pred_labels = np.array([])
    sim_mse = np.array([])
    sim_pcc = np.array([])
    sim_ssim = np.array([])
    for batch in tqdm(data_loader):
        if use_original_explanations:
            (
                original_images,
                original_explanations_pre,
                adversarial_images,
                adversarial_explanations_pre,
                original_labels,
                adversarial_labels,
                batch_indices,
            ) = batch
        else:
            (
                original_images,
                adversarial_images,
                original_labels,
                adversarial_labels,
                batch_indices,
            ) = batch

        original_images = original_images.to(device)
        adversarial_images = adversarial_images.to(device)

        # get explanation maps
        original_explanation_maps = model.explainer.explain(
            original_images, original_labels.to(device)
        )
        adversarial_explanation_maps = model.explainer.explain(
            adversarial_images, adversarial_labels.to(device)
        )
        # Predict orig and adv label
        original_pred_labels = torch.argmax(model(original_images), dim=-1)
        adversarial_pred_labels = torch.argmax(model(adversarial_images), dim=-1)

        # calculate similarities
        _, similarities_mse = get_similarities(
            "MSE", original_explanation_maps, adversarial_explanation_maps
        )
        _, similarities_pcc = get_similarities(
            "PCC", original_explanation_maps, adversarial_explanation_maps
        )
        _, similarities_ssim = get_similarities(
            "SSIM", original_explanation_maps, adversarial_explanation_maps
        )

        # concat arrays
        orig_labels = np.append(orig_labels, original_labels.cpu().detach().numpy())
        orig_pred_labels = np.append(
            orig_pred_labels, original_pred_labels.cpu().detach().numpy()
        )
        adv_labels = np.append(adv_labels, adversarial_labels.cpu().detach().numpy())
        adv_pred_labels = np.append(
            adv_pred_labels, adversarial_pred_labels.cpu().detach().numpy()
        )
        sim_mse = np.append(sim_mse, similarities_mse.cpu().detach().numpy())
        sim_pcc = np.append(sim_pcc, similarities_pcc.cpu().detach().numpy())
        sim_ssim = np.append(sim_ssim, similarities_ssim.cpu().detach().numpy())

    df_sim = pd.DataFrame(
        [
            sim_mse,
            sim_pcc,
            sim_ssim,
            orig_labels,
            orig_pred_labels,
            adv_labels,
            adv_pred_labels,
        ],
        index=[
            "mse_sim",
            "pcc_sim",
            "ssim_sim",
            "orig_label",
            "orig_pred_label",
            "adv_label",
            "adv_pred_label",
        ],
    ).T
    df_sim["orig_label"] = df_sim["orig_label"].astype(int)
    df_sim["orig_pred_label"] = df_sim["orig_pred_label"].astype(int)
    df_sim["adv_label"] = df_sim["adv_label"].astype(int)
    df_sim["adv_pred_label"] = df_sim["adv_pred_label"].astype(int)
    df_sim["orig_label_name"] = df_sim["orig_label"].map(data_set_map)
    sorted_df_sim = df_sim.sort_values(by="orig_label")

    # Save similarities DataFrame as csv
    csv_path = os.path.join(
        model.hparams["hparams"]["log_path"], f"{stage}_similarities.csv"
    )
    sorted_df_sim.to_csv(csv_path)

    manipulated_classes = [
        data_set_map[c] for c in model.hparams["hparams"]["included_classes"]
    ]

    # Plot similarity histograms for both MSE and PCC
    hist_mse, hist_pcc, kde_mse, kde_pcc = plot_similarities(
        sorted_df_sim, data_set_name, data_set_map, manipulated_classes, stage
    )
    # Save plots
    hist_mse.savefig(
        os.path.join(
            model.image_log_path, f"{stage}_explanation_similarity_hist_mse.png"
        ),
        transparent=True,
    )
    hist_pcc.savefig(
        os.path.join(
            model.image_log_path, f"{stage}_explanation_similarity_hist_pcc.png"
        ),
        transparent=True,
    )
    kde_mse.savefig(
        os.path.join(
            model.image_log_path, f"{stage}_explanation_similarity_kde_mse.png"
        ),
        transparent=True,
    )
    kde_pcc.savefig(
        os.path.join(
            model.image_log_path, f"{stage}_explanation_similarity_kde_pcc.png"
        ),
        transparent=True,
    )


def plot_similarities(
    sorted_df_sim, data_set_name, data_set_map, manipulated_classes, stage
):
    if DataSetNames.FASHION_MNIST in data_set_name:
        data_set_name = DataSetNames.FASHION_MNIST
    else:
        data_set_name = DataSetNames.CIFAR10
    hist_mse = plot_similarities_histogram_with_boxplot(
        sorted_df_sim,
        "orig_label_name",
        "mse_sim",
        f"{data_set_name} {stage} — "
        f"explanation similarities (MSE) "
        f"after manipulation on classes '{manipulated_classes}'",
        log_x=True,
        palette=MSE_COLOR_PALETTE,
    )
    hist_pcc = plot_similarities_histogram_with_boxplot(
        sorted_df_sim,
        "orig_label_name",
        "pcc_sim",
        f"{data_set_name} {stage} — "
        f"explanation similarities (PCC) histogram "
        f"post-manipulation on classes '{manipulated_classes}'",
        log_x=False,
        palette=PCC_COLOR_PALETTE,
    )
    kde_mse = plot_similarities_kde(
        sorted_df_sim,
        "mse_sim",
        list(data_set_map.values()),
        f"{data_set_name} {stage} — "
        f"explanation similarities (MSE) histogram"
        f" KDE plots post-manipulation on classes '{manipulated_classes}'",
        log_x=True,
    )
    kde_pcc = plot_similarities_kde(
        sorted_df_sim,
        "pcc_sim",
        list(data_set_map.values()),
        f"{data_set_name} {stage} — "
        f" explanation similarities (PCC)"
        f" KDE plots post-manipulation on classes '{manipulated_classes}'",
        log_x=False,
    )
    return hist_mse, hist_pcc, kde_mse, kde_pcc


def plot_similarities_histogram_with_boxplot(
    df,
    group_by_col,
    similarity_col,
    title,
    log_x=False,
    log_y=False,
    ylim_factor=1.06,
    palette="PuRd",
):
    group = df[[similarity_col, group_by_col]].groupby(group_by_col, sort=False)
    means = group.mean().values.flatten()

    fig, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True, sharey=True)
    color_palette = sns.color_palette(palette, 5)
    p = sns.color_palette(palette, 1)

    for (label, g), ax, mean in zip(group, axes.flatten(), means):
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.set_title(label)
        ax2 = ax.twinx()

        # Filter values that are 0 for mse_sim due to log plotting
        g_nonzero = g
        if similarity_col == "mse_sim":
            g_nonzero = g[g[similarity_col] != 0]

        try:
            ax = sns.histplot(
                g_nonzero, ax=ax, palette=p, label=similarity_col, bins=30
            )
            if similarity_col == "pcc_sim":
                ax.set(xlim=(-1, 1))
            y_lim = int(ax.get_ylim()[1] * ylim_factor)
            ax.set(ylim=(-5, y_lim))
            sns.boxplot(
                data=g_nonzero,
                x=similarity_col,
                ax=ax2,
                color=color_palette[-2],
            )
            ax2.set(ylim=(-5, 1))
            ax2.axvline(
                mean,
                color=color_palette[-1],
                linestyle="dashed",
                linewidth=5,
                label="mean",
            )
            ax.get_legend().remove()
            ax.tick_params(axis="both", labelsize=14)
            ax.title.set_size(16)
        except Exception as e:
            print(f"EXCEPTION: {e}")

    x_label = "PCC"
    if similarity_col == "mse_sim":
        x_label = "MSE"
    axes[-1][0].set_xlabel(x_label, fontsize=14)
    axes[-1][1].set_xlabel(x_label, fontsize=14)

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()
    return fig


def plot_similarities_kde(
    df_sim,
    similarity_col: str,
    labels: List[str],
    title: str = None,
    log_x: bool = False,
    palette=sns.color_palette("tab20", 10),
):
    categories = list(df_sim["orig_label_name"].unique())
    fig, axes = plt.subplots(5, 2, figsize=(16, 12), sharex=True)

    for idx, (category, ax) in enumerate(zip(categories, axes.flatten())):
        class_sim = df_sim[df_sim["orig_label_name"] == category]

        adv_categories = sorted(list(class_sim["adv_label"].unique()))
        _palette = [palette[int(c)] for c in adv_categories]

        for adv_cat, color in zip(adv_categories, _palette):
            sim = class_sim[class_sim["adv_label"] == adv_cat]
            ax = sns.kdeplot(
                data=sim, x=similarity_col, ax=ax, color=color, label=adv_cat
            )

        if log_x:
            ax.set_xscale("log")
        ax.set_title(category)

    handles = [lines.Line2D([0], [0], ls="-", c=c) for c in palette]
    fig.legend(handles, labels, loc="center left")

    if title is not None:
        fig.suptitle(title)
    plt.show()
    return fig
