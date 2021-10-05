import os

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pylab
from matplotlib import pyplot as plt

from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames
from hiding_adversarial_attacks.visualization.config import (
    CONFUSION_MATRIX_COLOR_PALETTE,
    DATA_SET_MAPPING,
)


def save_confusion_matrix(matrix: np.array, data_set_name: str, log_path: str):
    _matrix = matrix.astype("int")
    data_set_classes = DATA_SET_MAPPING[data_set_name].values()
    df = pd.DataFrame(_matrix, index=data_set_classes, columns=data_set_classes)
    df.to_csv(os.path.join(log_path, "confusion_matrix.csv"))
    fig = plt.figure(figsize=(12, 10))
    ax = sn.heatmap(df, annot=True, fmt="d", cmap=CONFUSION_MATRIX_COLOR_PALETTE)
    ax.xaxis.label.set_size(14)
    ax.tick_params(axis="y", labelrotation=0, labelsize=14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis="x", labelrotation=-38, labelsize=14)
    fig.suptitle(f"{data_set_name} post-manipulation confusion matrix", fontsize=18)
    fig.tight_layout()
    fig.savefig(
        os.path.join(log_path, "image_log/confusion_matrix.png"),
        transparent=True,
    )
    fig.show()


def plot_ce_weight_ablations_confusion_matrix_plots():
    data_dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data"
        "/preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool"
        "--eps=0.105--cp-run=HAA-1728/exp=GuidedBackprop"
    )

    dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/logs/"
        "manipulate_model/AdversarialFashionMNISTWithExplanations"
    )

    params = {
        "legend.fontsize": "large",
        "figure.figsize": (17, 11),
        "figure.titlesize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    pylab.rcParams.update(params)

    data_set_classes = DATA_SET_MAPPING[DataSetNames.FASHION_MNIST].values()
    ce_weight_to_run_id_mapping = {
        "1": "HAA-5576",
        "10": "HAA-5577",
        "100": "HAA-5578",
        "200": "HAA-5579",
        "400": "HAA-5581",
    }

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex="col", sharey="row")
    cbar_ax = fig.add_axes([0.93, 0.108, 0.025, 0.772])

    confusion_matrix = pd.read_csv(
        os.path.join(dir, ce_weight_to_run_id_mapping["1"], "pre-confusion_matrix.csv"),
        index_col=0,
    )
    sn.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap=CONFUSION_MATRIX_COLOR_PALETTE,
        ax=axes[0][0],
        cbar=True,
        cbar_ax=cbar_ax,
        annot_kws={"size": 13},
    )
    axes[0][0].set_title("pre-manipulation", fontsize=16)

    for (ce_weight, run_id), ax in zip(
        ce_weight_to_run_id_mapping.items(), axes.flatten()[1:]
    ):
        cm = pd.read_csv(
            os.path.join(dir, run_id, "confusion_matrix.csv"),
            index_col=0,
        )
        cm.index = data_set_classes
        sn.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=CONFUSION_MATRIX_COLOR_PALETTE,
            ax=ax,
            cbar=False,
            cbar_ax=None,
            annot_kws={"size": 13},
        )
        ax.set_title(rf"$w_t={ce_weight}$", fontsize=16)

    fig.suptitle(
        (
            r"Comparison of confusion matrices from before the manipulation "
            "\n and for different values of Cross Entropy class weight $w_t$"
        ),
        fontsize=18,
    )
    fig.savefig(
        os.path.join(data_dir, "ce_weight_ablation_confusion_matrices.png"),
        transparent=True,
    )
    plt.show()


if __name__ == "__main__":
    plot_ce_weight_ablations_confusion_matrix_plots()
