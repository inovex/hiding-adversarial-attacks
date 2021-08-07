import os

import matplotlib.pylab as pylab
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames
from hiding_adversarial_attacks.config.explainers.explainer_config import ExplainerNames
from hiding_adversarial_attacks.evaluation.config import (
    COLOR_PALETTES,
    SIM_Y_LIMITS,
    SIMILARITIES_COLS,
    Y_LABELS,
)
from hiding_adversarial_attacks.evaluation.utils import (
    load_post_manipulation_test_similarities,
    load_pre_manipulation_test_similarities,
)
from hiding_adversarial_attacks.visualization.config import (
    DATA_SET_MAPPING,
    DATA_SET_PLOT_NAMES,
    EXPLAINER_PLOT_NAMES,
)


def get_merged_sim_df(data_set_path, runs_path, run_ids, class_id):
    pre_sim = load_pre_manipulation_test_similarities(data_set_path)
    pre_sim_class = pre_sim[pre_sim["orig_label"] == class_id]
    post_sim = load_post_manipulation_test_similarities(runs_path, run_ids)
    post_sim_class = post_sim[post_sim["orig_label"] == class_id]
    merged_df = pd.merge(
        pre_sim_class,
        post_sim_class,
        on="orig_label",
        suffixes=["_pre", "_post"],
    )
    return merged_df


def plot_pre_manipulation_similarities(data_set_path, data_set_name, explainer_name):
    params = {
        "legend.fontsize": "large",
        "figure.figsize": (16, 8),
        "figure.titlesize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
    }
    pylab.rcParams.update(params)

    class_names = DATA_SET_MAPPING[data_set_name].values()
    explainer_plot_name = EXPLAINER_PLOT_NAMES[explainer_name]
    data_set_plot_name = DATA_SET_PLOT_NAMES[data_set_name]
    test_similarities_df = load_pre_manipulation_test_similarities(data_set_path)
    # palette = sns.color_palette("Spectral", 10)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, (col, ax, y_label, p, y_limits) in enumerate(
        zip(SIMILARITIES_COLS, axes, Y_LABELS, COLOR_PALETTES, SIM_Y_LIMITS)
    ):
        if col == "mse_sim":
            ax.set_yscale("log")
        sns.boxplot(
            data=test_similarities_df,
            y=col,
            x="orig_label",
            ax=ax,
            palette=p,
            showfliers=False,
            # showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
        )
        ax.set_ylim(y_limits)
        ax.set_xticklabels(class_names)
        ax.set_ylabel(y_label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=-30)
    fig.suptitle(
        f"{data_set_plot_name} test - pre-manipulation "
        f"{explainer_plot_name} explanation similarities"
    )
    fig.tight_layout()
    plt.show()
    fig.savefig(
        os.path.join(
            data_set_path,
            f"{data_set_name}_{explainer_name}_pre_manipulation_sim_boxplots.png",
        ),
        transparent=True,
    )


def plot_pre_and_post_manipulation_boxplot_similarities(
    data_set_path,
    runs_path,
    data_set_name,
    explainer_name,
    top_run_ids,
    bottom_run_ids,
    top_class_id,
    bottom_class_id,
):
    params = {
        "legend.fontsize": "large",
        "figure.figsize": (16, 8),
        "figure.titlesize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
    }
    pylab.rcParams.update(params)

    top_merged = get_merged_sim_df(data_set_path, runs_path, top_run_ids, top_class_id)
    bottom_merged = get_merged_sim_df(
        data_set_path, runs_path, bottom_run_ids, bottom_class_id
    )

    top_class_name = DATA_SET_MAPPING[data_set_name][top_class_id]
    bottom_class_name = DATA_SET_MAPPING[data_set_name][bottom_class_id]
    data_set_plot_name = DATA_SET_PLOT_NAMES[data_set_name]
    explainer_plot_name = EXPLAINER_PLOT_NAMES[explainer_name]

    print("")
    x_ticklabels = ["pre-manipulation", "post-manipulation"]

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True)
    for i, (col, ax, y_label, palette) in enumerate(
        zip(SIMILARITIES_COLS, axes, Y_LABELS, COLOR_PALETTES)
    ):
        if i == 0:
            ax[0].set_title(f"Top class: {top_class_name}")
            ax[1].set_title(f"Bottom class: {bottom_class_name}")

        if col == "mse_sim":
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
        sns.boxplot(
            data=top_merged[[f"{col}_pre", f"{col}_post"]],
            ax=ax[0],
            palette=palette,
            showfliers=False,
            # showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
        )
        sns.boxplot(
            data=bottom_merged[[f"{col}_pre", f"{col}_post"]],
            ax=ax[1],
            palette=palette,
            showfliers=False,
            # showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
        )
        ax[0].set_xticklabels(x_ticklabels)
        ax[0].set_ylabel(y_label)
    fig.suptitle(
        f"{data_set_plot_name}: Comparison of pre- and post-manipulation"
        f" \n {explainer_plot_name} explanation similarities"
    )
    fig.tight_layout()
    plt.show()
    fig.savefig(
        os.path.join(
            data_set_path,
            f"{data_set_name}_{explainer_name}_top_and_bottom_similarities.png",
        ),
        transparent=True,
    )


def plot_fashion_mnist_pre_manipulation_similarity_boxplots():
    # Grad-CAM
    grad_cam_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False"
    )
    plot_pre_manipulation_similarities(
        grad_cam_path, DataSetNames.FASHION_MNIST, ExplainerNames.GRAD_CAM
    )

    # Guided Backprop
    guided_backprop_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=FashionMNIST--attack=DeepFool--"
        "eps=0.105--cp-run=HAA-1728/exp=GuidedBackprop"
    )
    plot_pre_manipulation_similarities(
        guided_backprop_path,
        DataSetNames.FASHION_MNIST,
        ExplainerNames.GUIDED_BACKPROP,
    )


def plot_cifar10_pre_manipulation_similarity_boxplots():
    # Grad-CAM
    grad_cam_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=CIFAR10--attack=DeepFool--"
        "eps=0.1--cp-run=resnet18/exp=GradCAM--l=model.layer2.1.conv2--ra=False"
    )
    plot_pre_manipulation_similarities(
        grad_cam_path, DataSetNames.CIFAR10, ExplainerNames.GRAD_CAM
    )

    # Guided Backprop
    guided_backprop_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=CIFAR10--attack=DeepFool--"
        "eps=0.1--cp-run=resnet18/exp=GuidedBackprop"
    )
    plot_pre_manipulation_similarities(
        guided_backprop_path,
        DataSetNames.CIFAR10,
        ExplainerNames.GUIDED_BACKPROP,
    )


def plot_fashion_mnist_grad_cam_pre_and_post_manipulation_boxplots():
    runs_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/logs/"
        "manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    # Class: Coat
    bottom_class_id = 4
    bottom_run_ids = range(4503, 4508)
    # Class: Sandal
    top_class_id = 5
    top_run_ids = range(4493, 4497)
    path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False"
    )
    plot_pre_and_post_manipulation_boxplot_similarities(
        path,
        runs_path,
        DataSetNames.FASHION_MNIST,
        ExplainerNames.GRAD_CAM,
        top_run_ids,
        bottom_run_ids,
        top_class_id,
        bottom_class_id,
    )


if __name__ == "__main__":
    plot_fashion_mnist_pre_manipulation_similarity_boxplots()
    # plot_cifar10_pre_manipulation_similarity_boxplots()
    # load_pre_manipulation_test_similarities(path)
