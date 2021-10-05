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
    SIMILARITIES_FILE,
    Y_LABELS,
)
from hiding_adversarial_attacks.visualization.config import (
    DATA_SET_MAPPING,
    EXPLAINER_PLOT_NAMES,
)


def load_pre_manipulation_test_similarities(data_set_path):
    sim_df = pd.read_csv(os.path.join(data_set_path, SIMILARITIES_FILE), index_col=0)
    return sim_df


def get_merged_sim_df(data_path, run_path, class_id):
    pre_sim = pd.read_csv(
        os.path.join(data_path, "test_similarities.csv"),
        index_col=0,
    )

    post_sim = pd.read_csv(
        os.path.join(run_path, "concat_post_test_similarities.csv"), index_col=0
    )
    if type(class_id) is list:
        pre_sim_class = pre_sim[pre_sim["orig_label"].isin(class_id)]
        post_sim_class = post_sim[post_sim["orig_label"].isin(class_id)]
    else:
        pre_sim_class = pre_sim[pre_sim["orig_label"] == class_id]
        post_sim_class = post_sim[post_sim["orig_label"] == class_id]
    merged_df = pd.merge(
        pre_sim_class,
        post_sim_class,
        on="orig_label",
        suffixes=["_pre", "_post"],
    )
    return merged_df


def plot_pre_manipulation_similarities_PCC(
    data_set_path, data_set_name, explainer_name
):
    params = {
        "legend.fontsize": "large",
        "figure.figsize": (16, 8),
        "figure.titlesize": 18,
        "axes.labelsize": 16,
        "axes.titlesize": "x-large",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    pylab.rcParams.update(params)

    class_names = DATA_SET_MAPPING[data_set_name].values()
    explainer_plot_name = EXPLAINER_PLOT_NAMES[explainer_name]
    test_similarities_df = load_pre_manipulation_test_similarities(data_set_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
    sns.boxplot(
        data=test_similarities_df,
        y=SIMILARITIES_COLS[0],
        x="orig_label",
        ax=ax,
        palette=COLOR_PALETTES[0],
        showfliers=False,
        showmeans=True,
        meanprops={
            "markerfacecolor": "#fea040",
            "markeredgecolor": "black",
            "markersize": "10",
        },
        whis=[5, 95],
    )
    ax.set_ylim([-1.05, 1.05])
    ax.set_xticklabels(class_names)
    ax.set_ylabel(Y_LABELS[0])
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=-30)
    fig.suptitle(f"Pre-manipulation {explainer_plot_name} PCC explanation similarities")
    fig.tight_layout()
    plt.show()
    fig.savefig(
        os.path.join(
            data_set_path,
            f"{data_set_name}_{explainer_name}_pre_manipulation_sim_boxplots_PCC.png",
        ),
        transparent=True,
    )


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
    test_similarities_df = load_pre_manipulation_test_similarities(data_set_path)

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
            showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
            whis=[5, 95],
        )
        ax.set_ylim(y_limits)
        ax.set_xticklabels(class_names)
        ax.set_ylabel(y_label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=-30)
    fig.suptitle(f"Pre-manipulation " f"{explainer_plot_name} explanation similarities")
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
    pre_and_post_dim_df,
    explainer_name,
    class_names,
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

    x_ticklabels = ["pre-manipulation", "post-manipulation"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, (col, ax, y_label, palette) in enumerate(
        zip(SIMILARITIES_COLS, axes, Y_LABELS, COLOR_PALETTES)
    ):
        if i == 0:
            ax.set_title(f"Classes: {', '.join(class_names)}")
        if col == "pcc_sim":
            ax.set_ylim([-1, 1])
        if col == "mse_sim":
            ax.set_yscale("log")
        sns.boxplot(
            data=pre_and_post_dim_df[[f"{col}_pre", f"{col}_post"]],
            ax=ax,
            palette=palette,
            showfliers=False,
            showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
        )
        ax.set_xticklabels(x_ticklabels)
        ax.set_ylabel(y_label)
    fig.suptitle(
        f"Comparison of pre- and post-manipulation"
        f" \n {explainer_name} explanation similarities"
    )
    fig.tight_layout()
    plt.show()
    return fig


def plot_pre_and_post_manipulation_boxplot_similarities_multiclass(
    data_path,
    top_run_path,
    bottom_run_path,
    data_set_name,
    explainer_name,
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

    top_merged = get_merged_sim_df(data_path, top_run_path, top_class_id)
    bottom_merged = get_merged_sim_df(data_path, bottom_run_path, bottom_class_id)

    top_class_name = DATA_SET_MAPPING[data_set_name][top_class_id]
    bottom_class_name = DATA_SET_MAPPING[data_set_name][bottom_class_id]
    explainer_plot_name = EXPLAINER_PLOT_NAMES[explainer_name]

    print("")
    x_ticklabels = ["pre-manipulation", "post-manipulation"]

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey="row")
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
            showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
            whis=[5, 95],
        )
        sns.boxplot(
            data=bottom_merged[[f"{col}_pre", f"{col}_post"]],
            ax=ax[1],
            palette=palette,
            showfliers=False,
            showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
            whis=[5, 95],
        )
        ax[0].set_xticklabels(x_ticklabels)
        ax[0].set_ylabel(y_label)
    fig.suptitle(
        f"Pre- and post-manipulation"
        f" \n {explainer_plot_name} explanation similarities"
    )
    fig.tight_layout()
    plt.show()
    top_save_path = os.path.join(
        top_run_path,
        f"{data_set_name}_{explainer_name}_top_and_bottom_similarities.png",
    )
    bottom_save_path = os.path.join(
        bottom_run_path,
        f"{data_set_name}_{explainer_name}_top_and_bottom_similarities.png",
    )
    fig.savefig(
        top_save_path,
        transparent=True,
    )
    fig.savefig(
        bottom_save_path,
        transparent=True,
    )


def plot_pre_and_post_manipulation_boxplot_similarities_merged(
    data_path,
    first_run_path,
    second_run_path,
    merged_run_path,
    data_set_name,
    explainer_name,
    first_class_id,
    second_class_id,
    save_path,
):
    params = {
        "legend.fontsize": "large",
        "figure.figsize": (16, 8),
        "figure.titlesize": 18,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 13,
    }
    pylab.rcParams.update(params)

    first_merged = get_merged_sim_df(data_path, first_run_path, first_class_id)
    second_merged = get_merged_sim_df(data_path, second_run_path, second_class_id)
    merged_df = get_merged_sim_df(
        data_path, merged_run_path, [first_class_id, second_class_id]
    )

    first_class_name = DATA_SET_MAPPING[data_set_name][first_class_id]
    second_class_name = DATA_SET_MAPPING[data_set_name][second_class_id]
    merged_class_name = f"{first_class_name} & {second_class_name}"
    explainer_plot_name = EXPLAINER_PLOT_NAMES[explainer_name]

    print("")
    x_ticklabels = ["pre-manipulation", "post-manipulation"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True, sharey="row")
    for i, (col, ax, y_label, palette) in enumerate(
        zip(SIMILARITIES_COLS, axes, Y_LABELS, COLOR_PALETTES)
    ):
        if i == 0:
            ax[0].set_title(f"{first_class_name}")
            ax[1].set_title(f"{second_class_name}")
            ax[2].set_title(f"{merged_class_name}")

        if col == "mse_sim":
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
            ax[2].set_yscale("log")
        sns.boxplot(
            data=first_merged[[f"{col}_pre", f"{col}_post"]],
            ax=ax[0],
            palette=palette,
            showfliers=False,
            showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
            whis=[5, 95],
        )
        sns.boxplot(
            data=second_merged[[f"{col}_pre", f"{col}_post"]],
            ax=ax[1],
            palette=palette,
            showfliers=False,
            showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
            whis=[5, 95],
        )
        sns.boxplot(
            data=merged_df[[f"{col}_pre", f"{col}_post"]],
            ax=ax[2],
            palette=palette,
            showfliers=False,
            showmeans=True,
            meanprops={
                "markerfacecolor": "#fea040",
                "markeredgecolor": "black",
                "markersize": "10",
            },
            whis=[5, 95],
        )
        ax[0].set_xticklabels(x_ticklabels)
        ax[0].set_ylabel(y_label)
    fig.suptitle(
        f"Pre- and post-manipulation"
        f" \n {explainer_plot_name} explanation similarities"
    )
    fig.tight_layout()
    plt.show()
    fig_save_path = os.path.join(
        save_path,
        f"{data_set_name}_{explainer_name}_{first_class_name}_"
        f"{second_class_name}_and_merged_similarities.png",
    )
    fig.savefig(
        fig_save_path,
        transparent=True,
    )


def plot_gamma_ablation_similarities():
    data_set_dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "data/preprocessed/adversarial/data-set=FashionMNIST--attack="
        "DeepFool--eps=0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False"
    )
    dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "logs/manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    gamma_to_run_id_mapping = {
        "0": "HAA-5271",
        "0.1": "HAA-5274",
        "0.2": "HAA-5275",
        "0.4": "HAA-5277",
        "0.6": "HAA-5278",
        "0.8": "HAA-5272",
        "1.0": "HAA-5253",
        "2.0": "HAA-5279",
    }
    res_df = pd.read_csv(
        os.path.join(
            data_set_dir,
            "test_similarities.csv",
        ),
        index_col=0,
    )
    res_df["gamma"] = -1.0
    target_label = 5
    results_df = res_df[res_df["orig_label"] == target_label]

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

    x_tick_labels = list(
        [rf"$\gamma$={gamma}" for gamma in gamma_to_run_id_mapping.keys()]
    )
    x_tick_labels.insert(0, "pre-manipulation")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, (col, ax, y_label, p, y_limits) in enumerate(
        zip(SIMILARITIES_COLS, axes, Y_LABELS, COLOR_PALETTES, SIM_Y_LIMITS)
    ):
        if col == "mse_sim":
            ax.set_yscale("log")
        for gamma, run_id in gamma_to_run_id_mapping.items():
            post_similarities_df = pd.read_csv(
                os.path.join(dir, run_id, "concat_post_test_similarities.csv"),
                index_col=0,
            )
            df = post_similarities_df[
                post_similarities_df["orig_label"] == target_label
            ]
            df.loc[:, "gamma"] = gamma
            results_df = results_df.append(df)

            sns.boxplot(
                data=results_df,
                y=col,
                x="gamma",
                ax=ax,
                # palette=p,
                showfliers=False,
                showmeans=True,
                meanprops={
                    "markerfacecolor": "#fea040",
                    "markeredgecolor": "black",
                    "markersize": "10",
                },
                whis=[5, 95],
            )

        ax.set_ylim(y_limits)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylabel(y_label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=-30)
    fig.suptitle(
        r"Explanation similarities comparison for different values of $\gamma$"
    )
    fig.tight_layout()
    fig.savefig(
        "/home/steffi/dev/master_thesis/evaluation/Fashion-MNIST,Grad-CAM/"
        "gamma_ablation_Sandal/"
        "fmnist_grad_cam_sandal_ablations_similarities_all_gammas_tab10.png",
        transparent=True,
    )
    plt.show()


def plot_ce_weight_ablation_similarities():
    data_dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data"
        "/preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool"
        "--eps=0.105--cp-run=HAA-1728/exp=GuidedBackprop"
    )
    dir = (
        "/home/steffi/dev/master_thesis/evaluation/Fashion-MNIST,"
        "Guided_Backprop/ce_weight_ablation"
    )
    ce_weight_to_run_id_mapping = {
        "1": "HAA-5576",
        "10": "HAA-5577",
        "100": "HAA-5578",
        "200": "HAA-5579",
        "400": "HAA-5581",
    }
    res_df = pd.read_csv(
        os.path.join(
            data_dir,
            "test_similarities.csv",
        ),
        index_col=0,
    )
    res_df["ce_weight"] = -1.0
    target_label = 4
    results_df = res_df[res_df["orig_label"] == target_label]

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

    x_tick_labels = list(
        [rf"$w_t={ce_weight}$" for ce_weight in ce_weight_to_run_id_mapping.keys()]
    )
    x_tick_labels.insert(0, "pre-manipulation")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, (col, ax, y_label, p, y_limits) in enumerate(
        zip(SIMILARITIES_COLS, axes, Y_LABELS, COLOR_PALETTES, SIM_Y_LIMITS)
    ):
        if col == "mse_sim":
            ax.set_yscale("log")
        for ce_weight, run_id in ce_weight_to_run_id_mapping.items():
            post_similarities_df = pd.read_csv(
                os.path.join(dir, run_id, "concat_post_test_similarities.csv"),
                index_col=0,
            )
            df = post_similarities_df[
                post_similarities_df["orig_label"] == target_label
            ]
            df.loc[:, "ce_weight"] = ce_weight
            results_df = results_df.append(df)

            sns.boxplot(
                data=results_df,
                y=col,
                x="ce_weight",
                ax=ax,
                palette="tab20",
                showfliers=False,
                showmeans=True,
                meanprops={
                    "markerfacecolor": "#fea040",
                    "markeredgecolor": "black",
                    "markersize": "10",
                },
                whis=[5, 95],
            )

        ax.set_ylim(y_limits)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylabel(y_label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=-30)
    fig.suptitle(
        "Explanation similarities for different "
        r"values of Cross Entropy class weight $w_t$"
    )
    fig.tight_layout()
    fig.savefig(
        "/home/steffi/dev/master_thesis/evaluation/Fashion-MNIST,Guided_Backprop/"
        "ce_weight_ablation/ce_class_weight_explanation_similarities.png",
        transparent=True,
    )
    plt.show()


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


def plot_fashion_mnist_pre_manipulation_similarity_boxplots_PCC():
    # Grad-CAM
    grad_cam_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False"
    )
    plot_pre_manipulation_similarities_PCC(
        grad_cam_path, DataSetNames.FASHION_MNIST, ExplainerNames.GRAD_CAM
    )

    # Guided Backprop
    guided_backprop_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=FashionMNIST--attack=DeepFool--"
        "eps=0.105--cp-run=HAA-1728/exp=GuidedBackprop"
    )
    plot_pre_manipulation_similarities_PCC(
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
    data_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False"
    )
    # Bottom Class: Coat
    bottom_class_id = 4
    bottom_run_id = 5359

    # Top Class: Sandal
    top_class_id = 5
    top_run_id = 5360

    top_run_path = os.path.join(runs_path, f"HAA-{top_run_id}")
    bottom_run_path = os.path.join(runs_path, f"HAA-{bottom_run_id}")
    plot_pre_and_post_manipulation_boxplot_similarities_multiclass(
        data_path,
        top_run_path,
        bottom_run_path,
        DataSetNames.FASHION_MNIST,
        ExplainerNames.GRAD_CAM,
        top_class_id,
        bottom_class_id,
    )


def plot_fashion_mnist_guided_backprop_pre_and_post_manipulation_boxplots():
    runs_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/logs/"
        "manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    data_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=GuidedBackprop"
    )
    # Bottom class: Sandal
    bottom_class_id = 5
    bottom_run_id = 5502

    # Top class: Coat
    top_class_id = 4
    top_run_id = 5493
    top_run_path = os.path.join(runs_path, f"HAA-{top_run_id}")
    bottom_run_path = os.path.join(runs_path, f"HAA-{bottom_run_id}")
    plot_pre_and_post_manipulation_boxplot_similarities_multiclass(
        data_path,
        top_run_path,
        bottom_run_path,
        DataSetNames.FASHION_MNIST,
        ExplainerNames.GUIDED_BACKPROP,
        top_class_id,
        bottom_class_id,
    )


def plot_grad_cam_additional_classes_pre_and_post_boxplots():
    runs_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/logs/"
        "manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    data_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False"
    )
    save_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "reports/figures/quantitative_evaluation/appendix"
    )
    # Trousers
    first_class_id = 1
    first_run_id = 5602

    # Dress
    second_class_id = 3
    second_run_id = 5610

    # Trouser & Dress
    merged_run_id = 5612

    first_run_path = os.path.join(runs_path, f"HAA-{first_run_id}")
    second_run_path = os.path.join(runs_path, f"HAA-{second_run_id}")
    merged_run_path = os.path.join(runs_path, f"HAA-{merged_run_id}")
    plot_pre_and_post_manipulation_boxplot_similarities_merged(
        data_path,
        first_run_path,
        second_run_path,
        merged_run_path,
        DataSetNames.FASHION_MNIST,
        ExplainerNames.GRAD_CAM,
        first_class_id,
        second_class_id,
        save_path,
    )


def plot_guided_backprop_additional_classes_pre_and_post_boxplots():
    runs_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/logs/"
        "manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    data_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=GuidedBackprop"
    )
    save_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "reports/figures/quantitative_evaluation/appendix"
    )
    # Trousers
    first_class_id = 1
    first_run_id = 5583

    # Dress
    second_class_id = 3
    second_run_id = 5613

    # Trouser & Dress
    merged_run_id = 5599

    first_run_path = os.path.join(runs_path, f"HAA-{first_run_id}")
    second_run_path = os.path.join(runs_path, f"HAA-{second_run_id}")
    merged_run_path = os.path.join(runs_path, f"HAA-{merged_run_id}")
    plot_pre_and_post_manipulation_boxplot_similarities_merged(
        data_path,
        first_run_path,
        second_run_path,
        merged_run_path,
        DataSetNames.FASHION_MNIST,
        ExplainerNames.GUIDED_BACKPROP,
        first_class_id,
        second_class_id,
        save_path,
    )


def plot_integrated_gradients_additional_classes_pre_and_post_boxplots():
    runs_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/logs/"
        "manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    data_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=GuidedBackprop"
    )
    save_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "reports/figures/quantitative_evaluation/appendix"
    )
    # Trousers
    first_class_id = 1
    first_run_id = 5623

    # Dress
    second_class_id = 3
    second_run_id = 5626

    # Trouser & Dress
    merged_run_id = 5643

    first_run_path = os.path.join(runs_path, f"HAA-{first_run_id}")
    second_run_path = os.path.join(runs_path, f"HAA-{second_run_id}")
    merged_run_path = os.path.join(runs_path, f"HAA-{merged_run_id}")
    plot_pre_and_post_manipulation_boxplot_similarities_merged(
        data_path,
        first_run_path,
        second_run_path,
        merged_run_path,
        DataSetNames.FASHION_MNIST,
        ExplainerNames.INTEGRATED_GRADIENTS,
        first_class_id,
        second_class_id,
        save_path,
    )


def plot_input_x_gradient_additional_classes_pre_and_post_boxplots():
    runs_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/logs/"
        "manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    data_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool--eps="
        "0.105--cp-run=HAA-1728/exp=InputXGradient"
    )
    save_path = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "reports/figures/quantitative_evaluation/appendix"
    )
    # Trousers
    first_class_id = 1
    first_run_id = 5639

    # Dress
    second_class_id = 3
    second_run_id = 5653

    # Trouser & Dress
    merged_run_id = 5656

    first_run_path = os.path.join(runs_path, f"HAA-{first_run_id}")
    second_run_path = os.path.join(runs_path, f"HAA-{second_run_id}")
    merged_run_path = os.path.join(runs_path, f"HAA-{merged_run_id}")
    plot_pre_and_post_manipulation_boxplot_similarities_merged(
        data_path,
        first_run_path,
        second_run_path,
        merged_run_path,
        DataSetNames.FASHION_MNIST,
        ExplainerNames.INPUT_X_GRADIENT,
        first_class_id,
        second_class_id,
        save_path,
    )


if __name__ == "__main__":
    plot_fashion_mnist_grad_cam_pre_and_post_manipulation_boxplots()
    plot_fashion_mnist_guided_backprop_pre_and_post_manipulation_boxplots()
    plot_fashion_mnist_pre_manipulation_similarity_boxplots()
    plot_fashion_mnist_pre_manipulation_similarity_boxplots_PCC()
    plot_gamma_ablation_similarities()
    plot_ce_weight_ablation_similarities()
    plot_grad_cam_additional_classes_pre_and_post_boxplots()
    plot_guided_backprop_additional_classes_pre_and_post_boxplots()
    plot_integrated_gradients_additional_classes_pre_and_post_boxplots()
    plot_input_x_gradient_additional_classes_pre_and_post_boxplots()
