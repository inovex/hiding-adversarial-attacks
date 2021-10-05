import os

import pandas as pd
import seaborn as sns
from matplotlib import pylab
from matplotlib import pyplot as plt

from hiding_adversarial_attacks.eda.utils import load_test_results_as_df


def plot_and_save_aors(directory):
    results_df = load_test_results_as_df(directory)
    fig1, fig2 = plot_aors(results_df)
    fig1.savefig(os.path.join(directory, "aor.png"), transparent=True)
    fig2.savefig(os.path.join(directory, "aor_class.png"), transparent=True)


def plot_aors(test_results_df, titles=None):
    aor_df = get_aor_df_from_results(test_results_df, column_name="test_aor_tau")
    aor_df_class = get_aor_df_from_results(
        test_results_df, column_name="test_aor_class_tau"
    )

    title1, title2 = titles if titles is not None else ("", "")
    fig1 = _plot_aor(aor_df, title1)
    fig2 = _plot_aor(aor_df_class, title2)
    return fig1, fig2


def get_aor_df_from_results(test_results_df, column_name="test_aor_class_tau"):
    aor_df_class = test_results_df.filter(regex=f"^{column_name}*")
    aor_df_class.columns = [
        float(col.replace(f"{column_name}=", "")) for col in aor_df_class.columns
    ]
    aor_df_class = aor_df_class.T
    return aor_df_class


def plot_top_and_bottom_aor(aor_dfs, class_names, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    palette = sns.color_palette("Paired", 10)
    for i, aor_df, class_name in zip(range(6, 9, 2), aor_dfs, class_names):
        ax = aor_df.plot.line(
            style=["o-", "^-"], ax=ax, color=[palette[i], palette[i + 1]]
        )
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("AOR")
    ax.set_ylim(bottom=0, top=1)
    fig.suptitle(title)
    fig.show()
    return fig


def _plot_aor(aor_df, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = aor_df.plot.line(
        style=["o-", "^-"], color={"pre": "slateblue", "post": "purple"}, ax=ax
    )
    ax.legend(["pre-manipulation", "post-manipulation"])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("AOR")
    ax.set_ylim(bottom=0, top=1)
    fig.suptitle(title)
    fig.show()
    return fig


def plot_grad_cam_top_and_bottom_aor():
    dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "logs/manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    aor_dfs = []

    params = {
        "legend.fontsize": 16,
        "figure.figsize": (8, 6),
        "figure.titlesize": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    pylab.rcParams.update(params)

    # top class sandal
    top_id = "HAA-5360"
    pre = pd.read_csv(os.path.join(dir, top_id, "pre-test_results.csv"), index_col=0)
    post = pd.read_csv(
        os.path.join(dir, top_id, "concat_post_test_results.csv"), index_col=0
    )
    top_results_df = pre.append(post.loc["mean"])
    top_results_df.index = [
        "Sandal pre-manipulation (top class)",
        "Sandal post-manipulation (top class)",
    ]
    top_aor_class_df = get_aor_df_from_results(
        top_results_df, column_name="test_aor_class_tau"
    )
    aor_dfs.append(top_aor_class_df)

    # bottom class coat
    bottom_id = "HAA-5359"
    pre = pd.read_csv(os.path.join(dir, bottom_id, "pre-test_results.csv"), index_col=0)
    post = pd.read_csv(
        os.path.join(dir, bottom_id, "concat_post_test_results.csv"),
        index_col=0,
    )
    bottom_results_df = pre.append(post.loc["mean"])
    bottom_results_df.index = [
        "Coat pre-manipulation (bottom class)",
        "Coat post-manipulation (bottom class)",
    ]
    bottom_aor_class_df = get_aor_df_from_results(
        bottom_results_df, column_name="test_aor_class_tau"
    )
    aor_dfs.append(bottom_aor_class_df)
    fig = plot_top_and_bottom_aor(
        aor_dfs,
        class_names=["Sandal", "Coat"],
        title="Grad-CAM Adversarial Obfuscation Rate (AOR) curves",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(dir, bottom_id, "aor_bottom_top.png"), transparent=True)
    fig.savefig(os.path.join(dir, top_id, "aor_bottom_top.png"), transparent=True)


def plot_guided_backprop_top_and_bottom_aor():
    dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "logs/manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    aor_dfs = []

    params = {
        "legend.fontsize": 16,
        "figure.figsize": (8, 6),
        "figure.titlesize": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    pylab.rcParams.update(params)

    # top class Coat
    top_id = "HAA-5493"
    pre = pd.read_csv(os.path.join(dir, top_id, "pre-test_results.csv"), index_col=0)
    post = pd.read_csv(
        os.path.join(dir, top_id, "concat_post_test_results.csv"), index_col=0
    )
    top_results_df = pre.append(post.loc["mean"])
    top_results_df.index = [
        "Coat pre-manipulation (top class)",
        "Coat post-manipulation (top class)",
    ]
    top_aor_class_df = get_aor_df_from_results(
        top_results_df, column_name="test_aor_class_tau"
    )
    aor_dfs.append(top_aor_class_df)

    # bottom class Sandal
    bottom_id = "HAA-5502"
    pre = pd.read_csv(os.path.join(dir, bottom_id, "pre-test_results.csv"), index_col=0)
    post = pd.read_csv(
        os.path.join(dir, bottom_id, "concat_post_test_results.csv"),
        index_col=0,
    )
    bottom_results_df = pre.append(post.loc["mean"])
    bottom_results_df.index = [
        "Sandal pre-manipulation (bottom class)",
        "Sandal post-manipulation (bottom class)",
    ]
    bottom_aor_class_df = get_aor_df_from_results(
        bottom_results_df, column_name="test_aor_class_tau"
    )
    aor_dfs.append(bottom_aor_class_df)
    fig = plot_top_and_bottom_aor(
        aor_dfs,
        class_names=["Coat", "Sandal"],
        title="Guided Backpropagation Adversarial Obfuscation Rate (AOR) curves",
    )
    fig.savefig(os.path.join(dir, bottom_id, "aor_bottom_top.png"), transparent=True)
    fig.savefig(os.path.join(dir, top_id, "aor_bottom_top.png"), transparent=True)


def plot_gamma_ablation_aor_grad_cam_sandal():
    dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "logs/manipulate_model/AdversarialFashionMNISTWithExplanations"
    )
    gamma_to_run_id_mapping = {
        "0": "HAA-5271",
        "0.1": "HAA-5274",
        "0.2": "HAA-5275",
        "0.4": "HAA-5277",
        # "0.6": "HAA-5278",
        "0.8": "HAA-5272",
        "1.0": "HAA-5253",
        # "2.0": "HAA-5279",
    }
    palette_tab10 = sns.color_palette("tab10", 10)
    palette = sns.color_palette(
        [
            palette_tab10[0],
            palette_tab10[1],
            palette_tab10[2],
            palette_tab10[3],
            palette_tab10[4],
            palette_tab10[6],
            palette_tab10[7],
        ]
    )
    results_df = pd.read_csv(
        os.path.join(dir, gamma_to_run_id_mapping["0"], "pre-test_results.csv"),
        index_col=0,
    )
    for run_id in gamma_to_run_id_mapping.values():
        post = pd.read_csv(
            os.path.join(dir, run_id, "concat_post_test_results.csv"),
            index_col=0,
        )
        results_df = results_df.append(post.loc["mean"])
    new_index = [rf"$\gamma$={gamma}" for gamma in gamma_to_run_id_mapping.keys()]
    new_index.insert(0, "pre-manipulation")
    results_df.index = new_index
    aor_class_df = get_aor_df_from_results(results_df, column_name="test_aor_class_tau")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = aor_class_df.plot.line(ax=ax, color=palette)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("AOR")
    ax.set_ylim(bottom=0, top=1)
    fig.suptitle(r"AOR curves for different values of $\gamma$")
    fig.show()


if __name__ == "__main__":
    plot_gamma_ablation_aor_grad_cam_sandal()
    plot_guided_backprop_top_and_bottom_aor()
    plot_grad_cam_top_and_bottom_aor()
