import os
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import lines
from matplotlib import pyplot as plt


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
                data=g_nonzero, x=similarity_col, ax=ax2, color=color_palette[-2]
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

            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            ax.tick_params(axis="both", labelsize=14)
            ax.title.set_size(16)
        except Exception as e:
            print(f"EXCEPTION: {e}")

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


def load_test_results_as_df(directory: str):
    pre = pd.read_csv(os.path.join(directory, "pre-pre-test_results.csv"), index_col=0)
    post = pd.read_csv(os.path.join(directory, "post-test_results.csv"), index_col=0)
    results_df = pre.append(post)
    results_df.index = ["pre", "post"]
    return results_df


def plot_aor(directory: str, save: bool = True):
    results_df = load_test_results_as_df(directory)
    aor_df = results_df.filter(regex="^test_aor*")
    aor_df.columns = [float(col.replace("test_aor_tau=", "")) for col in aor_df.columns]
    aor_df = aor_df.T
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = aor_df.plot.line(
        style=["o-", "^-"], color={"pre": "slateblue", "post": "purple"}, ax=ax
    )
    ax.legend(["pre-manipulation", "post-manipulation"])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("AOR")
    ax.set_ylim(bottom=0, top=1)
    fig.show()
    if save:
        fig.savefig(os.path.join(directory, "aor.png"), transparent=True)


if __name__ == "__main__":
    dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "logs/manipulate_model/AdversarialFashionMNISTWithExplanations/HAA-2135"
    )
    plot_aor(dir)
