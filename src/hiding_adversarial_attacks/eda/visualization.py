from typing import List

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
):
    group = df[[similarity_col, group_by_col]].groupby(group_by_col, sort=False)
    means = group.mean().values.flatten()

    fig, axes = plt.subplots(5, 2, figsize=(16, 12), sharex=True, sharey=True)
    color_p = ["Wistia", "PuRd"]
    if similarity_col == "mse_sim":
        color_p = ["PuRd", "Wistia"]
    color_palette = sns.color_palette(color_p[0], 5)
    p = sns.color_palette(color_p[1], 1)

    for (label, g), ax, mean in zip(group, axes.flatten(), means):
        if log_x and not g[similarity_col].any(0):
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.set_title(label)
        ax2 = ax.twinx()

        try:
            ax = sns.histplot(g, ax=ax, palette=p, label=similarity_col)
            y_lim = int(ax.get_ylim()[1] * ylim_factor)
            ax.set(ylim=(-5, y_lim))
            sns.boxplot(data=g, x=similarity_col, ax=ax2, color=color_palette[2])
            ax2.set(ylim=(-5, 1))
            ax2.axvline(
                mean,
                color=color_palette[3],
                linestyle="dashed",
                linewidth=5,
                label="mean",
            )
            ax.legend(loc="lower left")
        except ValueError as e:
            print(f"WARNING: {e}")

    fig.suptitle(title, fontsize=16)
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
