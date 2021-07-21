import os

import pandas as pd
from matplotlib import pyplot as plt


def load_test_results_as_df(directory: str):
    pre = pd.read_csv(os.path.join(directory, "pre-pre-test_results.csv"), index_col=0)
    post = pd.read_csv(os.path.join(directory, "post-test_results.csv"), index_col=0)
    results_df = pre.append(post)
    results_df.index = ["pre", "post"]
    return results_df


def plot_aors(directory: str, save: bool = True):
    results_df = load_test_results_as_df(directory)
    aor_df = results_df.filter(regex="^test_aor_tau*")
    aor_df.columns = [float(col.replace("test_aor_tau=", "")) for col in aor_df.columns]
    aor_df = aor_df.T

    aor_df_class = results_df.filter(regex="^test_aor_class_tau*")
    aor_df_class.columns = [
        float(col.replace("test_aor_class_tau=", "")) for col in aor_df_class.columns
    ]
    aor_df_class = aor_df_class.T

    fig1 = _plot_aor(aor_df)
    fig2 = _plot_aor(aor_df_class)
    if save:
        fig1.savefig(os.path.join(directory, "aor.png"), transparent=True)
        fig2.savefig(os.path.join(directory, "aor_class.png"), transparent=True)


def _plot_aor(aor_df):
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
    return fig


if __name__ == "__main__":
    dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "logs/manipulate_model/AdversarialFashionMNISTWithExplanations/HAA-2135"
    )
    plot_aors(dir)
