import os

from matplotlib import pyplot as plt

from hiding_adversarial_attacks.eda.utils import load_test_results_as_df


def plot_and_save_aors(directory):
    results_df = load_test_results_as_df(directory)
    fig1, fig2 = plot_aors(results_df)
    fig1.savefig(os.path.join(directory, "aor.png"), transparent=True)
    fig2.savefig(os.path.join(directory, "aor_class.png"), transparent=True)


def plot_aors(test_results_df, titles=None):
    aor_df = test_results_df.filter(regex="^test_aor_tau*")
    aor_df.columns = [float(col.replace("test_aor_tau=", "")) for col in aor_df.columns]
    aor_df = aor_df.T

    aor_df_class = test_results_df.filter(regex="^test_aor_class_tau*")
    aor_df_class.columns = [
        float(col.replace("test_aor_class_tau=", "")) for col in aor_df_class.columns
    ]
    aor_df_class = aor_df_class.T

    title1, title2 = titles if titles is not None else ("", "")
    fig1 = _plot_aor(aor_df, title1)
    fig2 = _plot_aor(aor_df_class, title2)
    return fig1, fig2


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


if __name__ == "__main__":
    dir = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
        "logs/manipulate_model/AdversarialFashionMNISTWithExplanations/HAA-5135"
    )
    plot_and_save_aors(dir)
