import os

import pandas as pd

from hiding_adversarial_attacks.evaluation.utils import (
    load_pre_and_post_test_results_for_run_ids,
)
from hiding_adversarial_attacks.visualization.adversarial_obfuscation_rate import (
    plot_aors,
)


def aggregate_test_results_for_run_ids(directory, run_ids):
    pre_df, post_df = load_pre_and_post_test_results_for_run_ids(directory, run_ids)
    pre_df = pre_df.append(pre_df.agg(["mean", "std"], axis="index"))
    post_df = post_df.append(post_df.agg(["mean", "std"], axis="index"))

    pre_df.to_csv(os.path.join(directory, "pre_manipulation_results.csv"))
    post_df.to_csv(os.path.join(directory, "post_manipulation_results.csv"))

    classification_cols = [
        "test_orig_acc",
        "test_f1_score",
        "test_adv_accuracy_class",
    ]
    pre_df[classification_cols].to_csv(
        os.path.join(directory, "pre_manipulation_classification_results.csv")
    )
    post_df[classification_cols].to_csv(
        os.path.join(directory, "post_manipulation_classification_results.csv")
    )
    return pre_df, post_df


def save_test_results_for_fashion_mnist_guided_backprop():
    fmnist_top_coat_dir = (
        "/home/steffi/dev/master_thesis/evaluation"
        "/Fashion-MNIST,Guided_Backprop/top=Coat-4/test_runs"
    )
    fmnist_bottom_sandal_dir = (
        "/home/steffi/dev/master_thesis/evaluation"
        "/Fashion-MNIST,Guided_Backprop/bottom=Sandal-5/test_runs"
    )
    fmnist_guided_backprop_coat_run_ids = range(4928, 4934)
    fmnist_guided_backprop_sandal_run_ids = range(5138, 5143)
    # Top: Coat
    aggregate_test_results_for_run_ids(
        fmnist_top_coat_dir, fmnist_guided_backprop_coat_run_ids
    )
    # Bottom: Sandal
    aggregate_test_results_for_run_ids(
        fmnist_bottom_sandal_dir, fmnist_guided_backprop_sandal_run_ids
    )


def save_test_results_for_fashion_mnist_gradcam():
    fmnist_top_sandal_dir = (
        "/home/steffi/dev/master_thesis/evaluation"
        "/Fashion-MNIST,Grad-CAM/top=Sandal-5/test_runs"
    )
    fmnist_bottom_coat_dir = (
        "/home/steffi/dev/master_thesis/evaluation"
        "/Fashion-MNIST,Grad-CAM/bottom=Coat-4/test_runs"
    )
    fmnist_gradcam_coat_run_ids = range(5198, 5203)
    fmnist_gradcam_sandal_run_ids = range(5203, 5208)

    # Top: Sandal
    sandal_pre_df, sandal_post_df = aggregate_test_results_for_run_ids(
        fmnist_top_sandal_dir, fmnist_gradcam_sandal_run_ids
    )
    # Bottom: Coat
    coat_pre_df, coat_post_df = aggregate_test_results_for_run_ids(
        fmnist_bottom_coat_dir, fmnist_gradcam_coat_run_ids
    )

    # Save AOR plots
    # Coat
    coat_aor_df = pd.DataFrame([coat_pre_df.loc["mean"], coat_post_df.loc["mean"]])
    coat_aor_df.index = ["pre", "post"]
    fig1, fig2 = plot_aors(
        coat_aor_df,
        titles=[
            "Grad-CAM: Pre- and post-manipulation AOR curves for bottom class 'Coat'",
            "Grad-CAM: Pre- and post-manipulation AOR curves for all classes",
        ],
    )
    fig1.savefig(os.path.join(fmnist_bottom_coat_dir, "aor.png"), transparent=True)
    fig2.savefig(
        os.path.join(fmnist_bottom_coat_dir, "aor_class.png"), transparent=True
    )

    # Sandal
    sandal_aor_df = pd.DataFrame(
        [sandal_pre_df.loc["mean"], sandal_post_df.loc["mean"]]
    )
    sandal_aor_df.index = ["pre", "post"]
    fig1, fig2 = plot_aors(
        sandal_aor_df,
        titles=[
            "Grad-CAM: Pre- and post-manipulation AOR curves for top class 'Sandal'",
            "Grad-CAM: Pre- and post-manipulation AOR curves for all classes",
        ],
    )
    fig1.savefig(os.path.join(fmnist_top_sandal_dir, "aor.png"), transparent=True)
    fig2.savefig(os.path.join(fmnist_top_sandal_dir, "aor_class.png"), transparent=True)


if __name__ == "__main__":
    save_test_results_for_fashion_mnist_gradcam()
    # df = load_test_results_as_df(fmnist_top_sandal_dir, )
    print("")
