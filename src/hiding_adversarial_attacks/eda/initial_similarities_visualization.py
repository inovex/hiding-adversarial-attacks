import os

import pandas as pd
import torch

from hiding_adversarial_attacks.manipulation.utils import get_similarities
from hiding_adversarial_attacks.visualization.config import DATA_SET_MAPPING
from hiding_adversarial_attacks.visualization.explanation_similarities import (
    plot_similarities_histogram_with_boxplot,
)


def load_explanations(data_set_path: str, test: bool = False):
    stage = "test" if test else "training"
    expl_orig, labels_orig, _ = torch.load(
        os.path.join(data_set_path, f"{stage}_orig_exp.pt")
    )
    expl_adv, labels_adv, _ = torch.load(
        os.path.join(data_set_path, f"{stage}_adv_exp.pt")
    )
    return expl_orig, labels_orig, expl_adv, labels_adv


def get_similarity_tensors(data_path):
    (
        train_expl_orig,
        train_label_orig,
        train_expl_adv,
        train_label_adv,
    ) = load_explanations(data_path)
    (
        test_expl_orig,
        test_label_orig,
        test_expl_adv,
        test_label_adv,
    ) = load_explanations(data_path, test=True)
    _, train_similarities_mse = get_similarities("MSE", train_expl_orig, train_expl_adv)
    _, train_similarities_ssim = get_similarities(
        "SSIM", train_expl_orig, train_expl_adv
    )
    _, train_similarities_pcc = get_similarities("PCC", train_expl_orig, train_expl_adv)
    _, test_similarities_mse = get_similarities("MSE", test_expl_orig, test_expl_adv)
    _, test_similarities_ssim = get_similarities("SSIM", test_expl_orig, test_expl_adv)
    _, test_similarities_pcc = get_similarities("PCC", test_expl_orig, test_expl_adv)
    return (
        train_similarities_pcc,
        train_similarities_mse,
        train_similarities_ssim,
        train_label_orig,
        train_label_adv,
        test_similarities_pcc,
        test_similarities_mse,
        test_similarities_ssim,
        test_label_orig,
        test_label_adv,
    )


def create_sorted_similarities_df(
    similarities_mse,
    similarities_pcc,
    similarities_ssim,
    label_orig,
    label_adv,
    data_set_name,
):
    sim_df = pd.DataFrame(
        [
            similarities_mse.numpy(),
            similarities_pcc.numpy(),
            similarities_ssim.numpy(),
            label_orig.numpy(),
            label_adv.numpy(),
        ],
        index=["mse_sim", "pcc_sim", "ssim_sim", "orig_label", "adv_label"],
    ).T
    sim_df["orig_label"] = sim_df["orig_label"].astype(int)
    sim_df["adv_label"] = sim_df["adv_label"].astype(int)
    sim_df["orig_label_name"] = sim_df["orig_label"].map(
        DATA_SET_MAPPING[data_set_name]
    )
    sorted_df_sim = sim_df.sort_values(by="orig_label")
    return sorted_df_sim


def plot_initial_similarities(
    data_path: str,
    data_set_name: str,
    explainer_name: str,
    output_path: str = None,
):
    (
        train_similarities_pcc,
        train_similarities_mse,
        train_similarities_ssim,
        train_label_orig,
        train_label_adv,
        test_similarities_pcc,
        test_similarities_mse,
        test_similarities_ssim,
        test_label_orig,
        test_label_adv,
    ) = get_similarity_tensors(data_path)

    train_sim_df = create_sorted_similarities_df(
        train_similarities_mse,
        train_similarities_pcc,
        train_similarities_ssim,
        train_label_orig,
        train_label_adv,
        data_set_name,
    )
    test_sim_df = create_sorted_similarities_df(
        test_similarities_mse,
        test_similarities_pcc,
        test_similarities_ssim,
        test_label_orig,
        test_label_adv,
        data_set_name,
    )

    train_mse = plot_similarities_histogram_with_boxplot(
        train_sim_df,
        "orig_label_name",
        "mse_sim",
        f"{data_set_name} train — initial {explainer_name} explanation map similarities"
        f" (MSE) by class",
        log_x=True,
        palette="afmhot_r",
    )
    train_pcc = plot_similarities_histogram_with_boxplot(
        train_sim_df,
        "orig_label_name",
        "pcc_sim",
        f"{data_set_name} train — initial {explainer_name} explanation map similarities"
        f" (PCC) by class",
        palette="PuRd",
    )

    test_mse = plot_similarities_histogram_with_boxplot(
        test_sim_df,
        "orig_label_name",
        "mse_sim",
        f"{data_set_name} test — initial {explainer_name} explanation map similarities"
        f" (MSE) by class",
        log_x=True,
        palette="afmhot_r",
    )
    test_pcc = plot_similarities_histogram_with_boxplot(
        test_sim_df,
        "orig_label_name",
        "pcc_sim",
        f"{data_set_name} test — initial {explainer_name} explanation map similarities"
        f" (PCC) by class",
        palette="PuRd",
    )
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        # Save stats as csv
        train_sim_df.to_csv(os.path.join(output_path, "train_similarities.csv"))
        test_sim_df.to_csv(os.path.join(output_path, "test_similarities.csv"))

        train_stats = (
            train_sim_df[["mse_sim", "pcc_sim", "orig_label_name"]]
            .groupby("orig_label_name")
            .agg(["mean", "std", "median"])
        )
        train_stats.to_csv(os.path.join(output_path, "train_sim_stats.csv"))
        test_stats = (
            test_sim_df[["mse_sim", "pcc_sim", "orig_label_name"]]
            .groupby("orig_label_name")
            .agg(["mean", "std", "median"])
        )
        test_stats.to_csv(os.path.join(output_path, "test_sim_stats.csv"))

        train_mse.savefig(
            os.path.join(
                output_path,
                f"{data_set_name}_{explainer_name}_train_mse_sim.png",
            ),
            transparent=True,
        )
        test_mse.savefig(
            os.path.join(
                output_path,
                f"{data_set_name}_{explainer_name}_test_mse_sim.png",
            ),
            transparent=True,
        )
        train_pcc.savefig(
            os.path.join(
                output_path,
                f"{data_set_name}_{explainer_name}_train_pcc_sim.png",
            ),
            transparent=True,
        )
        test_pcc.savefig(
            os.path.join(
                output_path,
                f"{data_set_name}_{explainer_name}_test_pcc_sim.png",
            ),
            transparent=True,
        )


def plot_fashion_mnist_similarities():
    # Input X Gradient
    plot_initial_similarities(
        "/home/steffi/dev/master_thesis/"
        "hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=FashionMNIST--attack=DeepFool"
        "--eps=0.105--cp-run=HAA-1728/exp=InputXGradient",
        "FashionMNIST",
        "InputXGradient",
        "/home/steffi/dev/master_thesis/"
        "hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=FashionMNIST--attack=DeepFool"
        "--eps=0.105--cp-run=HAA-1728/exp=InputXGradient",
    )
    # Grad-CAM
    plot_initial_similarities(
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool"
        "--eps=0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False",
        "FashionMNIST",
        "Grad-CAM",
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool"
        "--eps=0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False",
    )


def plot_cifar10_similarities():
    # Input X Gradient
    plot_initial_similarities(
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=CIFAR10--attack=DeepFool--eps=0.1--cp-run=resnet18/"
        "exp=InputXGradient",
        "CIFAR10",
        "InputXGradient",
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=CIFAR10--attack=DeepFool--eps=0.1--cp-run=resnet18/"
        "exp=InputXGradient",
    )
    # Grad-CAM
    plot_initial_similarities(
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=CIFAR10--attack=DeepFool--eps=0.1--cp-run=resnet18/"
        "exp=GradCAM--l=model.layer2.1.conv2--ra=False",
        "CIFAR10",
        "Grad-CAM",
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/"
        "adversarial/data-set=CIFAR10--attack=DeepFool--eps=0.1--cp-run=resnet18/"
        "exp=GradCAM--l=model.layer2.1.conv2--ra=False",
    )


if __name__ == "__main__":
    plot_cifar10_similarities()
    plot_fashion_mnist_similarities()
