import os

import pandas as pd
import torch

from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames
from hiding_adversarial_attacks.eda.visualization import (
    plot_similarities_histogram_with_boxplot,
)
from hiding_adversarial_attacks.manipulation.utils import get_similarities

data_set_mappings = {
    DataSetNames.FASHION_MNIST: {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    },
    DataSetNames.CIFAR10: {
        0: "Airplane",
        1: "Car",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Ship",
        9: "Truck",
    },
}


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
    _, test_similarities_mse = get_similarities("MSE", test_expl_orig, test_expl_adv)
    _, train_similarities_pcc = get_similarities("PCC", train_expl_orig, train_expl_adv)
    _, test_similarities_pcc = get_similarities("PCC", test_expl_orig, test_expl_adv)
    return (
        train_similarities_pcc,
        train_similarities_mse,
        train_label_orig,
        train_label_adv,
        test_similarities_pcc,
        test_similarities_mse,
        test_label_orig,
        test_label_adv,
    )


def create_sorted_similarities_df(
    similarities_mse,
    similarities_pcc,
    label_orig,
    label_adv,
    data_set_name,
):
    sim_df = pd.DataFrame(
        [
            similarities_mse.numpy(),
            similarities_pcc.numpy(),
            label_orig.numpy(),
            label_adv.numpy(),
        ],
        index=["mse_sim", "pcc_sim", "orig_label", "adv_label"],
    ).T
    sim_df["orig_label"] = sim_df["orig_label"].astype(int)
    sim_df["adv_label"] = sim_df["adv_label"].astype(int)
    sim_df["orig_label_name"] = sim_df["orig_label"].map(
        data_set_mappings[data_set_name]
    )
    sorted_df_sim = sim_df.sort_values(by="orig_label")
    return sorted_df_sim


def plot_initial_similarities(
    data_path: str, data_set_name: str, output_path: str = None
):
    (
        train_similarities_pcc,
        train_similarities_mse,
        train_label_orig,
        train_label_adv,
        test_similarities_pcc,
        test_similarities_mse,
        test_label_orig,
        test_label_adv,
    ) = get_similarity_tensors(data_path)

    train_sim_df = create_sorted_similarities_df(
        train_similarities_mse,
        train_similarities_pcc,
        train_label_orig,
        train_label_adv,
        data_set_name,
    )
    test_sim_df = create_sorted_similarities_df(
        test_similarities_mse,
        test_similarities_pcc,
        test_label_orig,
        test_label_adv,
        data_set_name,
    )

    train_mse = plot_similarities_histogram_with_boxplot(
        train_sim_df,
        "orig_label_name",
        "mse_sim",
        f"{data_set_name} train -initial explanation similarities (MSE) by class",
        log_x=True,
        palette="afmhot_r",
    )
    train_pcc = plot_similarities_histogram_with_boxplot(
        train_sim_df,
        "orig_label_name",
        "pcc_sim",
        f"{data_set_name} train - initial explanation similarities (PCC) by class",
        palette="PuRd",
    )

    test_mse = plot_similarities_histogram_with_boxplot(
        test_sim_df,
        "orig_label_name",
        "mse_sim",
        f"{data_set_name} test - initial explanation similarities (MSE) by class",
        log_x=True,
        palette="afmhot_r",
    )
    test_pcc = plot_similarities_histogram_with_boxplot(
        test_sim_df,
        "orig_label_name",
        "pcc_sim",
        f"{data_set_name} test - initial explanation similarities (PCC) by class",
        palette="PuRd",
    )
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        train_mse.savefig(
            os.path.join(output_path, f"{data_set_name}_train_mse_sim.png"),
            transparent=True,
        )
        test_mse.savefig(
            os.path.join(output_path, f"{data_set_name}_test_mse_sim.png"),
            transparent=True,
        )
        train_pcc.savefig(
            os.path.join(output_path, f"{data_set_name}_train_pcc_sim.png"),
            transparent=True,
        )
        test_pcc.savefig(
            os.path.join(output_path, f"{data_set_name}_test_pcc_sim.png"),
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
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/"
        "preprocessed/adversarial/data-set=FashionMNIST--attack=DeepFool"
        "--eps=0.105--cp-run=HAA-1728/exp=GradCAM--l=conv2--ra=False",
    )


if __name__ == "__main__":
    plot_fashion_mnist_similarities()
