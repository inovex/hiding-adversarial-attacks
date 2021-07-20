import os

import hydra
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from hiding_adversarial_attacks.classifiers.utils import get_model_from_checkpoint
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.utils import get_data_module
from hiding_adversarial_attacks.eda.visualization import (
    plot_similarities_histogram_with_boxplot,
    plot_similarities_kde,
)
from hiding_adversarial_attacks.manipulation.utils import (
    get_manipulatable_model,
    get_similarities,
)

PCC_COLOR_PALETTE = "PuRd"
MSE_COLOR_PALETTE = "afmhot_r"

data_set_mappings = {
    AdversarialDataSetNames.ADVERSARIAL_MNIST: {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    },
    AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST: {
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
    AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST_EXPL: {
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
    AdversarialDataSetNames.ADVERSARIAL_CIFAR10: {
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
    AdversarialDataSetNames.ADVERSARIAL_CIFAR10_EXPL: {
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


def save_classification_confidences(
    data_path: str,
    data_set_name: str,
    model_checkpoint: str,
    device: torch.device,
):
    model = get_model_from_checkpoint(data_set_name, model_checkpoint, device)
    model.eval()
    model.freeze()

    data_module = get_data_module(
        data_set=data_set_name,
        data_path=data_path,
        download=False,
        batch_size=64,
        val_split=0.0,
        transform=None,
        random_seed=42,
    )

    train_loader = data_module.train_dataloader(shuffle=False)

    adv_confidences = []
    adv_labels = []
    orig_confidences = []
    orig_labels = []

    orig_accuracy = Accuracy()
    adv_accuracy = Accuracy()

    for batch in tqdm(train_loader):
        (
            original_images,
            adversarial_images,
            original_labels,
            adversarial_labels,
            batch_indices,
        ) = batch
        adv_pred_softmax = model(adversarial_images.to(device))
        adv_pred_conf = torch.exp(adv_pred_softmax)
        adv_accuracy(adv_pred_conf.detach().cpu(), adversarial_labels)
        adv_confidences.append(
            adv_pred_conf.cpu().detach().numpy(),
        )
        adv_labels.append(adversarial_labels.numpy())

        orig_pred_softmax = model(original_images.to(device))
        orig_pred_conf = torch.exp(orig_pred_softmax)
        orig_accuracy(orig_pred_conf.detach().cpu(), original_labels)
        orig_confidences.append(
            orig_pred_conf.cpu().detach().numpy(),
        )
        orig_labels.append(original_labels.numpy())

    orig_acc = orig_accuracy.compute()
    adv_acc = adv_accuracy.compute()
    print(f"Data set: {data_set_name}")
    print(f"Orig accuracy: {orig_acc}")
    print(f"Adv accuracy: {adv_acc}")
    adv_confidences = np.concatenate(adv_confidences, axis=0)
    adv_lbl = np.concatenate(adv_labels, axis=0)
    orig_confidences = np.concatenate(orig_confidences, axis=0)
    orig_lbl = np.concatenate(orig_labels, axis=0)

    torch.save(
        (torch.from_numpy(adv_confidences), torch.from_numpy(adv_lbl)),
        os.path.join(data_path, "confidences_adv.pt"),
    )
    torch.save(
        (torch.from_numpy(orig_confidences), torch.from_numpy(orig_lbl)),
        os.path.join(data_path, "confidences_orig.pt"),
    )


def visualize_explanation_similarities(
    model: LightningModule,
    data_loader: DataLoader,
    data_set_name: str,
    device: torch.device,
    stage: str,
):
    data_set_map = data_set_mappings[data_set_name]
    use_original_explanations = (
        "Explanations" in model.hparams["hparams"]["data_set"]["name"]
    )

    orig_labels = np.array([])
    orig_pred_labels = np.array([])
    adv_labels = np.array([])
    adv_pred_labels = np.array([])
    sim_mse = np.array([])
    sim_pcc = np.array([])
    sim_ssim = np.array([])
    for batch in tqdm(data_loader):
        if use_original_explanations:
            (
                original_images,
                original_explanations_pre,
                adversarial_images,
                adversarial_explanations_pre,
                original_labels,
                adversarial_labels,
                batch_indices,
            ) = batch
        else:
            (
                original_images,
                adversarial_images,
                original_labels,
                adversarial_labels,
                batch_indices,
            ) = batch

        original_images = original_images.to(device)
        adversarial_images = adversarial_images.to(device)

        # get explanation maps
        original_explanation_maps = model.explainer.explain(
            original_images, original_labels.to(device)
        )
        adversarial_explanation_maps = model.explainer.explain(
            adversarial_images, adversarial_labels.to(device)
        )
        # Predict orig and adv label
        original_pred_labels = torch.argmax(model(original_images), dim=-1)
        adversarial_pred_labels = torch.argmax(model(adversarial_images), dim=-1)

        # calculate similarities
        _, similarities_mse = get_similarities(
            "MSE", original_explanation_maps, adversarial_explanation_maps
        )
        _, similarities_pcc = get_similarities(
            "PCC", original_explanation_maps, adversarial_explanation_maps
        )
        _, similarities_ssim = get_similarities(
            "SSIM", original_explanation_maps, adversarial_explanation_maps
        )

        # concat arrays
        orig_labels = np.append(orig_labels, original_labels.cpu().detach().numpy())
        orig_pred_labels = np.append(
            orig_pred_labels, original_pred_labels.cpu().detach().numpy()
        )
        adv_labels = np.append(adv_labels, adversarial_labels.cpu().detach().numpy())
        adv_pred_labels = np.append(
            adv_pred_labels, adversarial_pred_labels.cpu().detach().numpy()
        )
        sim_mse = np.append(sim_mse, similarities_mse.cpu().detach().numpy())
        sim_pcc = np.append(sim_pcc, similarities_pcc.cpu().detach().numpy())
        sim_ssim = np.append(sim_ssim, similarities_ssim.cpu().detach().numpy())

    df_sim = pd.DataFrame(
        [
            sim_mse,
            sim_pcc,
            sim_ssim,
            orig_labels,
            orig_pred_labels,
            adv_labels,
            adv_pred_labels,
        ],
        index=[
            "mse_sim",
            "pcc_sim",
            "ssim_sim",
            "orig_label",
            "orig_pred_label",
            "adv_label",
            "adv_pred_label",
        ],
    ).T
    df_sim["orig_label"] = df_sim["orig_label"].astype(int)
    df_sim["orig_pred_label"] = df_sim["orig_pred_label"].astype(int)
    df_sim["adv_label"] = df_sim["adv_label"].astype(int)
    df_sim["adv_pred_label"] = df_sim["adv_pred_label"].astype(int)
    df_sim["orig_label_name"] = df_sim["orig_label"].map(data_set_map)
    sorted_df_sim = df_sim.sort_values(by="orig_label")

    # Save similarities DataFrame as csv
    csv_path = os.path.join(
        model.hparams["hparams"]["log_path"], f"{stage}_similarities.csv"
    )
    sorted_df_sim.to_csv(csv_path)

    manipulated_classes = [
        data_set_map[c] for c in model.hparams["hparams"]["included_classes"]
    ]

    # Plot similarity histograms for both MSE and PCC
    hist_mse, hist_pcc, kde_mse, kde_pcc = plot_similarities(
        sorted_df_sim, data_set_name, data_set_map, manipulated_classes, stage
    )
    # Save plots
    hist_mse.savefig(
        os.path.join(
            model.image_log_path, f"{stage}_explanation_similarity_hist_mse.png"
        ),
        transparent=True,
    )
    hist_pcc.savefig(
        os.path.join(
            model.image_log_path, f"{stage}_explanation_similarity_hist_pcc.png"
        ),
        transparent=True,
    )
    kde_mse.savefig(
        os.path.join(
            model.image_log_path, f"{stage}_explanation_similarity_kde_mse.png"
        ),
        transparent=True,
    )
    kde_pcc.savefig(
        os.path.join(
            model.image_log_path, f"{stage}_explanation_similarity_kde_pcc.png"
        ),
        transparent=True,
    )


def plot_similarities(
    sorted_df_sim, data_set_name, data_set_map, manipulated_classes, stage
):
    hist_mse = plot_similarities_histogram_with_boxplot(
        sorted_df_sim,
        "orig_label_name",
        "mse_sim",
        f"{data_set_name} {stage} original vs. adversarial "
        f"explanation similarities (MSE) "
        f"after manipulating on classes '{manipulated_classes}'",
        log_x=True,
        palette=MSE_COLOR_PALETTE,
    )
    hist_pcc = plot_similarities_histogram_with_boxplot(
        sorted_df_sim,
        "orig_label_name",
        "pcc_sim",
        f"{data_set_name} {stage} original vs. adversarial "
        f"explanation similarities (PCC) histogram "
        f"after manipulating on classes '{manipulated_classes}'",
        log_x=False,
        palette=PCC_COLOR_PALETTE,
    )
    kde_mse = plot_similarities_kde(
        sorted_df_sim,
        "mse_sim",
        list(data_set_map.values()),
        f"{data_set_name} {stage} original vs. adversarial "
        f"explanation similarities (MSE) histogram"
        f" KDE plots after manipulating on classes '{manipulated_classes}'",
        log_x=True,
        palette=MSE_COLOR_PALETTE,
    )
    kde_pcc = plot_similarities_kde(
        sorted_df_sim,
        "pcc_sim",
        list(data_set_map.values()),
        f"{data_set_name} {stage} original vs. "
        f"adversarial explanation similarities (PCC)"
        f" KDE plots after manipulating on classes '{manipulated_classes}'",
        log_x=False,
        palette=PCC_COLOR_PALETTE,
    )
    return hist_mse, hist_pcc, kde_mse, kde_pcc


@hydra.main(config_name="manipulated_model_training_config")
def run_visualize_explanation_similarities(
    config: ManipulatedModelTrainingConfig,
):
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.gpus != 0) else "cpu"
    )

    model = get_manipulatable_model(config).load_from_checkpoint(config.checkpoint)
    model.to(device)
    model.eval()
    model.freeze()

    print(
        f"Visualizing explanation similarities for"
        f" model checkpoint '{config.checkpoint}' and "
        f"XAI technique '{model.hparams['hparams']['explainer']['name']}'."
    )

    data_set_name = config.data_set.name
    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_path,
        download=False,
        batch_size=64,
        val_split=0.0,
        transform=None,
    )

    train_loader = data_module.train_dataloader(shuffle=False)
    visualize_explanation_similarities(
        model,
        train_loader,
        data_set_name,
        device,
        stage="train",
    )


def run():
    data_set_mapping = [
        {
            "name": AdversarialDataSetNames.ADVERSARIAL_MNIST,
            "path": "/home/steffi/dev/master_thesis/"
            "hiding_adversarial_attacks/data/preprocessed/adversarial/"
            "data-set=MNIST--attack=DeepFool--eps=0.2--cp-run=HAA-946",
            "checkpoint": "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
            "models/MNIST-HAA-946/checkpoints/model-epoch=11-val_loss=0.04.ckpt",
        },
        {
            "name": AdversarialDataSetNames.ADVERSARIAL_FASHION_MNIST,
            "path": "/home/steffi/dev/master_thesis/"
            "hiding_adversarial_attacks/data/preprocessed/adversarial/"
            "data-set=FashionMNIST--attack=DeepFool--eps=0.105--cp-run=HAA-952",
            "checkpoint": "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/"
            "models/FashionMNIST-HAA-952/checkpoints/model-epoch=18-val_loss=0.20.ckpt",
        },
        {
            "name": AdversarialDataSetNames.ADVERSARIAL_CIFAR10,
            "path": "/home/steffi/dev/master_thesis/"
            "hiding_adversarial_attacks/data/preprocessed/adversarial/"
            "data-set=CIFAR10--attack=DeepFool--eps=0.1--cp-run=HAA-943",
            "checkpoint": "/home/steffi/dev/master_thesis/hiding_adversarial_attacks"
            "/models/CIFAR10-mobilenetv2-HAA-943/checkpoints/"
            "model-epoch=00-val_loss=0.06.ckpt",
        },
    ]
    for data_set in data_set_mapping:
        save_classification_confidences(
            data_set["path"],
            data_set["name"],
            data_set["checkpoint"],
            torch.device("cuda"),
        )


if __name__ == "__main__":
    run_visualize_explanation_similarities()
