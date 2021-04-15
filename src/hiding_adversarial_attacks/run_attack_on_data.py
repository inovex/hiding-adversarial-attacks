import logging
import os
from datetime import datetime
from typing import List, Sequence, Union

import eagerpy as ep
import foolbox as fb
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from hiding_adversarial_attacks.attack.attack_results import (
    AttackResults,
    BatchAttackResults,
)
from hiding_adversarial_attacks.attack.foolbox_utils import (
    get_attack,
    get_correctly_classified_mask,
)
from hiding_adversarial_attacks.attack.logging_utils import (
    log_attack_info,
    log_attack_results,
    setup_logger,
)
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.adversarial_attack_config import (
    AdversarialAttackConfig,
)
from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    ALL_CLASSES,
)
from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames
from hiding_adversarial_attacks.data_modules.utils import get_data_module

LOGGER = logging.Logger(os.path.basename(__file__))


def attack_batch(
    foolbox_model: fb.PyTorchModel,
    images: ep.Tensor,
    labels: ep.Tensor,
    attack: fb.Attack,
    epsilons: Union[Sequence[Union[float, None]], float, None],
) -> List[BatchAttackResults]:

    # Get a mask of the images correctly classified by the model before being attacked
    correctly_classified_mask = get_correctly_classified_mask(
        foolbox_model, images, labels
    )
    # images that are not correctly classified by the model as-is
    misclassified_images, misclassified_labels = (
        images[~correctly_classified_mask],
        labels[~correctly_classified_mask],
    )
    images, labels = (
        images[correctly_classified_mask],
        labels[correctly_classified_mask],
    )

    if images.nelement() == 0:
        raise SystemExit(
            "SYSTEM_EXIT: No images left after filtering. "
            "Are you sure you trained your model correctly "
            "to classify the input images?"
        )

    raw, clipped, is_adv = attack(foolbox_model, images, labels, epsilons=epsilons)
    robust_accuracy = 1 - ep.astensor(is_adv.type(torch.FloatTensor)).mean(axis=-1)

    attack_results_list = []
    for i, (eps, acc, clipped_adv, adv_mask) in enumerate(
        zip(epsilons, robust_accuracy, clipped, is_adv)
    ):
        adv_count = len(adv_mask.nonzero())
        attacked_count = len(images)
        adv_images = clipped_adv[adv_mask]
        if adv_count > 0:
            adv_logits = foolbox_model(adv_images)
            adv_labels = torch.argmax(adv_logits, dim=-1)
        else:
            adv_labels = torch.empty_like(labels[adv_mask])

        batch_attack_results = BatchAttackResults(
            images[adv_mask],
            labels[adv_mask],
            adv_images,
            adv_labels,
            misclassified_images,
            misclassified_labels,
            attacked_count,
            adv_count,
            len(misclassified_images),
            eps,
        )
        attack_results_list.append(batch_attack_results)
    return attack_results_list


def run_attack(
    foolbox_model: fb.PyTorchModel,
    attack: fb.Attack,
    data_loader: DataLoader,
    epsilons: Sequence[float],
    stage: str,
    device: torch.device,
) -> List[AttackResults]:
    print(f"Attacking images for stage '{stage}'.")

    attack_results_list = [AttackResults(stage, eps, device) for eps in epsilons]
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_attack_results_list = attack_batch(
            foolbox_model, images, labels, attack, epsilons
        )
        for batch_attack_results, attack_result in zip(
            batch_attack_results_list, attack_results_list
        ):
            attack_result.add_batch(batch_attack_results)
    return attack_results_list


@hydra.main(config_name="adversarial_attack_config")
def run(config: AdversarialAttackConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # Logging
    os.makedirs(config.log_path, exist_ok=True)
    log_file_name = config.log_file_name.format(
        timestamp=int(datetime.now().timestamp()),
        data_set=config.data_set.name,
        attack=config.attack.name,
        epsilons=config.attack.epsilons,
    )
    log_file_path = os.path.join(config.log_path, log_file_name)
    setup_logger(LOGGER, log_file_path, log_level=config.logging.log_level)

    # GPU or CPU
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.gpus != 0) else "cpu"
    )

    # Setup data module
    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_set.external_path,
        download=False,
        batch_size=config.batch_size,
        val_split=0.0,
        transform=transforms.ToTensor(),
        random_seed=config.random_seed,
    )

    # Data loaders
    train_loader = data_module.train_dataloader(shuffle=False)
    test_loader = data_module.test_dataloader()

    # Load foolbox wrapped model
    if (
        config.data_set.name == DataSetNames.MNIST
        or config.data_set.name == DataSetNames.FASHION_MNIST
    ):
        foolbox_model = MNISTNet.as_foolbox_wrap(config, device)

    log_attack_info(
        LOGGER,
        config.attack.name,
        config.attack.epsilons,
        config.data_set.name,
        config.checkpoint,
        ALL_CLASSES,
    )

    # Run adversarial attack
    attack = get_attack(config.attack.name)

    train_attack_results_list = run_attack(
        foolbox_model, attack, train_loader, config.attack.epsilons, "training", device
    )
    test_attack_results_list = run_attack(
        foolbox_model, attack, test_loader, config.attack.epsilons, "test", device
    )

    # Log and save results
    LOGGER.info("")
    LOGGER.info("******** Results *********")
    for train_attack_results, test_attack_results in zip(
        train_attack_results_list, test_attack_results_list
    ):
        output_dirname = config.output_dirname.format(
            data_set=config.data_set.name,
            attack=config.attack.name,
            epsilon=test_attack_results.epsilon,
        )
        target_path = os.path.join(config.data_set.adversarial_path, output_dirname)
        train_attack_results.save_results(target_path)
        test_attack_results.save_results(target_path)

        # Log results
        LOGGER.info(f"---- Epsilon: {test_attack_results.epsilon}")
        LOGGER.info(f"Output path: '{target_path}'")
        LOGGER.info("\t Train: ")
        log_attack_results(
            LOGGER,
            train_attack_results,
            len(train_loader.dataset),
        )
        LOGGER.info("\t Test: ")
        log_attack_results(LOGGER, test_attack_results, len(test_loader.dataset))


if __name__ == "__main__":
    run()
