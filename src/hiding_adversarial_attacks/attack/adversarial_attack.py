import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Sequence, Union

import eagerpy as ep
import foolbox as fb
import numpy as np
import torch
from pytorch_lightning import Trainer
from tqdm import tqdm

from hiding_adversarial_attacks.attack.attack_results import (
    AttackResults,
    BatchAttackResults,
)
from hiding_adversarial_attacks.attack.foolbox_utils import (
    filter_correctly_classified,
    get_attack,
)
from hiding_adversarial_attacks.config import (
    AdversarialAttackConfig,
    DataConfig,
    MNISTConfig,
)
from hiding_adversarial_attacks.mnist.data_module import init_mnist_data_module
from hiding_adversarial_attacks.mnist.mnist_net import MNISTNet

LOGGER = logging.Logger(__name__)


def setup_logger(logger, args):
    logs_path = os.path.join(AdversarialAttackConfig.LOGS_PATH, "MNIST")
    os.makedirs(logs_path, exist_ok=True)
    timestamp_seconds = int(datetime.now().timestamp())
    log_file = f"{str(timestamp_seconds)}-mnist-{args.attack}-{args.epsilons}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    fh = logging.FileHandler(os.path.join(logs_path, log_file))
    fh.setLevel(AdversarialAttackConfig.LOG_LEVEL)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(AdversarialAttackConfig.LOG_LEVEL)
    logger.addHandler(ch)

    logger.setLevel(AdversarialAttackConfig.LOG_LEVEL)


def parse_attack_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        required=True,
        help="<Required> model checkpoint used for running the adversarial attack",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--target-dir",
        default=os.path.join(DataConfig.ADVERSARIAL_PATH, "MNIST"),
        help="path to store adversarially attacked MNIST data to",
    )
    parser.add_argument(
        "--attack",
        default="FGSM",
        const="FGSM",
        nargs="?",
        choices=["FGSM", "DeepFool"],
        help="adversarial attack method (default: %(default)s)",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        help="epsilon values (= attack strength) to use on images",
        default=np.linspace(0.225, 0.3, num=1),
    )
    parser = MNISTNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint was 'None'.")
    return args


def attack_batch(
    foolbox_model: fb.PyTorchModel,
    images: ep.Tensor,
    labels: ep.Tensor,
    attack: fb.Attack,
    epsilons: Union[Sequence[Union[float, None]], float, None],
):
    images, labels = filter_correctly_classified(foolbox_model, images, labels)

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
        batch_attack_results = BatchAttackResults(
            images, labels, clipped_adv[adv_mask], None, len(images), adv_count, eps
        )
        attack_results_list.append(batch_attack_results)
    return attack_results_list


def run_attack(foolbox_model, attack, data_loader, epsilons, stage, device):
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


def run():
    args = parse_attack_args()

    # Logging
    setup_logger(LOGGER, args)

    # GPU or CPU
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )

    # Setup MNIST data module and loaders
    data_module = init_mnist_data_module(
        batch_size=args.batch_size, val_split=0.0, download_mnist=False, seed=args.seed
    )
    train_loader = data_module.train_dataloader(shuffle=False)
    test_loader = data_module.test_dataloader()

    # Load model
    mnist_model = MNISTNet(args).load_from_checkpoint(args.checkpoint)
    mnist_model.eval()

    foolbox_model = fb.PyTorchModel(
        mnist_model,
        bounds=MNISTConfig.BOUNDS,
        preprocessing=MNISTConfig.PREPROCESSING,
        device=device,
    )

    LOGGER.info("Attack info:")
    LOGGER.info(f"\t - Attack type: {args.attack}")
    LOGGER.info(f"\t - Epsilon(s): {args.epsilons}")
    LOGGER.info("\t - Data set: MNIST")
    LOGGER.info(f"\t - Target dir: {args.target_dir}")
    LOGGER.info(f"\t - Model checkpoint: {args.checkpoint}")

    attack = get_attack(args.attack)
    if attack is None:
        raise SystemExit("Unknown adversarial attack was specified. Exiting.")

    train_attack_results_list = run_attack(
        foolbox_model, attack, train_loader, args.epsilons, "training", device
    )
    test_attack_results_list = run_attack(
        foolbox_model, attack, test_loader, args.epsilons, "test", device
    )
    for train_attack_results, test_attack_results in zip(
        train_attack_results_list, test_attack_results_list
    ):
        target_dir = os.path.join(
            args.target_dir, args.attack, f"epsilon_{train_attack_results.epsilon}"
        )
        train_attack_results.save_results(target_dir)
        test_attack_results.save_results(target_dir)


if __name__ == "__main__":
    run()
    # data, labels = torch.load(
    #     "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data/preprocessed/adversarial/MNIST/DeepFool/epsilon_0.225/test_adv.pt"
    # )
    # print("")
