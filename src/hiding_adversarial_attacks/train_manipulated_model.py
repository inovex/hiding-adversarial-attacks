import logging
import os

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from torch._vmap_internals import vmap

from hiding_adversarial_attacks._neptune.utils import get_neptune_logger
from hiding_adversarial_attacks.callbacks.neptune_callback import NeptuneLoggingCallback
from hiding_adversarial_attacks.classifiers.mnist_net import MNISTNet
from hiding_adversarial_attacks.config.config_validator import ConfigValidator
from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
)
from hiding_adversarial_attacks.config.losses.similarity_loss_config import (
    SimilarityLossMapping,
)
from hiding_adversarial_attacks.config.manipulated_model_training_config import (
    ManipulatedModelTrainingConfig,
)
from hiding_adversarial_attacks.data_modules.utils import (
    VisionDataModuleUnionType,
    get_data_module,
)
from hiding_adversarial_attacks.manipulated_classifiers.manipulated_mnist_net import (
    ManipulatedMNISTNet,
)
from hiding_adversarial_attacks.manipulated_classifiers.metricized_explanations import (
    MetricizedTopAndBottomExplanations,
)

logger = logging.getLogger(__file__)


def get_manipulatable_model(config):
    if config.data_set.name in [
        AdversarialDataSetNames.ADVERSARIAL_MNIST,
    ]:
        classifier_model = MNISTNet(config).load_from_checkpoint(
            config.classifier_checkpoint
        )
        model = ManipulatedMNISTNet(classifier_model, config)
        return model
    else:
        raise SystemExit(
            f"Unknown data set specified: {config.data_set.name}. Exiting."
        )


def load_explanations(config, device: torch.device):
    training_adv_expl = torch.load(
        os.path.join(config.explanations_path, "training_adv_exp.pt"),
        map_location=device,
    )
    training_orig_expl = torch.load(
        os.path.join(config.explanations_path, "training_orig_exp.pt"),
        map_location=device,
    )
    test_adv_expl = torch.load(
        os.path.join(config.explanations_path, "test_adv_exp.pt"),
        map_location=device,
    )
    test_orig_expl = torch.load(
        os.path.join(config.explanations_path, "test_orig_exp.pt"),
        map_location=device,
    )
    return training_orig_expl, training_adv_expl, test_orig_expl, test_adv_expl


def load_attacked_data(config, device: torch.device):
    training_adv = torch.load(
        os.path.join(config.data_path, "training_adv.pt"),
        map_location=device,
    )
    training_orig = torch.load(
        os.path.join(config.data_path, "training_orig.pt"),
        map_location=device,
    )
    test_adv = torch.load(
        os.path.join(config.data_path, "test_adv.pt"),
        map_location=device,
    )
    test_orig = torch.load(
        os.path.join(config.data_path, "test_orig.pt"),
        map_location=device,
    )
    return training_orig, training_adv, test_orig, test_adv


def get_metricized_top_and_bottom_explanations(
    config: ManipulatedModelTrainingConfig, device: torch.device
) -> MetricizedTopAndBottomExplanations:
    (
        training_orig_expl,
        training_adv_expl,
        test_orig_expl,
        test_adv_expl,
    ) = load_explanations(config, device)

    training_orig, training_adv, test_orig, test_adv = load_attacked_data(
        config, device
    )
    similarity_loss = SimilarityLossMapping[config.similarity_loss.name]
    batched_sim_loss = vmap(similarity_loss)
    (
        top_orig_expl,
        top_adv_expl,
        top_similarities,
        top_indices,
        bottom_orig_expl,
        bottom_adv_expl,
        bottom_similarities,
        bottom_indices,
    ) = get_top_and_bottom_k_explanations(
        training_adv_expl[0],
        training_orig_expl[0],
        batched_sim_loss,
    )

    metricized_top_and_bottom_explanations = MetricizedTopAndBottomExplanations(
        device=device,
        sorted_by=config.similarity_loss.name,
        top_k_indices=top_indices,
        bottom_k_indices=bottom_indices,
        top_k_original_images=training_orig[0][top_indices],
        top_k_original_explanations=top_orig_expl,
        top_k_original_labels=training_orig[1][top_indices].long(),
        top_k_adversarial_images=training_adv[0][top_indices],
        top_k_adversarial_explanations=top_adv_expl,
        top_k_adversarial_labels=training_adv[1][top_indices].long(),
        bottom_k_original_images=training_orig[0][bottom_indices],
        bottom_k_original_explanations=bottom_orig_expl,
        bottom_k_original_labels=training_orig[1][bottom_indices].long(),
        bottom_k_adversarial_images=training_adv[0][bottom_indices],
        bottom_k_adversarial_explanations=bottom_adv_expl,
        bottom_k_adversarial_labels=training_adv[1][bottom_indices].long(),
    )
    return metricized_top_and_bottom_explanations


def get_top_and_bottom_k_explanations(
    training_adv_expl,
    training_orig_expl,
    batched_sim_loss,
):
    similarity_results = batched_sim_loss(training_orig_expl, training_adv_expl)
    # largest mse
    bottom_similarities, bottom_indices = torch.topk(similarity_results, 4)
    bottom_similarities, bottom_indices = (
        torch.flip(bottom_similarities, dims=(0,)),
        torch.flip(bottom_indices, dims=(0,)),
    )
    # smallest mse
    top_similarities, top_indices = torch.topk(similarity_results, 4, largest=False)
    return (
        training_orig_expl[top_indices],
        training_adv_expl[top_indices],
        top_similarities,
        top_indices,
        training_orig_expl[bottom_indices],
        training_adv_expl[bottom_indices],
        bottom_similarities,
        bottom_indices,
    )


def train(
    data_module: VisionDataModuleUnionType,
    neptune_logger: NeptuneLogger,
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
    metricized_top_and_bottom_explanations: MetricizedTopAndBottomExplanations,
):
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    checkpoint_callback = hydra.utils.instantiate(config.checkpoint_config)

    model = get_manipulatable_model(config)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)
    model.to(device)

    neptune_callback = NeptuneLoggingCallback(
        log_path=config.log_path,
        image_log_path=model.image_log_path,
        trash_run=config.trash_run,
    )
    trainer = Trainer(
        gpus=config.gpus,
        logger=neptune_logger,
        callbacks=[checkpoint_callback, neptune_callback],
    )

    trainer.fit(model, train_loader, validation_loader)


def test(
    data_module,
    neptune_logger: NeptuneLogger,
    device: torch.device,
    config: ManipulatedModelTrainingConfig,
    metricized_top_and_bottom_explanations: MetricizedTopAndBottomExplanations,
):
    test_loader = data_module.test_dataloader()

    trainer = Trainer(gpus=config.gpus, logger=neptune_logger)

    model = get_manipulatable_model(config).load_from_checkpoint(config.checkpoint)
    model.set_metricized_explanations(metricized_top_and_bottom_explanations)

    trainer.test(model, test_loader, ckpt_path="best")


@hydra.main(config_name="manipulated_model_training_config")
def run(config: ManipulatedModelTrainingConfig) -> None:
    config_validator = ConfigValidator()
    config_validator.validate(config)

    logger.info("Starting train_manipulated_model.py")
    logger.info(f"cwd: {os.getcwd()}")
    logger.info("**** Parameters: ******")
    logger.info(OmegaConf.to_yaml(config))

    data_module = get_data_module(
        data_set=config.data_set.name,
        data_path=config.data_path,
        download=False,
        batch_size=config.batch_size,
        val_split=config.val_split,
        transform=None,
        random_seed=config.random_seed,
    )

    # GPU or CPU
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.gpus != 0) else "cpu"
    )

    metricized_top_and_bottom_explanations = get_metricized_top_and_bottom_explanations(
        config, device
    )

    experiment_name = config.data_set.name
    config.tags.append(config.data_set.name)
    config.tags.append(config.explainer.name)
    config.tags.append("test" if config.test else "train")
    if config.trash_run:
        config.tags.append("trash")
    neptune_logger = get_neptune_logger(config, experiment_name, list(config.tags))

    # Override log path
    config.log_path = os.path.join(
        config.log_path, neptune_logger.name, neptune_logger.version
    )
    os.makedirs(config.log_path, exist_ok=True)

    if config.test:
        test(
            data_module,
            neptune_logger,
            device,
            config,
            metricized_top_and_bottom_explanations,
        )
    else:
        train(
            data_module,
            neptune_logger,
            device,
            config,
            metricized_top_and_bottom_explanations,
        )


if __name__ == "__main__":
    run()
