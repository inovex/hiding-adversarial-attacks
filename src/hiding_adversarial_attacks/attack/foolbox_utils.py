import eagerpy as ep
import foolbox as fb
from foolbox.attacks.gradient_descent_base import BaseGradientDescent
from torch import Tensor

from hiding_adversarial_attacks.config.attack.adversarial_attack_config import (
    AdversarialAttackNames,
)


def get_attack(attack_name: str) -> BaseGradientDescent:
    if attack_name == AdversarialAttackNames.FGSM:
        attack = fb.attacks.LinfFastGradientAttack()
    elif attack_name == AdversarialAttackNames.DEEP_FOOL:
        attack = fb.attacks.LinfDeepFoolAttack()
    else:
        raise SystemExit("Unknown adversarial attack was specified. Exiting.")
    return attack


def filter_correctly_classified(
    foolbox_model: fb.PyTorchModel, images: Tensor, labels: ep.Tensor
):
    correctly_classified_mask = get_correctly_classified_mask(
        foolbox_model, images, labels
    )
    return images[correctly_classified_mask], labels[correctly_classified_mask]


def get_correctly_classified_mask(
    foolbox_model: fb.PyTorchModel, images: ep.Tensor, labels: ep.Tensor
):
    """
    Checks whether the model predicts the "correct", ground-truth labels
    for the (non-adversarial) input images.

    :param foolbox_model: Foolbox-wrapped PyTorch model
    :param images: tensor of non-adversarial images (B, C, W, H)
    :param labels: tensor of ground truth labels for the images (B, N),
                   where N = number of classes
    :return: boolean tensor containing True for images that were
             correctly classified by the model, and False otherwise
    """
    images_, restore_type_images = ep.astensor_(images)
    labels_, restore_type_labels = ep.astensor_(labels)

    predictions = foolbox_model(images_).argmax(axis=-1)
    return restore_type_images(predictions == labels_)
