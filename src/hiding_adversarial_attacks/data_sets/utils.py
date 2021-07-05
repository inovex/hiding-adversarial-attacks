from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames


def get_transform(data_set_name: str, data_is_tensor: bool = False):
    transform = []
    if not data_is_tensor:
        transform = transform.append(transforms.ToTensor())
    if DataSetNames.CIFAR10 in data_set_name:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(transform)
    if len(transform) == 0:
        transform = None
    return transform
