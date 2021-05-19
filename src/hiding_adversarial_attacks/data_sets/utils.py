from torchvision.transforms import transforms

from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames


def get_transform(data_set_name: str):
    if data_set_name == DataSetNames.CIFAR10:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    else:
        transform = transforms.ToTensor()
    return transform
