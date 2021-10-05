from hiding_adversarial_attacks.config.data_sets.data_set_config import (
    AdversarialDataSetNames,
    DataSetNames,
)
from hiding_adversarial_attacks.config.explainers.explainer_config import ExplainerNames

PCC_COLOR_PALETTE = "PuRd"
MSE_COLOR_PALETTE = "afmhot_r"
SSIM_COLOR_PALETTE = "BuGn"
CONFUSION_MATRIX_COLOR_PALETTE = "PuRd"

DATA_SET_PLOT_NAMES = {
    DataSetNames.FASHION_MNIST: "Fashion-MNIST",
    DataSetNames.CIFAR10: "CIFAR-10",
}

EXPLAINER_PLOT_NAMES = {
    ExplainerNames.GRAD_CAM: "Grad-CAM",
    ExplainerNames.GUIDED_BACKPROP: "Guided Backpropagation",
    ExplainerNames.INTEGRATED_GRADIENTS: "Integrated Gradients",
    ExplainerNames.INPUT_X_GRADIENT: "Gradient * Input",
}

DATA_SET_MAPPING = {
    DataSetNames.FASHION_MNIST: {
        0: "T-shirt/top",
        1: "Trousers",
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
