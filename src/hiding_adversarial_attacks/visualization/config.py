from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames

PCC_COLOR_PALETTE = "PuRd"
MSE_COLOR_PALETTE = "afmhot_r"
CONFUSION_MATRIX_COLOR_PALETTE = "PuRd"
DATA_SET_MAPPING = {
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
