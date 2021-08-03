import os

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

from hiding_adversarial_attacks.visualization.config import (
    CONFUSION_MATRIX_COLOR_PALETTE,
    DATA_SET_MAPPING,
)


def save_confusion_matrix(matrix: np.array, data_set_name: str, log_path: str):
    _matrix = matrix.astype("int")
    data_set_classes = DATA_SET_MAPPING[data_set_name].values()
    df = pd.DataFrame(_matrix, index=data_set_classes, columns=data_set_classes)
    df.to_csv(os.path.join(log_path, "confusion_matrix.csv"))
    fig = plt.figure(figsize=(12, 10))
    ax = sn.heatmap(df, annot=True, fmt="d", cmap=CONFUSION_MATRIX_COLOR_PALETTE)
    ax.xaxis.label.set_size(14)
    ax.tick_params(axis="y", labelrotation=0, labelsize=14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis="x", labelrotation=-38, labelsize=14)
    fig.suptitle(f"{data_set_name} post-manipulation confusion matrix", fontsize=18)
    fig.tight_layout()
    fig.savefig(
        os.path.join(log_path, "image_log/confusion_matrix.png"),
        transparent=True,
    )
    fig.show()
