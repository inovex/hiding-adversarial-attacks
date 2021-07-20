import numpy as np
import torch
from matplotlib import pyplot as plt


def display_tensor_as_image(
    tensor: torch.Tensor, title: str = None, cmap: str = "gray"
):
    np_img = tensor_to_pil_numpy(tensor)
    plt.imshow(np_img, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.show()


def tensor_to_pil_numpy(rgb_tensor):
    if len(rgb_tensor.shape) == 3:
        return np.transpose(rgb_tensor.cpu().detach().numpy(), (1, 2, 0))
    else:
        return np.transpose(rgb_tensor.cpu().detach().numpy(), (0, 2, 3, 1))
