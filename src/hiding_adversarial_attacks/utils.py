import argparse
from functools import wraps
from time import time

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage

toPilImage = ToPILImage()


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f"function {func.__name__} took {1000*(te-ts):.1f} ms")
        return result

    return wrap


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


def display_tensor_as_image(tensor: torch.Tensor):
    plt.imshow(tensor.numpy()[0], cmap="gray")
    plt.show()


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def to_pil_image(tensor, mode="L"):
    return ToPILImage(mode=mode)(tensor)
