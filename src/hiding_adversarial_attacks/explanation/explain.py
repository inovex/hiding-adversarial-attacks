from argparse import ArgumentParser, Namespace

import torch
from captum.attr import visualization as viz

from hiding_adversarial_attacks.config import DeepLiftBaselineConfig
from hiding_adversarial_attacks.explanation.explainers import DeepLiftExplainer
from hiding_adversarial_attacks.mnist.mnist_net import MNISTNet
from hiding_adversarial_attacks.utils import tensor_to_pil_numpy


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for creating explanations (default: 64)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        required=True,
        help="<Required> model checkpoint used for creating explanations.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    adversarials = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data"
        "/preprocessed/adversarial/MNIST/DeepFool/epsilon_0.225/"
        "class_1/test_adv.pt"
    )
    originals = (
        "/home/steffi/dev/master_thesis/hiding_adversarial_attacks/data"
        "/preprocessed/adversarial/MNIST/DeepFool/epsilon_0.225/"
        "class_1/test_orig.pt"
    )
    baseline_name = DeepLiftBaselineConfig.ZERO

    args = parse_args()

    # GPU or CPU
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )

    # Load model
    lit_model = MNISTNet.load_from_checkpoint(checkpoint_path=args.checkpoint)
    model = lit_model.to(device)

    # Explainer
    deeplift_explainer = DeepLiftExplainer(
        model, baseline_name=baseline_name, device=device
    )

    # Load images and labels
    orig_img, orig_labels = torch.load(originals)
    adv_img, adv_labels = torch.load(adversarials)

    orig_img_batch = orig_img[0:8].cuda()
    orig_labels_batch = orig_labels[0:8].long().cuda()

    orig_explanations = deeplift_explainer.explain(orig_img_batch, orig_labels_batch)
    expl = tensor_to_pil_numpy(orig_explanations)
    img = tensor_to_pil_numpy(orig_img_batch)
    viz.visualize_image_attr(
        expl[0],
        img[0],
        method="blended_heat_map",
        sign="all",
        show_colorbar=True,
        title=baseline_name,
    )
