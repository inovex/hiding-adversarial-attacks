import argparse
import os

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from hiding_adversarial_attacks.config import DataConfig, LoggingConfig, MNISTConfig
from hiding_adversarial_attacks.mnist.data_modules import (
    init_fashion_mnist_data_module,
    init_mnist_data_module,
)
from hiding_adversarial_attacks.mnist.mnist_net import MNISTNet


def parse_mnist_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="validation split (default: 0.1)"
    )
    parser.add_argument(
        "--data-set",
        type=str,
        default=DataConfig.MNIST,
        const=DataConfig.MNIST,
        nargs="?",
        choices=[DataConfig.MNIST, DataConfig.FASHION_MNIST],
        help="data set to use (default: 'MNIST')",
    )
    parser.add_argument(
        "--logs-dir",
        default=LoggingConfig.LOGS_PATH,
        help="base path to store classifier training logs and checkpoints to."
        " The final path is '<logs-dir>/<data-set>'.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        default=False,
        help="whether the data set should be downloaded",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="runs model testing stage",
    )
    parser.add_argument(
        "--test-checkpoint",
        default=None,
        help="model checkpoint used for testing stage",
    )
    parser = MNISTNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if not args.test and args.test_checkpoint is not None:
        parser.error("--test-checkpoint can only be set when --test flag is used.")
    if args.test and args.test_checkpoint is None:
        parser.error("--test-checkpoint was empty while --test was specified.")
    if args.logs_dir is not None:
        if not os.path.isdir(args.logs_dir):
            parser.error("--logs-dir needs to be a valid directory path.")
        args.logs_dir = os.path.join(args.logs_dir, args.data_set)
    return args


def get_model(args: argparse.Namespace):
    if args.data_set == DataConfig.MNIST or args.data_set == DataConfig.FASHION_MNIST:
        return MNISTNet(args)
    else:
        raise SystemExit(f"Unknown data set specified: {args.data_set}. Exiting.")


def train(data_module, args):
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    checkpoint_callback = ModelCheckpoint(
        monitor=MNISTConfig.VAL_LOSS,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    tb_logger = pl_loggers.TensorBoardLogger(args.logs_dir)
    trainer = Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback]
    )

    model = get_model(args)

    trainer.fit(model, train_loader, validation_loader)


def test(data_module, args):
    test_loader = data_module.test_dataloader()

    tb_logger = pl_loggers.TensorBoardLogger(args.logs_dir)
    trainer = Trainer.from_argparse_args(args, logger=tb_logger)

    model = get_model(args).load_from_checkpoint(args.test_checkpoint)

    trainer.test(model, test_loader, ckpt_path="best")


def run():
    args = parse_mnist_args()

    if args.data_set == DataConfig.MNIST:
        data_module = init_mnist_data_module(
            args.batch_size, args.val_split, args.download_data, args.seed
        )
    elif args.data_set == DataConfig.FASHION_MNIST:
        data_module = init_fashion_mnist_data_module(
            args.batch_size, args.val_split, args.download_data, args.seed
        )
    else:
        raise SystemExit(f"Unknown data set specified: {args.data_set}. Exiting.")

    if args.test:
        test(data_module, args)
    else:
        train(data_module, args)


if __name__ == "__main__":
    run()
