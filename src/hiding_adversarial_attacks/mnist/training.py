import argparse

from pytorch_lightning import loggers as pl_loggers, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from hiding_adversarial_attacks.config import MNISTConfig, DataConfig
from hiding_adversarial_attacks.data.MNIST import MNISTDataModule
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
        "--logs-dir",
        default=MNISTConfig.LOGS_PATH,
        help="path to store MNIST training logs and checkpoints to",
    )
    parser.add_argument(
        "--download-mnist",
        action="store_true",
        default=False,
        help="download & process MNIST data set",
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
    return args


def init_mnist_data_module(batch_size, download_mnist, seed):
    data_module = MNISTDataModule(
        DataConfig.EXTERNAL_PATH, batch_size=batch_size, random_seed=seed
    )
    if download_mnist:
        data_module.prepare_data()
    data_module.setup()
    return data_module


def train(data_module, args):
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

    checkpoint_callback = ModelCheckpoint(
        monitor=MNISTConfig.VAL_LOSS,
        filename="mnist-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    tb_logger = pl_loggers.TensorBoardLogger(args.logs_dir)
    trainer = Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback]
    )

    mnist_model = MNISTNet(args)

    trainer.fit(mnist_model, train_loader, validation_loader)


def test(data_module, args):
    test_loader = data_module.test_dataloader()

    tb_logger = pl_loggers.TensorBoardLogger(args.logs_dir)
    trainer = Trainer.from_argparse_args(args, logger=tb_logger)

    mnist_model = MNISTNet(args).load_from_checkpoint(args.test_checkpoint)

    trainer.test(mnist_model, test_loader, ckpt_path="best")


def run():
    args = parse_mnist_args()

    data_module = init_mnist_data_module(
        args.batch_size, args.download_mnist, args.seed
    )
    if args.test:
        test(data_module, args)
    else:
        train(data_module, args)


if __name__ == "__main__":
    run()
