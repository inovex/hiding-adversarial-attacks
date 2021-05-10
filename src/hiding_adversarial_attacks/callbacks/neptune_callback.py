import glob
import os

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from hiding_adversarial_attacks.callbacks.utils import copy_run_outputs


class NeptuneLoggingCallback(Callback):
    def __init__(self, log_path: str, image_log_path: str = None, trash_run=False):
        self.log_path = log_path
        self.image_log_path = image_log_path or log_path
        self.trash_run = trash_run

    def on_train_end(self, trainer, pl_module: LightningModule):
        cwd = os.getcwd()
        if not self.trash_run:
            copy_run_outputs(
                self.log_path, cwd, trainer.logger.name, trainer.logger.version
            )
            # self._upload_image_log(trainer)
            # self._upload_checkpoints(trainer)

    def _upload_image_log(self, trainer):
        for image_path in glob.glob(os.path.join(self.image_log_path, "*.png")):
            image_name = f"image_log/{os.path.basename(image_path)}"
            trainer.logger.experiment.log_artifact(image_path, image_name)

    def _upload_checkpoints(self, trainer):
        for checkpoint in glob.glob(
            os.path.join(self.log_path, "checkpoints", "*.ckpt")
        ):
            model_name = "checkpoints/" + checkpoint.split("/")[-1]
            trainer.logger.experiment.log_artifact(checkpoint, model_name)
