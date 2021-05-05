import glob
import os
import shutil

from pytorch_lightning.callbacks import Callback


class NeptuneLoggingCallback(Callback):
    def __init__(self, log_path: str, image_log_path: str = None, trash_run=False):
        self.log_path = log_path
        self.image_log_path = image_log_path or log_path
        self.trash_run = trash_run

    def on_train_end(self, trainer, pl_module):
        cwd = os.getcwd()
        if not self.trash_run:
            self._copy_logs_and_outputs(
                self.log_path, cwd, trainer.logger.name, trainer.logger.version
            )
            # self._upload_image_log(trainer)
            # self._upload_checkpoints(trainer)

    @staticmethod
    def _copy_logs_and_outputs(log_path, cwd, experiment_name, run_id):
        hydra_experiment_path = os.path.join(cwd, experiment_name, run_id)
        # copy model checkpoints
        checkpoint_src = os.path.join(hydra_experiment_path, "checkpoints")
        checkpoints_dest = os.path.join(log_path, "checkpoints")
        os.makedirs(checkpoints_dest, exist_ok=True)
        for checkpoint in glob.glob(os.path.join(checkpoint_src, "*.ckpt")):
            shutil.copy(
                checkpoint,
                os.path.join(checkpoints_dest, os.path.basename(checkpoint)),
            )

        # copy log files
        for log in glob.glob(os.path.join(cwd, "*.log")):
            shutil.copy(
                log,
                os.path.join(log_path, os.path.basename(log)),
            )

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
