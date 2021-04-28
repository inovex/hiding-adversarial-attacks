import glob
import os
import shutil

from pytorch_lightning.callbacks import Callback


class NeptuneLoggingCallback(Callback):
    def __init__(self, log_path: str, trash_run=False):
        self.log_path = log_path
        self.trash_run = trash_run

    def on_train_end(self, trainer, pl_module):
        cwd = os.getcwd()
        if not self.trash_run:
            self._move_logs_and_outputs(
                self.log_path, cwd, trainer.logger.name, trainer.logger.version
            )
            self._upload_checkpoints(trainer)

    @staticmethod
    def _move_logs_and_outputs(log_path, cwd, experiment_name, run_id):
        hydra_experiment_path = os.path.join(cwd, experiment_name, run_id)
        # move model checkpoints
        shutil.move(
            os.path.join(hydra_experiment_path, "checkpoints"),
            os.path.join(log_path, "checkpoints"),
        )
        # move log files
        for log in glob.glob(os.path.join(cwd, "*.log")):
            shutil.move(
                log,
                os.path.join(log_path, os.path.basename(log)),
            )

    def _upload_checkpoints(self, trainer):
        for checkpoint in glob.glob(
            os.path.join(self.log_path, "checkpoints", "*.ckpt")
        ):
            model_name = "checkpoints/" + checkpoint.split("/")[-1]
            trainer.logger.experiment.log_artifact(checkpoint, model_name)


class ModelManipulationNeptuneLoggingCallback(NeptuneLoggingCallback):
    def __init__(self, log_path: str, image_log_path: str, trash_run=False):
        super().__init__(log_path=log_path, trash_run=trash_run)
        self.image_log_path = image_log_path

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self._upload_image_log(trainer)

    def _upload_image_log(self, trainer):
        for image_path in glob.glob(os.path.join(self.image_log_path, "*.png")):
            image_name = f"image_log/{os.path.basename(image_path)}"
            trainer.logger.experiment.log_artifact(image_path, image_name)
