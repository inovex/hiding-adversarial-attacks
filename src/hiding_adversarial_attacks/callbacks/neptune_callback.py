import glob
import os
import shutil

from pytorch_lightning.callbacks import Callback


class NeptuneLoggingCallback(Callback):
    def __init__(self, log_path: str, trash_run=False):
        self.log_path = log_path
        self.trash_run = trash_run

    def on_train_end(self, trainer, pl_module):
        if not self.trash_run:
            cwd = os.getcwd()
            self._move_logs_and_outputs(
                self.log_path, cwd, trainer.logger.name, trainer.logger.version
            )
            self._upload_hydra_files(trainer)
            self._upload_checkpoints(trainer)

    @staticmethod
    def _move_logs_and_outputs(log_path, cwd, experiment_name, run_id):
        hydra_experiment_path = os.path.join(cwd, experiment_name, run_id)
        # move model checkpoints
        shutil.move(
            os.path.join(hydra_experiment_path, "checkpoints"),
            os.path.join(log_path, "checkpoints"),
        )
        # move hydra yaml files
        shutil.move(
            os.path.join(cwd, ".hydra"),
            os.path.join(log_path, ".hydra"),
        )
        # move log files
        for log in glob.glob(os.path.join(cwd, "*.log")):
            shutil.move(
                log,
                os.path.join(log_path, log),
            )

    def _upload_hydra_files(self, trainer):
        hydra_dir = os.path.join(self.log_path, ".hydra")
        trainer.logger.experiment.log_artifact(hydra_dir, "hydra")

    def _upload_checkpoints(self, trainer):
        for checkpoint in glob.glob(
            os.path.join(self.log_path, "checkpoints", "*.ckpt")
        ):
            model_name = "checkpoints/" + checkpoint.split("/")[-1]
            trainer.logger.experiment.log_artifact(checkpoint, model_name)
