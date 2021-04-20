import os

from pytorch_lightning.callbacks import Callback


class NeptuneLoggingCallback(Callback):
    def __init__(self, trash_run=False):
        self.trash_run = trash_run

    def on_train_end(self, trainer, pl_module):
        if not self.trash_run:
            self._upload_hydra_files(trainer)
            self._upload_checkpoints(trainer)

    @staticmethod
    def _upload_hydra_files(trainer):
        cwd = os.getcwd()
        hydra_dir = os.path.join(cwd, ".hydra")
        trainer.logger.experiment.log_artifact(hydra_dir, "hydra")

    @staticmethod
    def _upload_checkpoints(trainer):
        for k in trainer.checkpoint_callback.best_k_models.keys():
            model_name = "checkpoints/" + k.split("/")[-1]
            trainer.logger.experiment.log_artifact(k, model_name)
