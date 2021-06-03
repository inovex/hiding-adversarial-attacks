from pytorch_lightning.callbacks import EarlyStopping


class CustomEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        self._run_early_stopping_check(trainer, pl_module)
        if trainer.should_stop:
            trainer.logger.experiment.append_tags(["early_stop", "pruned"])
