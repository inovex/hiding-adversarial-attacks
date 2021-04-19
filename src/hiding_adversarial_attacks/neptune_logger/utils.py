from omegaconf import OmegaConf
from pytorch_lightning.loggers import NeptuneLogger


def get_neptune_logger(config):
    # config dictionary needs to be flattened so that parameters are displayed correctly
    # in Neptune dashboard and can be used for filtering
    flattened_params = NeptuneLogger._flatten_dict(OmegaConf.to_container(config))
    OmegaConf.structured()
    neptune_logger = NeptuneLogger(
        project_name="stefaniestoppel/hiding-adversarial-attacks",
        params=flattened_params,
    )
    return neptune_logger
