from typing import Any, List, Union

import neptune.new as neptune
from omegaconf import OmegaConf
from pytorch_lightning.loggers import NeptuneLogger

NEPTUNE_PROJECT_NAME = "stefaniestoppel/hiding-adversarial-attacks"


def get_neptune_logger(
    config, experiment_name: str, tags: List[Union[str, Any]] = None
):
    # config dictionary needs to be flattened so that parameters are displayed correctly
    # in Neptune dashboard and can be used for filtering
    flattened_params = NeptuneLogger._flatten_dict(OmegaConf.to_container(config))
    neptune_logger = NeptuneLogger(
        project_name=NEPTUNE_PROJECT_NAME,
        params=flattened_params,
        experiment_name=experiment_name,
        tags=tags,
    )
    return neptune_logger


def init_neptune_run(tags: List[Union[str, Any]] = None):
    run = neptune.init(project=NEPTUNE_PROJECT_NAME, tags=tags)
    return run
