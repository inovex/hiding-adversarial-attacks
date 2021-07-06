import os
from functools import reduce
from typing import Any, List, Union

import neptune.new as neptune
import pandas as pd
from neptune.new import Run
from omegaconf import OmegaConf
from pytorch_lightning.loggers import NeptuneLogger

from hiding_adversarial_attacks.config.config import NEPTUNE_PROJECT_NAME


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


def init_neptune_run(tags: List[Union[str, Any]] = None) -> neptune.Run:
    run = neptune.init(project=NEPTUNE_PROJECT_NAME, tags=tags)
    return run


def init_current_neptune_run(run_id: str) -> neptune.Run:
    run = neptune.init(
        project=NEPTUNE_PROJECT_NAME, run=run_id  # for example 'SAN-123'
    )
    return run


def save_run_data(run: Run, log_path: str, stage: str = "train"):
    assert stage in ["train", "val", "test"], f"Unknown stage: '{stage}'."

    logs = run["logs"]
    metrics = [
        "exp_sim",
        "ce_orig",
        "ce_adv",
        "adv_acc",
        "orig_acc",
        "exp_pcc",
        "exp_ssim",
        "exp_mse",
        "normalized_total_loss",
    ]
    # Train
    stage_metrics = [f"{stage}_{m}" for m in metrics]
    dfs = []
    for tm in stage_metrics:
        if run.exists(f"logs/{tm}"):
            df = logs[tm].fetch_values(include_timestamp=False)
            df.columns = ["step", tm]
            dfs.append(df)
    metrics_df = reduce(
        lambda left, right: pd.merge(left, right, on=["step"], how="outer"),
        dfs,
    )
    metrics_df.to_csv(os.path.join(log_path, f"{stage}_metrics.csv"), index=False)
