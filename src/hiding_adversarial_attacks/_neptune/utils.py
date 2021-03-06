import os
from functools import reduce
from typing import Any, List, Union

import neptune.new as neptune
import pandas as pd
from neptune.new import Run
from omegaconf import OmegaConf
from pytorch_lightning.loggers import NeptuneLogger

from hiding_adversarial_attacks.config.config import (
    DIRECTORIES_TO_LOG,
    NEPTUNE_PROJECT,
    ROOT_DIR,
)


def get_neptune_logger(
    config, experiment_name: str, tags: List[Union[str, Any]] = None
):
    # config dictionary needs to be flattened so that parameters are displayed correctly
    # in Neptune dashboard and can be used for filtering
    flattened_params = NeptuneLogger._flatten_dict(OmegaConf.to_container(config))
    neptune_logger = NeptuneLogger(
        project_name=NEPTUNE_PROJECT,
        params=flattened_params,
        experiment_name=experiment_name,
        offline_mode=config.neptune_offline_mode,
        tags=tags,
    )
    return neptune_logger


def init_neptune_run(tags: List[Union[str, Any]] = None) -> neptune.Run:
    run = neptune.init(project=NEPTUNE_PROJECT, tags=tags)
    return run


def init_current_neptune_run(run_id: str) -> neptune.Run:
    run = neptune.init(project=NEPTUNE_PROJECT, run=run_id)
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


def log_code(neptune_logger):
    for dir_to_log in DIRECTORIES_TO_LOG:
        for root, subdirs, files in os.walk(
            os.path.join(ROOT_DIR, "src/hiding_adversarial_attacks", dir_to_log)
        ):
            for file in files:
                if file.endswith(".py") and not file.startswith("__init__"):
                    code_file_path = os.path.join(root, file)
                    neptune_logger.log_artifact(
                        code_file_path, f"source_code/{dir_to_log}/{file}"
                    )
