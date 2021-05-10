import glob
import os
import shutil


def copy_run_outputs(log_path, cwd, experiment_name, run_id):
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
