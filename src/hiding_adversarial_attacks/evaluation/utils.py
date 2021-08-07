import os

import pandas as pd

from hiding_adversarial_attacks.config.data_sets.data_set_config import DataSetNames
from hiding_adversarial_attacks.evaluation.config import SIMILARITIES_FILE
from hiding_adversarial_attacks.visualization.config import DATA_SET_MAPPING

PRE_TEST_FILE = "pre-pre-test_results.csv"
POST_TEST_FILE = "post-test_results.csv"


def save_top_and_bottom_post_similarities_stats_fashion_mnist_gradcam():
    data_set_name = DataSetNames.FASHION_MNIST
    # Bottom class: Coat
    bottom_runs_path = (
        "/home/steffi/dev/master_thesis/evaluation/Fashion-MNIST,"
        "Grad-CAM/bottom=Coat-4/original_runs"
    )
    bottom_class_id = 4
    bottom_run_ids = range(4503, 4508)

    # Top class: Sandal
    top_runs_path = (
        "/home/steffi/dev/master_thesis/evaluation/Fashion-MNIST,"
        "Grad-CAM/top=Sandal-5/original_runs"
    )
    top_class_id = 5
    top_run_ids = range(4493, 4497)

    save_top_and_bottom_post_similarities_stats(
        data_set_name,
        bottom_runs_path,
        top_runs_path,
        bottom_run_ids,
        top_run_ids,
        top_class_id,
        bottom_class_id,
    )


def save_top_and_bottom_post_similarities_stats(
    data_set_name,
    bottom_runs_path,
    top_runs_path,
    bottom_run_ids,
    top_run_ids,
    top_class_id,
    bottom_class_id,
):
    top_class_name = DATA_SET_MAPPING[data_set_name][top_class_id]
    bottom_class_name = DATA_SET_MAPPING[data_set_name][bottom_class_id]
    bottom_post_sim = load_post_manipulation_test_similarities(
        bottom_runs_path, bottom_run_ids
    )
    top_post_sim = load_post_manipulation_test_similarities(top_runs_path, top_run_ids)
    # Save concatenated data across runs
    top_post_sim.to_csv(
        os.path.join(
            top_runs_path,
            f"post_top_class_{top_class_name}_similarities_all_runs.csv",
        )
    )
    bottom_post_sim.to_csv(
        os.path.join(
            bottom_runs_path,
            f"post_bottom_class_{bottom_class_name}_similarities_all_runs.csv",
        )
    )
    # Save similarity stats csv
    top_post_sim.groupby("orig_label_name")[["mse_sim", "pcc_sim", "ssim_sim"]].agg(
        ["mean", "std", "median"]
    ).to_csv(
        os.path.join(
            top_runs_path,
            f"post_top_class_{top_class_name}_similarities_stats.csv",
        )
    )
    bottom_post_sim.groupby("orig_label_name")[["mse_sim", "pcc_sim", "ssim_sim"]].agg(
        ["mean", "std", "median"]
    ).to_csv(
        os.path.join(
            bottom_runs_path,
            f"post_bottom_class_{bottom_class_name}_similarities_stats.csv",
        )
    )


def load_pre_manipulation_test_similarities(data_set_path):
    sim_df = pd.read_csv(os.path.join(data_set_path, SIMILARITIES_FILE), index_col=0)
    return sim_df


def load_post_manipulation_test_similarities(runs_path, run_ids):
    df = None
    for run_id in run_ids:
        run_df = pd.read_csv(
            os.path.join(runs_path, f"HAA-{run_id}", SIMILARITIES_FILE),
            index_col=0,
        )
        df = pd.concat([df, run_df])
    return df


def load_pre_and_post_test_results_for_run_ids(directory, run_ids):
    pre_df, post_df = None, None
    for run_id in run_ids:
        run_id_str = f"HAA-{run_id}"
        dirpath = os.path.join(directory, run_id_str)
        if not os.path.exists(dirpath):
            continue
        pre_test_df = pd.read_csv(
            os.path.join(dirpath, PRE_TEST_FILE),
            index_col=0,
        )
        pre_test_df.index = [run_id_str]
        post_test_df = pd.read_csv(
            os.path.join(dirpath, POST_TEST_FILE),
            index_col=0,
        )
        post_test_df.index = [run_id_str]
        pre_df = pd.concat([pre_df, pre_test_df])
        post_df = pd.concat([post_df, post_test_df])
    return pre_df, post_df


if __name__ == "__main__":
    save_top_and_bottom_post_similarities_stats_fashion_mnist_gradcam()
