import os

import pandas as pd


def save_similarity_stats_csv(output_path, train_sim_df, stage="train"):
    train_stats = (
        train_sim_df[["mse_sim", "pcc_sim", "orig_label_name"]]
        .groupby("orig_label_name")
        .agg(["mean", "std", "median"])
    )
    train_stats.to_csv(os.path.join(output_path, f"{stage}_sim_stats.csv"))


def load_test_results_as_df(directory: str):
    pre = pd.read_csv(os.path.join(directory, "pre-test_results.csv"), index_col=0)
    post = pd.read_csv(os.path.join(directory, "post-test_results.csv"), index_col=0)
    results_df = pre.append(post)
    results_df.index = ["pre", "post"]
    return results_df
