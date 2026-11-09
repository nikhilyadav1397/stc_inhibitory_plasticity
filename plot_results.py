import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json

from scripts.stats_utils import *
from scripts.plotting import *

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

plt.rcParams.update(
    {
        "font.size": 14,  # base font
        "axes.titlesize": 18,  # title
        "axes.labelsize": 16,  # x/y labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
)


def main(config, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    learning_path = os.path.join(data_dir, "learning1_results.pkl")

    learning = load_pickle_if_exists(learning_path)
    plot_learning_results(learning, config, "./plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to simulation folder (if not provided, use latest)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to config file",
    )
    args = parser.parse_args()
    config = json.load(open(args.config_file, "r"))
    main(config, args.data_dir)
