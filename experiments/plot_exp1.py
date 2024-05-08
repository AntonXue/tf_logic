"""
Plot the heatmap for results from Experiment 1.

Usage:
    python experiments/plot_exp1.py --config_file <config_file> --metric <metric> --plot_file <plot_file>

Example for Experiment 1:
    python experiments/plot_exp1.py --config_file synthetic_experiments_config_exp1_plot.json --metric eval/TargetStatesAcc --plot_file test_synthetic_experiments_plot.png --plot_x num_vars --plot_y embed_dim
"""

import argparse
import itertools
import json
from pathlib import Path
from utils.model_loader_utils import load_big_grid_stats_from_wandb
from utils.plotting import heatmap, annotate_heatmap
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = str(Path(Path(__file__).parent.parent, "plots"))


plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.style.use('seaborn-v0_8-colorblind')

def get_param_dict_list_from_config_file(config_file) -> list:
    """Returns a list of dictionaries of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config

def get_param_mapping(param_dict, param_tuple) -> str:
    """Returns a dictionary that maps keys in param_dict to values in param_tuple.

    Args:
        param_dict: Dictionary mapping parameter names to lists of possible values.
        param_tuple: Tuple of parameter values to use for this experiment.
    """
    param_instance = {}
    for i, param in enumerate(param_tuple):
        param_instance[list(param_dict.keys())[i]] = param
    return param_instance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type = str,
        help = "Path to JSON config file.",
    )

    parser.add_argument(
        "--metric",
        type = str,
        default = "eval/exph3.000_UpToStep3Acc",
        help = "Path to save plot.",
    )

    parser.add_argument(
        "--plot_file",
        default = None,
        type = str,
        help = "Path to save plot.",
    )

    parser.add_argument(
        "--aggregate_by",
        type = str,
        default = "mean",
        choices = ["mean", "std", "median"],
        help = "Metric to use for aggregating when there are multiple values for the same (plot_x, plot_y) pair. Default: mean.",
    )

    args = parser.parse_args()
    return args


def plot_and_save_heatmap(args):
    config_file = args.config_file

    # Get list of configurations
    print("Loading configurations from {}".format(config_file))
    config_list = get_param_dict_list_from_config_file(config_file)
    print("Config list: ", config_list)

    param_dict = config_list[0] # We care only about the first one in this case

    metric_dict = {n: {d: [] for d in param_dict["embed_dim"]} for n in param_dict["num_vars"]}

    # Iterate over all combinations of parameters
    for param_tuple in itertools.product(*param_dict.values()):
        param_mapping = get_param_mapping(param_dict, param_tuple)
        n, d = param_mapping["num_vars"], param_mapping["embed_dim"]

        try:
            stats = load_big_grid_stats_from_wandb(
                embed_dim = d,
                num_vars = n,
                num_heads = param_mapping["num_heads"],
                model_name = param_mapping["model_name"],
                num_layers = param_mapping["num_layers"],
                num_rules = param_mapping["num_rules"],
                exph = param_mapping["exp_hots"],
                train_len = param_mapping["train_len"],
                eval_len = param_mapping["eval_len"],
                learning_rate = param_mapping["learning_rate"],
                batch_size = param_mapping["batch_size"],
                seed = param_mapping["seed"],
            )

            metric_val = stats[args.metric]
        except:
            metric_val = -1
        
        metric_dict[n][d].append(metric_val)

    # Aggregate metric values for each (x, y) pair
    if args.aggregate_by == "mean":
        for n in metric_dict.keys():
            for d in metric_dict[n].keys():
                metric_dict[n][d] = np.mean(metric_dict[n][d])
    elif args.aggregate_by == "std":
        for n in metric_dict.keys():
            for d in metric_dict[n].keys():
                metric_dict[n][d] = np.std(metric_dict[n][d])
    elif args.aggregate_by == "median":
        for n in metric_dict.keys():
            for d in metric_dict[n].keys():
                metric_dict[n][d] = np.median(metric_dict[n][d])
    else:
        raise Exception(f"Unknown aggregation method: {args.aggregate_by}")

    # Create a 2D array of metric values (for empty cells, set value to -1)
    # Sort by x and y
    n_keys = sorted(metric_dict.keys())
    d_keys = sorted(metric_dict[n_keys[0]].keys(), reverse=True)

    data = np.array([[metric_dict[n][d] for n in n_keys] for d in d_keys])
    print(data)

    fig, ax = plt.subplots()
    im, cbar = heatmap(data, d_keys, n_keys,
        row_title = "Number of Propositions",
        col_title = "Embedding Dimension",
        ax = ax,
        cmap = "YlGn",
    )
    
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()

    if args.plot_file is None:
        metric_str = args.metric.replace("/","_")
        plot_file = str(Path(PLOTS_DIR, f"autoreg_{metric_str}.png"))

    plt.savefig(plot_file, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    args = parse_args()
    ret = plot_and_save_heatmap(args)

