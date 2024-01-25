import argparse
import itertools
import json
from utils.model_loader_utils import load_stats_from_wandb
from utils.plotting import heatmap, annotate_heatmap
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="test_synthetic_experiments_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval/TargetStatesAcc",
        help="Metric to plot",
    )
    parser.add_argument(
        "--plot_file",
        type=str,
        default="test_synthetic_experiments_plot.png",
        help="Path to save plot.",
    )
    parser.add_argument(
        "--plot_x",
        type=str,
        default="num_vars",
        help="Parameter to plot on x-axis.",
    )
    parser.add_argument(
        "--plot_y",
        type=str,
        default="embed_dim",
        help="Parameter to plot on y-axis.",
    )
    
    args = parser.parse_args()
    config_file = args.config_file
    
    # Get list of configurations
    print("Loading configurations from {}".format(config_file))
    config_list = get_param_dict_list_from_config_file(config_file)
    print("Config list: ", config_list)

    # Create a dictionary with (x, y) coordinates as keys and metric values as values
    metric_dict = {}

    # Iterate over all configurations
    for param_dict in config_list:
        # print("Config: ", param_dict)
        # Iterate over all combinations of parameters
        for param_tuple in itertools.product(*param_dict.values()):
            param_mapping = get_param_mapping(param_dict, param_tuple)
            try:
                stats = load_stats_from_wandb(
                    model_name=param_mapping["model_name"],
                    embed_dim=param_mapping["embed_dim"],
                    num_layers=param_mapping["num_layers"],
                    num_heads=param_mapping["num_heads"],
                    num_vars=param_mapping["num_vars"],
                    num_steps=param_mapping["num_steps"],
                    num_rules_range=(param_mapping["min_num_rules"], param_mapping["max_num_rules"]),
                    ante_prob_range = (param_mapping["min_ante_prob"], param_mapping["max_ante_prob"]),
                    conseq_prob_range = (param_mapping["min_conseq_prob"], param_mapping["max_conseq_prob"]),
                    train_len = param_mapping["train_len"],
                    eval_len = param_mapping["eval_len"],
                    syn_exp_name=param_mapping["syn_exp_name"],
                    seed=param_mapping["seed"],
                )
                metric_val = stats[args.metric]
            except:
                metric_val = -1

            if param_mapping[args.plot_x] not in metric_dict.keys():
                metric_dict[param_mapping[args.plot_x]] = {}
            metric_dict[param_mapping[args.plot_x]][param_mapping[args.plot_y]] = metric_val


        # Create a 2D array of metric values (for empty cells, set value to -1)
        # Sort by x and y
        x_keys = sorted(metric_dict.keys())
        y_keys = sorted(metric_dict[x_keys[0]].keys(), reverse=True)

        data = np.array([[metric_dict[x][y] for x in x_keys] for y in y_keys])
        print(data)

        fig, ax = plt.subplots()
        im, cbar = heatmap(data, y_keys, x_keys, row_title=args.plot_y, col_title=args.plot_x, ax=ax,
                           cmap="YlGn", cbarlabel=args.metric)
        
        texts = annotate_heatmap(im, valfmt="{x:.2f}")

        fig.tight_layout()
        plt.savefig(args.plot_file)



