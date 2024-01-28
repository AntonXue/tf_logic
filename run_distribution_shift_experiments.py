"""
Script for running experiments/distribution_shift_experiments_base.py with multiple sets of parameters.
The parameters are specified in a JSON config file (see test_distribution_shift_experiments_config.json for an example).
Each combination of parameters is an experiment. 

Usage:
    python3 run_distribution_shift_experiments.py --config_file <path_to_config_file> --log_file <path_to_log_file>
    python run_distribution_shift_experiments.py --config_file synthetic_experiments_config_exp3_plot.json --log_file distribution_shift_exps.json
"""

import argparse
import subprocess
import itertools
import time
import os
import json
from pathlib import Path
from experiments.common import * 
import numpy as np

common_params = ["output_dir", "syn_exp_name", "include_seed_in_run_name", "model_name", "embed_dim", "num_layers", "num_heads", "batch_size", "seed", "use_gpu"]
train_params = ["train_num_vars", "train_min_num_rules", "train_max_num_rules", "train_num_steps", "train_min_num_states", "train_max_num_states", "train_min_ante_prob", "train_max_ante_prob", "train_min_conseq_prob", "train_max_conseq_prob", "train_min_state_prob", "train_max_state_prob", "train_min_chain_len", "train_max_chain_len", "train_len", "train_eval_len"]
eval_params = ["eval_num_vars", "eval_min_num_rules", "eval_max_num_rules", "eval_num_steps", "eval_min_num_states", "eval_max_num_states", "eval_min_ante_prob", "eval_max_ante_prob", "eval_min_conseq_prob", "eval_max_conseq_prob", "eval_min_state_prob", "eval_max_state_prob", "eval_min_chain_len", "eval_max_chain_len", "eval_eval_len"]

output_dir = str(Path(DUMP_DIR, "distribution_shift_experiments"))

# List of OOD datasets to use for evaluation
EVAL_PROB_DICT_LIST = [
    {
        "min_ante_prob": k,
        "max_ante_prob": k,
        "min_conseq_prob": k,
        "max_conseq_prob": k,
    }
    for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
]

# EVAL_RULES_DICT_LIST = [
#     {
#         "min_num_rules": k,
#         "max_num_rules": k,
#     }
#     for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# ]

EVAL_RULES_DICT_LIST = [
    {
        "min_num_rules": k,
        "max_num_rules": k,
    }
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
]

def get_param_dict_list_from_config_file(config_file) -> list:
    """Returns a list of dictionaries of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config

def get_flags_str(param_dict, param_tuple, eval_dict) -> str:
    """Returns a string of flags for the experiment script.

    Args:
        param_dict: Dictionary mapping parameter names to lists of possible values.
        param_tuple: Tuple of parameter values to use for this experiment.
        eval_dict: Dictionary mapping evaluation parameter names to lists of possible values.
                If a parameter is not in this dictionary, then its value is the same as the corresponding training parameter.
    """
    param_str = ""
    for i, param in enumerate(param_tuple):
        param_key = list(param_dict.keys())[i]
        if list(param_dict.keys())[i] in common_params:
            param_str += "--{} {} ".format(param_key, param)
        else:
            train_param_key = param_key if param_key.startswith("train_") else f"train_{param_key}"
            if train_param_key in train_params:
                param_str += "--{} {} ".format(train_param_key, param)
            if f"eval_{param_key}" in eval_params:
                if param_key not in eval_dict:
                    param_str += "--eval_{} {} ".format(param_key, param)
                else:
                    param_str += "--eval_{} {} ".format(param_key, eval_dict[param_key])
    return param_str

def log_experiment_status(json_file, experiment_id, status):
    """Logs the status of an experiment to json file.
    
    Returns False is the experiment is already finished.
    """
    # If experiment_id is not in json_file, add it
    # If experiment_id is in json_file, update its status

    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if experiment_id not in data.keys():
        data[experiment_id] = {
            "status": status,
            "start_time": time.time()
        }
    elif data[experiment_id]["status"] == "finished":
        print("Experiment {} already finished.".format(experiment_id))
        return False
    else:
        data[experiment_id]["status"] = status
        data[experiment_id]["end_time"] = time.time()

    with open(json_file, "w") as f:
        json.dump(data, f)

    return True

def run_experiment(param_dict, param_tuple, eval_dict, log_file):
    param_str = get_flags_str(param_dict, param_tuple, eval_dict)

    command = "python3 experiments/distribution_shift_experiments_base.py {}".format(param_str)
    print("Running command: {}".format(command))

    if log_experiment_status(log_file, param_str, "running"):
        # Run experiment
        ret = subprocess.call(command, shell=True)
    log_experiment_status(log_file, param_str, "finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="test_distribution_shift_experiments_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="test_distribution_shift_experiments_log.json",
        help="Path to JSON log file.",
    )

    args = parser.parse_args()
    config_file = args.config_file
    log_file = args.log_file

    # Get list of configurations
    print("Loading configurations from {}".format(config_file))
    config_list = get_param_dict_list_from_config_file(config_file)
    print("Config list: ", config_list)
    print("Number of configurations: ", len(config_list))

    # Iterate over all configurations
    for param_dict in config_list:
        print("Config: ", param_dict)
        # Iterate over all combinations of parameters
        for param_tuple in itertools.product(*param_dict.values()):

            # Iterate over all combinations of evaluation parameters
            for eval_prob_dict in EVAL_PROB_DICT_LIST:
                run_experiment(param_dict, param_tuple, eval_prob_dict, log_file)

            for eval_rules_dict in EVAL_RULES_DICT_LIST:
                run_experiment(param_dict, param_tuple, eval_rules_dict, log_file)
            