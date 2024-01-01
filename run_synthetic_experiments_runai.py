"""
Script for running experiments/synthetic_experiments_base.py with multiple sets of parameters on Runai.
The parameters are specified in a JSON config file (see synthetic_experiments_config.json for an example).
Each combination of parameters is an experiment which is run on a free GPU. 

Usage:
    python3 run_synthetic_experiments_runai.py --config_file <path_to_config_file>
"""

import argparse
import subprocess
import itertools
import time
import os

RUNAI_BASE_COMMAND = "runai submit {0} -i python:3.10 -v /home/akhare/repos/tf_logic:/home/akhare/tf_logic --working-dir /home/akhare/tf_logic -g 1  -e WANDB_API_KEY=aab33abcba77bf0aef847aa73e25b577a3d9afb6 -e USER=67228 --backoff-limit 0 -- ./runai_start.sh {1}"

param_to_id_dict = {
        "model_name": "m",
        "syn_exp_name": "e",
        "embed_dim": "d",
        "num_layers": "nl",
        "num_heads": "nh",
        "min_num_rules": "minnr",
        "max_num_rules": "maxnr",
        "min_num_states": "minns",
        "max_num_states": "maxns",
        "num_vars": "nv",
        "min_ante_prob": "minap",
        "max_ante_prob": "maxap",
        "min_conseq_prob": "mincp",
        "max_conseq_prob": "maxcp",
        "min_state_prob": "minsp",
        "max_state_prob": "maxsp",
        "train_len": "tl",
        "eval_len": "el",
        "num_epochs": "ne",
        "train_batch_size": "tbs",
        "eval_batch_size": "ebs"
    }

def get_param_dict_from_config_file(config_file):
    """Returns a dictionary of parameters from a JSON config file."""
    import json

    config = json.load(open(config_file, "r"))
    return config

def get_exp_id(param_dict, param_tuple):
    exp_id = "tflogic"
    for i, param in enumerate(param_tuple):
        param_id = param_to_id_dict[list(param_dict.keys())[i]]
        if param_id not in ["m", "e", "d", "nl", "nh", "nv"]:
            continue
        if "." in str(param):
            param = "".join(str(param).split("."))
        if "_" in str(param):
            param = "".join(str(param).split("_"))
        exp_id += "-{}{}".format(param_id, param)
    return exp_id

def get_flags_str(param_dict, param_tuple):
    """Returns a string of flags for the experiment script.

    Args:
        param_dict: Dictionary mapping parameter names to lists of possible values.
        param_tuple: Tuple of parameter values to use for this experiment.
    """
    param_str = ""
    for i, param in enumerate(param_tuple):
        param_str += "--{} {} ".format(list(param_dict.keys())[i], param)
    return param_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="test_synthetic_experiments_config.json",
        help="Path to JSON config file.",
    )

    args = parser.parse_args()
    config_file = args.config_file

    # Get dictionary of parameters
    print("Loading parameters from {}".format(config_file))
    param_dict = get_param_dict_from_config_file(config_file)
    print("Parameter dictionary: ", param_dict)

    # Iterate over all combinations of parameters
    for param_tuple in itertools.product(*param_dict.values()):
        exp_id = get_exp_id(param_dict, param_tuple)
        param_str = get_flags_str(param_dict, param_tuple)

        command = RUNAI_BASE_COMMAND.format(exp_id, param_str)
        print("Running command: {}".format(command))

        # Start experiment
        subprocess.run(command, shell=True)
        # exit()