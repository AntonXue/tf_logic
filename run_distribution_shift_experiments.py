"""
Script for running experiments/distribution_shift_experiments_base.py with multiple sets of parameters.
The parameters are specified in a JSON config file (see test_distribution_shift_experiments_config.json for an example).
Each combination of parameters is an experiment. 

Usage:
    python3 run_distribution_shift_experiments.py --config_file <path_to_config_file> --log_file <path_to_log_file>
"""

import argparse
import subprocess
import itertools
import time
import os
import json


def get_param_dict_list_from_config_file(config_file) -> list:
    """Returns a list of dictionaries of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config

def get_flags_str(param_dict, param_tuple) -> str:
    """Returns a string of flags for the experiment script.

    Args:
        param_dict: Dictionary mapping parameter names to lists of possible values.
        param_tuple: Tuple of parameter values to use for this experiment.
    """
    param_str = ""
    for i, param in enumerate(param_tuple):
        param_str += "--{} {} ".format(list(param_dict.keys())[i], param)
    return param_str

def log_experiment_status(json_file, experiment_id, pid, status):
    """Logs the status of an experiment to json file."""
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
            "start_time": time.time(),
            "pid": pid
        }
    else:
        data[experiment_id]["status"] = status
        data[experiment_id]["end_time"] = time.time()

    with open(json_file, "w") as f:
        json.dump(data, f)


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
            param_str = get_flags_str(param_dict, param_tuple)

            command = "python3 experiments/distribution_shift_experiments_base.py {}".format(param_str)
            print("Running command: {}".format(command))

            # Run experiment
            proc = subprocess.Popen(command, shell=True)

            # Get PID of process
            print("PID: {}".format(proc.pid))

            # Log experiment status
            log_experiment_status(log_file, param_str, proc.pid, "running")

            # Wait for process to finish
            proc.wait()
            log_experiment_status(log_file, param_str, proc.pid, "finished")