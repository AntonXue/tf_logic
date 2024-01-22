"""
Script for running experiments/synthetic_experiments_base.py with multiple sets of parameters.
The parameters are specified in a JSON config file (see synthetic_experiments_config.json for an example).
Each combination of parameters is an experiment which is run on a free GPU. 

Usage:
    python3 run_synthetic_experiments.py --config_file <path_to_config_file>

TODO: Add support for logging status of experiments to a file.
"""

import argparse
import subprocess
import itertools
import time
import os
import json


def get_param_dict_from_config_file(config_file) -> dict:
    """Returns a dictionary of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config

def get_param_dict_list_from_config_file(config_file) -> list:
    """Returns a list of dictionaries of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config

def get_gpu_tracker() -> dict:
    """Returns a dictionary mapping GPU IDs to the process running on them."""
    try:
        gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        gpu_list = [int(gpu_id) for gpu_id in gpu_list]
    except KeyError:
        # If the CUDA_VISIBLE_DEVICES environment variable is not set, use all GPUs
        import torch
        gpu_list = [int(x) for x in range(torch.cuda.device_count())]
    return {gpu: None for gpu in gpu_list}


def get_free_gpu(gpu_tracker) -> int:
    """Returns the ID of a free GPU, or None if none are free."""
    for gpu in gpu_tracker.keys():
        if (gpu_tracker[gpu] is None) or (gpu_tracker[gpu].poll() is not None):
            return gpu
    return None


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

def log_experiment_status(json_file, experiment_id, gpu, pid, status):
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
            "gpu": gpu,
            "pid": pid,
            "status": status,
            "start_time": time.time()
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
        default="test_synthetic_experiments_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="test_synthetic_experiments_log.json",
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

    gpu_tracker = get_gpu_tracker()
    print("GPU Tracker: ", gpu_tracker)

    # Iterate over all configurations
    for param_dict in config_list:
        print("Config: ", param_dict)
        # Iterate over all combinations of parameters
        for param_tuple in itertools.product(*param_dict.values()):
            param_str = get_flags_str(param_dict, param_tuple)

            # Check if any gpus are free
            proc_gpu = get_free_gpu(gpu_tracker)
            while proc_gpu is None:
                time.sleep(5)
                proc_gpu = get_free_gpu(gpu_tracker)

            print("Using GPU {}".format(proc_gpu))
            command = "CUDA_VISIBLE_DEVICES={} python3 experiments/synthetic_experiments_base.py {}".format(
                proc_gpu, param_str
            )
            print("Running command: {}".format(command))

            # Run experiment on free gpu
            gpu_tracker[proc_gpu] = subprocess.Popen(command, shell=True)

            # Get PID of process
            print("PID: {}".format(gpu_tracker[proc_gpu].pid))

            # Log experiment status
            log_experiment_status(log_file, param_str, proc_gpu, gpu_tracker[proc_gpu].pid, "running")

    # Wait for all processes to finish
    for gpu in gpu_tracker.keys():
        if gpu_tracker[gpu] is not None:
            print(
                "Waiting for process {} on GPU {} to finish".format(
                    gpu_tracker[gpu].pid, gpu
                )
            )
            gpu_tracker[gpu].wait()
