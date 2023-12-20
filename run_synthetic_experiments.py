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


def get_param_dict_from_config_file(config_file):
    """Returns a dictionary of parameters from a JSON config file."""
    import json

    config = json.load(open(config_file, "r"))
    return config


def get_gpu_tracker():
    """Returns a dictionary mapping GPU IDs to the process running on them."""
    try:
        gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        gpu_list = [int(gpu_id) for gpu_id in gpu_list]
    except KeyError:
        # If the CUDA_VISIBLE_DEVICES environment variable is not set, use all GPUs
        import torch

        gpu_list = [int(x) for x in range(torch.cuda.device_count())]
    return {gpu: None for gpu in gpu_list}


def get_free_gpu(gpu_tracker):
    """Returns the ID of a free GPU, or None if none are free."""
    for gpu in gpu_tracker.keys():
        if (gpu_tracker[gpu] is None) or (gpu_tracker[gpu].poll() is not None):
            return gpu
    return None


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

    gpu_tracker = get_gpu_tracker()
    print("GPU Tracker: ", gpu_tracker)

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

    # Wait for all processes to finish
    for gpu in gpu_tracker.keys():
        if gpu_tracker[gpu] is not None:
            print(
                "Waiting for process {} on GPU {} to finish".format(
                    gpu_tracker[gpu].pid, gpu
                )
            )
            gpu_tracker[gpu].wait()
