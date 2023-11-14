import os
import wandb
import argparse
from tqdm import tqdm


""" Some wandb setup """

WANDB_PROJECT = "transformer_friends"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


""" Functionalities """

def delete_runs_with_prefix(prefix):
    print(f"Deleting runs with prefix '{prefix}' ...")
    api = wandb.Api()
    deleted_runs = []
    pbar = tqdm(api.runs(WANDB_PROJECT))
    for run in pbar:
        if run.name.startswith(prefix):
            deleted_runs.append((run.id, run.name))
            run.delete()
            pbar.set_description(f"Deleted {len(deleted_runs)}")

    print(f"Deleted {len(deleted_runs)} runs:")
    for id, name in deleted_runs:
        print(f"  - ({id}) {name}")


""" Parse args and run """

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-syn-runs", action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    return dict(args._get_kwargs())


if __name__ == "__main__":
    wandb.login()
    args = parse_args()

    if args["delete_syn_runs"]:
        delete_runs_with_prefix("Syn")


