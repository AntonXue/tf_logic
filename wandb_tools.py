import os
import wandb
import argparse
import json
from tqdm import tqdm

""" Some wandb and directory setup """

WANDB_PROJECT = "transformer_friends"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

OUTPUT_DIR = "wandb_dump"


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
        print(f"- ({id}) {name}")


def download_run_summaries_with_prefix(prefix):
    print(f"Downloading runs with prefix '{prefix}' ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api = wandb.Api()
    count = 0
    for run in api.runs(WANDB_PROJECT):
        if not run.name.startswith(prefix):
            continue

        summary_dict = dict(run.summary)
        if "_wandb" in summary_dict:
            del summary_dict["_wandb"]

        summary_file = os.path.join(OUTPUT_DIR, "summary_" + run.name + ".json")
        with open(summary_file, "w") as fp:
            json.dump(summary_dict, fp, indent=4)

        print(f"+ {summary_file}")
        count += 1

    print(f"Downloaded {count} runs")


""" Parse args and run """

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-syn-runs", action="store_true", default=False)
    parser.add_argument("--delete-syn-runs", action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    return dict(args._get_kwargs())


if __name__ == "__main__":
    wandb.login()
    args = parse_args()

    if args["download_syn_runs"]:
        download_run_summaries_with_prefix("Syn")

    if args["delete_syn_runs"]:
        delete_runs_with_prefix("Syn")


