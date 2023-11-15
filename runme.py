import os
from typing import Optional
from dataclasses import asdict, dataclass, field
from transformers import HfArgumentParser
import wandb

""" """

from experiments import AutoTrainer, SyntheticExperimentArguments


""" wandb setup """
WANDB_PROJECT = "transformer_friends"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


""" Main stuff """

if __name__ == "__main__":
    parser = HfArgumentParser(SyntheticExperimentArguments)
    synexp_args = parser.parse_args_into_dataclasses()[0]

    if synexp_args.syn_exp_name in ["one_shot", "next_state", "autoreg_ksteps"]:
        trainer = AutoTrainer.from_synthetic_experiment_args(synexp_args)
        trainer.train()
        wandb.finish()

