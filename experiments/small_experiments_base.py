import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import Trainer, TrainingArguments
import wandb
import json

""" Our imports """
from common import *    # Critical definitions and path inserts
from models import *
from my_datasets import *
from utils.metrics import *
from utils.model_loader_utils import load_checkpoint_from_wandb
from transformers import Trainer, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=16)
    parser.add_argument("--seed", default=1234)
    parser.add_argument("--lr", default=5e-4)
    parser.add_argument("--num_epochs", default=int(16))
    parser.add_argument("--batch_size", default=int(2**7))
    parser.add_argument("--dataset_len", default=int(2**18))
    parser.add_argument("--output_dir", default=str(Path(DUMP_DIR, "small_experiments")))
    args = parser.parse_args()
    return args

args = parse_args()


todos = [
    (SmallTfE(args.n), SmallTfSuccTokensDataset(args.n, args.dataset_len), args.seed),
]


for i, (model, dataset, seed) in enumerate(todos):
    torch.manual_seed(seed)
    print(f"\n>>> Starting {i+1}/{len(todos)}\n")

    desc_str = f"SMS_{model.desc_str}_{dataset.desc_str}_lr{args.lr:.5f}_seed{seed}"

    training_args = TrainingArguments(
        str(Path(args.output_dir, desc_str)),
        num_train_epochs = args.num_epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        auto_find_batch_size = True,
        evaluation_strategy = "epoch",
        report_to = "wandb",
        run_name = desc_str,
        logging_steps = 32,
        learning_rate = args.lr,
        warmup_ratio = 0.10,
        save_strategy = "epoch",
        save_total_limit = 2
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset = dataset,
        eval_dataset = dataset,
        compute_metrics = small_succ_metrics
    )

    trainer.train()
    wandb.finish()



