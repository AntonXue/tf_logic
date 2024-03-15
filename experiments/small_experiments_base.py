import sys
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

seed = 111
torch.manual_seed(seed)

n = 32
num_epochs = 64
batch_size = 2048
output_dir = str(Path(DUMP_DIR, "small_experiments"))
dataset_len = int(2**18)

models = [
    SmallGpt2(n, 4*n, loss_fn="bce"),
    SmallGpt2(n, 4*n, loss_fn="margin"),
    SmallTfA(n, 4*n, loss_fn="bce"),
    SmallTfA(n, 4*n, loss_fn="margin"),
    SmallTfB(n, loss_fn="bce"),
    SmallTfB(n, loss_fn="margin"),
] \
+ [SmallTfC(n, attn_fn=a, use_residual=r, loss_fn=l) \
    for a in ["sigmoid", "relu"] \
    for r in [True, False] \
    for l in ["bce", "margin"]
]


datasets = [
    SmallTfSuccTokensDataset(n, dataset_len),
    SmallAutoreg1StepTokensDataset(n, dataset_len)
]


total_iters = len(models) * len(datasets)

for (di, dataset) in enumerate(datasets):
    for (mi, model) in enumerate(models):
        print(f"Starting iteration {di*len(models) + mi + 1}/{total_iters}")

        desc_str = f"SmallSucc_{model.desc_str}_{dataset.desc_str}_seed{seed}"

        training_args = TrainingArguments(
            str(Path(output_dir, desc_str)),
            num_train_epochs = num_epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            auto_find_batch_size = True,
            evaluation_strategy = "epoch",
            report_to = "wandb",
            run_name = desc_str,
            logging_steps = 16,
            learning_rate = 5e-4,
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



