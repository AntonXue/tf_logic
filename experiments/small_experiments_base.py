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

seed = 121
torch.manual_seed(seed)

n = 32
num_epochs = 64
batch_size = int(2**8)
output_dir = str(Path(DUMP_DIR, "small_experiments"))
dataset_len = int(2**17)

models = [SmallTfC(n, attn_fn=a, use_bias=b, loss_fn=l, init_ones=i1) \
    for a in ["relu", "sigmoid", "softplus"] \
    for b in [True] \
    for l in ["margin"] \
    for i1 in [True, False]
] + [
    # SmallTfA(n, 1 + 2*n, loss_fn="bce"),
    SmallTfA(n, 1 + 2*n, loss_fn="margin"),
    # SmallTfB(n, loss_fn="bce"),
    SmallTfB(n, loss_fn="margin"),
]


datasets = [
    SmallTfSuccTokensDataset(n, dataset_len),
    SmallAutoreg1StepTokensDataset(n, dataset_len)
]


total_iters = len(models) * len(datasets)

for (di, dataset) in enumerate(datasets):
    for (mi, model) in enumerate(models):
        print(f"Starting iteration {di*len(models) + mi + 1}/{total_iters}")

        desc_str = f"SMS_{model.desc_str}_{dataset.desc_str}_seed{seed}"

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
            learning_rate = 1e-3,
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



