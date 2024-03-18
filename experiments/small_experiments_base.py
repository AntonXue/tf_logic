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

seed = 616
torch.manual_seed(seed)

n = 16
num_epochs = 64
batch_size = int(2**9)
output_dir = str(Path(DUMP_DIR, "small_experiments"))
dataset_len = int(2**16)

models = [SmallTfC(n, attn_fn=a, loss_fn=l, init_value=iv) \
    for l in ["margin", "regcat"] \
    for a in ["sigmoid"] \
    for iv in [0, None]
] \
    + [SmallTfA(n, 1 + 2*n, loss_fn=l) for l in ["margin", "regcat"]] \
    + [SmallTfB(n, loss_fn=l) for l in ["margin", "regcat"]] \
    + []

datasets = [
    SmallTfSuccTokensDataset(n, dataset_len),
    SmallAutoreg1StepTokensDataset(n, dataset_len)
]


total_iters = len(models) * len(datasets)

for (di, dataset) in enumerate(datasets):
    for (mi, model) in enumerate(models):
        torch.manual_seed(seed)
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
            logging_steps = 32,
            learning_rate = 1e-2,
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



