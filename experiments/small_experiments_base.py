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

seed = 1411
torch.manual_seed(seed)
output_dir = str(Path(DUMP_DIR, "small_experiments"))

num_epochs = 16
batch_size = int(2**7)
dataset_len = int(2**18)
lr = 1e-3


"""
model_datasets = [
    (SmallTfE(n, 2*n, use_attn_bias=ab, use_learned_embed=le, init_value=iv),
     SmallTfSuccTokensDataset(n, dataset_len)) \
        for n in [8, 16, 24, 32, 40, 48, 56, 64] \
        for le in [True, False] \
        for ab in [True, False] \
        for iv in [None]
]
"""

"""
model_datasets = [
    (SmallTfE(n, 2*n + 1, use_attn_bias=ab, use_learned_embed=le, init_value=iv),
     SmallTfSuccTokensDataset(n, dataset_len)) \
        for n in [8, 16, 24, 32, 40, 48, 56, 64] \
        for le in [True, False] \
        for ab in [True, False] \
        for iv in [None]
]

"""
model_datasets = [
    (SmallTfE(n, 2*n - 1, use_attn_bias=ab, use_learned_embed=le, init_value=iv),
     SmallTfSuccTokensDataset(n, dataset_len)) \
        for n in [8, 16, 24, 32, 40, 48, 56, 64] \
        for le in [True] \
        for ab in [True, False] \
        for iv in [None]
]


for i, (model, dataset) in enumerate(model_datasets):
    torch.manual_seed(seed)
    print(f"\n>>> Starting {i+1}/{len(model_datasets)}\n")

    desc_str = f"SMS_{model.desc_str}_{dataset.desc_str}_lr{lr:.5f}_seed{seed}"

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
        learning_rate = lr,
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



