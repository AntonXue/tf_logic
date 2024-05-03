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
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--num_vars", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--attn_fn", type=str, default="softmax")
    parser.add_argument("--dataset", type=str, default="succ")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2**7)
    parser.add_argument("--dataset_len", type=int, default=2**18)
    parser.add_argument("--output_dir", default=str(Path(DUMP_DIR, "small_experiments")))
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--num_tries", type=int, default=1)
    parser.add_argument("--loss_fn", type=str, default="bce")
    parser.add_argument("--attn_loss_scale", type=float, default=0.0)
    parser.add_argument("--ap_min", type=float, default=0.5)
    parser.add_argument("--ap_max", type=float, default=0.5)
    parser.add_argument("--bp_min", type=float, default=0.5)
    parser.add_argument("--bp_max", type=float, default=0.5)
    args = parser.parse_args()
    return args

args = parse_args()


if args.model == "tfa":
    todos = [
        (SmallTfA(args.num_vars, attn_fn=args.attn_fn, loss_fn=args.loss_fn),
         SmallTfSuccTokensDataset(
            args.num_vars,
            args.dataset_len,
            ante_prob_range = (args.ap_min, args.ap_max),
            conseq_prob_range = (args.bp_min, args.bp_max)
         ),
         args.seed+i
        )
        for i in range(args.num_tries)
    ]
elif args.model == "tfb":
    todos = [
        (SmallTfB(args.num_vars, args.embed_dim, attn_fn=args.attn_fn, loss_fn=args.loss_fn),
         SmallTfSuccTokensDataset(
            args.num_vars,
            args.dataset_len,
            ante_prob_range = (args.ap_min, args.ap_max),
            conseq_prob_range = (args.bp_min, args.bp_max)
         ),
         args.seed+i
        )
        for i in range(args.num_tries)
    ]
elif args.model == "gpt2" and args.dataset == "succ":
    todos = [
        (SmallGPT2(
            args.num_vars,
            args.embed_dim,
            loss_fn = args.loss_fn,
            attn_loss_scale = args.attn_loss_scale
        ),
         SmallTfSuccTokensDataset(
            args.num_vars,
            args.dataset_len,
            ante_prob_range = (args.ap_min, args.ap_max),
            conseq_prob_range = (args.bp_min, args.bp_max)
         ),
         args.seed+i
        )
        for i in range(args.num_tries)
    ]
elif args.model == "gpt2" and args.dataset == "autoreg1":
    todos = [
        (SmallGPT2(
            args.num_vars,
            args.embed_dim,
            loss_fn = args.loss_fn,
            attn_loss_scale = args.attn_loss_scale
        ),
         SmallAutoreg1StepTokensDataset(
            args.num_vars,
            args.dataset_len,
            ante_prob_range = (args.ap_min, args.ap_max),
            conseq_prob_range = (args.bp_min, args.bp_max)
         ),
         args.seed+i
        )
        for i in range(args.num_tries)
    ]
else:
    raise ValueError(f"Unknown model_name {args.model_name}")


for i, (model, dataset, seed) in enumerate(todos):
    torch.manual_seed(seed)
    print(f"\n>>> Starting {i+1}/{len(todos)}\n")

    desc_str = f"SMS_{model.desc_str}_{dataset.desc_str}_lr{args.lr:.5f}_seed{seed}"

    training_args = TrainingArguments(
        str(Path(args.output_dir, desc_str)),
        num_train_epochs = args.num_epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = 4 * args.batch_size,
        auto_find_batch_size = False,
        evaluation_strategy = "epoch",
        report_to = "wandb",
        run_name = desc_str,
        logging_steps = 128,
        learning_rate = args.lr,
        warmup_ratio = 0.10,
        save_strategy = "epoch",
        save_total_limit = 2
    )

    succ_eval_datasets = {
        f"succ_p{p:.2f}-{(p+0.1):.2f}":
        SmallTfSuccTokensDataset(
            args.num_vars,
            args.dataset_len//4,
            ante_prob_range = (p, p+0.1),
            conseq_prob_range = (p, p+0.1)
        )
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }

    eval_datasets = {"training_dataset": dataset} | succ_eval_datasets

    trainer = Trainer(
        model,
        training_args,
        train_dataset = dataset,
        eval_dataset = eval_datasets,
        compute_metrics = small_succ_metrics
    )

    trainer.train()
    wandb.finish()



