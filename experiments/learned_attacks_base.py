import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser
import numpy as np
import wandb
import json

""" Our imports """
from common import *    # Definitions and path inserts, particularly for WANDB
from models import *
from my_datasets import *
from utils.model_loader_utils import *

""" Parser for Hugging Face """

@dataclass
class LearnedAttackExperimentsArguments:
    """ This class captures ALL the possible synthetic experiments we may want to run """
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "learned_attacks")),
        metadata = {"help": "Output directory of synthetic experiments."}
    )

    """ Model details """

    num_vars: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of propositional variables to use."}
    )

    embed_dim: Optional[int] = field(
        default = None,
        metadata = {"help": "The reasoner model's embedding (i.e., hidden) dimension."}
    )

    num_attack_tokens: Optional[int] = field(
        default = None,
        metadata = {"help": "The reasoner model's embedding (i.e., hidden) dimension."}
    )

    token_range: Optional[str] = field(
        default = None,
        metadata = {"help": "Unbounded or clamped to (0,1). We also binarize accordingly."}
    )

    """ Attack training details """

    train_len: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of elements in the training dataset."}
    )

    eval_len: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of elements in the eval (i.e., validation) dataset."}
    )

    num_epochs: Optional[int] = field(
        default = 16,
        metadata = {"help": "The number of epochs for training."}
    )

    learning_rate: Optional[float] = field(
        default = 1e-4,
        metadata = {"help": "Learning rate."}
    )

    batch_size: Optional[int] = field(
        default = 64,
        metadata = {"help": "The train batch size."}
    )

    auto_find_batch_size: Optional[bool] = field(
        default = False,
        metadata = {"help": "Automatically scale batch size if it encounters out-of-memory."}
    )

    reasoner_seed: Optional[int] = field(
        default = None,
        metadata = {"help": "The seed that the reasoner model was trained with"}
    )

    attacker_seed: Optional[int] = field(
        default = 1234,
        metadata = {"help": "RNG seed"}
    )

    logging_steps: int = field(
        default = 16,
        metadata = {"help": "How often the HF's Trainer logs."}
    )


def args_to_wandb_run_name(args):
    res_model_str = f"gpt2_d{args.embed_dim}_nv{args.num_vars}_rseed{args.reasoner_seed}"
    return f"SynLAtk_{args.token_range}_natk{args.num_attack_tokens}" + \
        f"_{res_model_str}_" + \
        f"_ntr{args.train_len}_ntt{args.eval_len}" + \
        f"_bsz{args.batch_size}_lr{args.learning_rate:.5f}" + \
        f"_aseed{args.attacker_seed}"


def load_reasoner_model_and_dataset(args):
    return load_model_and_dataset_from_big_grid(
        num_vars = args.num_vars,
        embed_dim = args.embed_dim,
        num_steps = 1,
        seed = args.reasoner_seed,
    )

def learned_attack_metrics(eval_preds):
    logits, labels = eval_preds
    logits = logits[0] if isinstance(logits, tuple) else logits
    num_vars = labels.shape[-1]
    res_logits, bin_res_logits = logits[:,0], logits[:,1]
    res_pred = (res_logits > 0).astype(np.int64)
    bin_res_pred = (bin_res_logits > 1 - 1e-5).astype(np.int64)
    acc = np.mean((res_pred == labels).sum(axis=-1) == num_vars)
    bin_acc = np.mean((bin_res_pred == labels).sum(axis=-1) == num_vars)
    return {
        "Acc": acc,
        "BinAcc": bin_acc,
        "AvgOnes": np.mean(res_pred),
    }


def make_trainer(args):
    reasoner_model, reasoner_dataset = load_reasoner_model_and_dataset(args)

    atk_wrap_model = AttackWrapperModel(
        reasoner_model = reasoner_model,
        num_attack_tokens = args.num_attack_tokens,
        token_range = args.token_range
    )

    train_dataset = AttackWrapperDataset(
        reasoner_dataset = reasoner_dataset,
        num_attack_tokens = args.num_attack_tokens,
        dataset_len = args.train_len
    )

    eval_dataset = AttackWrapperDataset(
        reasoner_dataset = reasoner_dataset,
        num_attack_tokens = args.num_attack_tokens,
        dataset_len = args.eval_len
    )

    run_name = args_to_wandb_run_name(args)
    training_args = TrainingArguments(
        str(Path(args.output_dir, run_name)),
        num_train_epochs = args.num_epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        auto_find_batch_size = False,
        evaluation_strategy = "epoch",
        report_to = "wandb",
        run_name = run_name,
        logging_steps = args.logging_steps,
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        save_strategy = "epoch",
        save_total_limit = 2
    )

    return Trainer(
        atk_wrap_model,
        training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics = learned_attack_metrics
    )


if __name__ == "__main__":
    parser = HfArgumentParser(LearnedAttackExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.attacker_seed)
    trainer = make_trainer(args)
    
    trainer.train()
    wandb.finish()



