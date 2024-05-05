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

from learned_attacks_stuff import *

""" Parser for Hugging Face """

@dataclass
class LearnedAttackExperimentsArguments:
    """ This class captures ALL the possible synthetic experiments we may want to run """
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "learned_attacks")),
        metadata = {"help": "Output directory of synthetic experiments."}
    )

    attack_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The name of the attack."}
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

    device: str = field(
        default = "cuda",
        metadata = {"help": "The device we run things on."}
    )

    reasoner_type: str = field(
        default = "learned",
        metadata = {"help": "The type (learned/theory) of reasoner that we use."}
    )


def args_to_wandb_run_name(args):
    if args.attack_name == "coerce_state":
        res_model_str = f"gpt2_d{args.embed_dim}_nv{args.num_vars}_rseed{args.reasoner_seed}"
        return f"SynLAtkCS_{args.token_range}_natk{args.num_attack_tokens}" + \
            f"_{res_model_str}_" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}" + \
            f"_bsz{args.batch_size}_lr{args.learning_rate:.5f}" + \
            f"_aseed{args.attacker_seed}"

    elif args.attack_name == "suppress_rule":
        res_model_str = f"gpt2_d{args.embed_dim}_nv{args.num_vars}_rseed{args.reasoner_seed}"
        return f"SynLAtkSR" + \
            f"_{res_model_str}_" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}" + \
            f"_bsz{args.batch_size}_lr{args.learning_rate:.5f}" + \
            f"_aseed{args.attacker_seed}"

    else:
        raise ValueError(f"Unknown attack name {args.attack_name}")


def load_reasoner_model_and_dataset(args):
    return load_model_and_dataset_from_big_grid(
        num_vars = args.num_vars,
        embed_dim = args.embed_dim,
        seed = args.reasoner_seed,
        dataset_len = args.train_len,
    )


def make_coerce_state_trainer(args):

    def coerce_state_metrics(eval_preds):
        logits, labels = eval_preds
        logits = logits[0] if isinstance(logits, tuple) else logits
        assert logits.shape[1] >= 3 # Should be at least three terms

        num_vars = labels.shape[-1]

        # Non-binary stuff
        res_logits = logits[:,-2]
        res_pred = (res_logits > 0).astype(np.int64)
        elems_acc = np.mean(res_pred == labels)
        state_acc = np.mean((res_pred == labels).sum(axis=-1) == num_vars)

        # Binary stuff
        bin_res_logits = logits[:,-1]
        bin_res_pred = (bin_res_logits > 0).astype(np.int64)
        bin_elems_acc = np.mean(bin_res_pred == labels)
        bin_state_acc = np.mean((bin_res_pred == labels).sum(axis=-1) == num_vars)

        # Track the alignment of the attack logits (averaged scheme)
        atk_logits = logits[:,:-2]
        atk_pred = (atk_logits > 0).astype(np.int64)
        atk_align = np.mean(atk_pred == labels.reshape(-1,1,num_vars))
        atk_size = np.mean(np.absolute(atk_logits))

        return {
            "ElemsAcc": elems_acc,
            "StateAcc": state_acc,
            "AvgOnes": np.mean(res_pred),
            "BinElemsAcc": bin_elems_acc,
            "BinStateAcc": bin_state_acc,
            "BinAvgOnes": np.mean(bin_res_pred),
            "AtkAlign": atk_align,
            "AtkSize": atk_size
        }

    reasoner_model, reasoner_dataset = load_reasoner_model_and_dataset(args)

    atk_wrap_model = CoerceStateWrapperModel(
        reasoner_model = reasoner_model,
        num_attack_tokens = args.num_attack_tokens,
        token_range = args.token_range
    )

    train_dataset = CoerceStateDataset(
        reasoner_dataset = reasoner_dataset,
        num_attack_tokens = args.num_attack_tokens,
        dataset_len = args.train_len
    )

    eval_dataset = CoerceStateDataset(
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
        compute_metrics = coerce_state_metrics
    )


def make_suppress_rule_trainer(args):

    def suppress_rule_metrics(eval_preds):
        logits, labels = eval_preds
        logits = logits[0] if isinstance(logits, tuple) else logits
        assert logits.shape[1] >= 5
        num_vars = labels.shape[-1]

        supp_ante, supp_conseq = logits[:,0], logits[:,1]
        atk_ante, atk_conseq = logits[:,2], logits[:,3]
        res_logits = logits[:,4:]

        ante_align = np.mean((atk_ante > 0).astype(np.int64) == supp_ante)
        conseq_align = np.mean((atk_conseq < 0).astype(np.int64) == supp_conseq)

        pred = (res_logits > 0).astype(np.int64)
        _, num_steps, num_vars = logits.shape

        elems_acc = np.mean(pred == labels)
        state_acc = np.mean(np.sum(pred == labels, axis=(-1,-2)) == num_steps * num_vars)
        return {
            "ElemsAcc": elems_acc,
            "StateAcc": state_acc,
            "AnteAlign": ante_align,
            "ConseqAlign": conseq_align,
        }

    reasoner_model, reasoner_dataset = load_reasoner_model_and_dataset(args)

    atk_wrap_model = SuppressRuleWrapperModel(
        reasoner_model = reasoner_model,
    )

    train_dataset = SuppressRuleDataset(
        reasoner_dataset = reasoner_dataset,
        dataset_len = args.train_len
    )

    eval_dataset = SuppressRuleDataset(
        reasoner_dataset = reasoner_dataset,
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
        compute_metrics = suppress_rule_metrics
    )



if __name__ == "__main__":
    parser = HfArgumentParser(LearnedAttackExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.attacker_seed)

    if args.attack_name == "coerce_state":
        trainer = make_coerce_state_trainer(args)

    elif args.attack_name == "suppress_rule":
        # trainer = make_suppress_rule_trainer(args)
        config = LearnedSuppressRuleConfig(
            num_vars = args.num_vars,
            embed_dim = args.embed_dim,
            train_len = args.train_len,
            eval_len = args.eval_len,
            batch_size = args.batch_size,
            num_epochs = args.num_epochs,
            learning_rate = args.learning_rate,
            reasoner_seed = args.reasoner_seed,
            attacker_seed = args.attacker_seed,
            reasoner_type = args.reasoner_type,
        )

        run_learned_suppress_rule(config)

    else:
        raise ValueError(f"Unknown attack name {args.attack_name}")
    



