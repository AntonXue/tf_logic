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

    attack_tokens_style: Optional[str] = field(
        default = "repeat",
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
            num_attack_tokens = args.num_attack_tokens,
            attack_tokens_style = args.attack_tokens_style,
            train_len = args.train_len,
            eval_len = args.eval_len,
            batch_size = args.batch_size,
            num_epochs = args.num_epochs,
            learning_rate = args.learning_rate,
            reasoner_seed = args.reasoner_seed,
            attacker_seed = args.attacker_seed,
            reasoner_type = args.reasoner_type,
            output_dir = args.output_dir
        )

        run_learned_suppress_rule(config)

    else:
        raise ValueError(f"Unknown attack name {args.attack_name}")
    



