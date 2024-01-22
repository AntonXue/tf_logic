import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, HfArgumentParser
from tqdm import tqdm

""" Our imports """
from common import *    # Critical definitions and path inserts

from models import *
from my_datasets import NextStateTokensDataset, AutoregKStepsTokensDataset
from utils.model_loader_utils import load_next_state_model_from_wandb
from utils.metrics import *

""" Parser """

@dataclass
class DistributionShiftExperimentsArguments:
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "distribution_shift_experiments")),
        metadata = {"help": "The output directory of distribution shift experiments."}
    )

    syn_exp_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The experiment to run."}
    )

    """ Model details """

    model_name: Optional[str] = field(
        default = "gpt2",
        metadata = {"help": "The seq2seq model we trained on."}
    )

    embed_dim: Optional[int] = field(
        default = None,
        metadata = {"help": "The model's embedding (i.e., hidden) dimension."}
    )

    num_layers: Optional[int] = field(
        default = None,
        metadata = {"help": "The model's number of transformer layers."}
    )

    num_heads : Optional[int] = field(
        default = None,
        metadata = {"help": "The model's number of attention heads."}
    )

    

    """ Train dataset details """

    train_num_vars: Optional[int] = field(
        default = None,
        metadata = {"help": "(Train) The number of propositional variables to use."}
    )

    train_min_num_rules: Optional[int] = field(
        default = None,
        metadata = {"help": "(Train) The minimum number of rules to use."}
    )

    train_max_num_rules: Optional[int] = field(
        default = None,
        metadata = {"help": "(Train) The maximum number of rules to use."}
    )

    train_num_steps: Optional[int] = field(
        default = 3,
        metadata = {"help": "(Train) The number of steps; used for autoreg_ksteps."}
    )

    train_min_num_states: Optional[int] = field(
        default = None,
        metadata = {"help": "(Train) The minimum number of states to use."}
    )

    train_max_num_states: Optional[int] = field(
        default = None,
        metadata = {"help": "(Train) The maximum number of states to use."}
    )

    train_min_ante_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Train) The minimum probability of a variable in the antecedent being true."}
    )

    train_max_ante_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Train) The maximum probability of a variable in the antecedent being true."}
    )

    train_min_conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Train) The minimum probability of a variable in the consequent being true."}
    )

    train_max_conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Train) The maximum probability of a variable in the consequent being true."}
    )

    train_min_state_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Train) The minimum_probability of a variable in an initial state being true."}
    )

    train_max_state_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Train) The maximum_probability of a variable in an initial state being true."}
    )

    train_min_chain_len: Optional[int] = field(
        default = 2,
        metadata = {"help": "(Train) The minimum attmepted length of the deduction chain for a random dataset."}
    )

    train_max_chain_len: Optional[int] = field(
        default = 5,
        metadata = {"help": "(Train) The maximum attmepted length of the deduction chain for a random dataset."}
    )

    train_len: Optional[int] = field(
        default = None,
        metadata = {"help": "(Train) The number of elements in the dataset."}
    )

    train_eval_len: Optional[int] = field(
        default = None,
        metadata = {"help": "(Train) The number of elements in the eval dataset used during training."}
    )

    """ Eval dataset details """

    eval_num_vars: Optional[int] = field(
        default = None,
        metadata = {"help": "(Eval) The number of propositional variables to use."}
    )

    eval_min_num_rules: Optional[int] = field(
        default = None,
        metadata = {"help": "(Eval) The minimum number of rules to use."}
    )

    eval_max_num_rules: Optional[int] = field(
        default = None,
        metadata = {"help": "(Eval) The maximum number of rules to use."}
    )

    eval_num_steps: Optional[int] = field(
        default = 3,
        metadata = {"help": "(Eval) The number of steps; used for autoreg_ksteps."}
    )

    eval_min_num_states: Optional[int] = field(
        default = None,
        metadata = {"help": "(Eval) The minimum number of states to use."}
    )

    eval_max_num_states: Optional[int] = field(
        default = None,
        metadata = {"help": "(Eval) The maximum number of states to use."}
    )

    eval_min_ante_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Eval) The minimum probability of a variable in the antecedent being true."}
    )

    eval_max_ante_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Eval) The maximum probability of a variable in the antecedent being true."}
    )

    eval_min_conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Eval) The minimum probability of a variable in the consequent being true."}
    )

    eval_max_conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Eval) The maximum probability of a variable in the consequent being true."}
    )

    eval_min_state_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Eval) The minimum_probability of a variable in an initial state being true."}
    )

    eval_max_state_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "(Eval) The maximum_probability of a variable in an initial state being true."}
    )

    eval_min_chain_len: Optional[int] = field(
        default = 2,
        metadata = {"help": "(Eval) The minimum attmepted length of the deduction chain for a random dataset."}
    )

    eval_max_chain_len: Optional[int] = field(
        default = 5,
        metadata = {"help": "(Eval) The maximum attmepted length of the deduction chain for a random dataset."}
    )

    eval_len: Optional[int] = field(
        default = None,
        metadata = {"help": "(Eval) The number of elements in the dataset."}
    )

    """ Other details """

    batch_size: Optional[int] = field(
        default = 8,
        metadata = {"help": "The batch size to use when evaluating stuff."}
    )

    seed: Optional[int] = field(
        default = 1234,
        metadata = {"help": "RNG seed"}
    )

    use_gpu: bool = field(
        default = False,
        metadata = {"help": "Whether or not to use the GPU."}
    )


def dataset_description(
    num_vars: int,
    min_num_rules: int,
    max_num_rules: int,
    min_num_states: int,
    max_num_states: int,
    min_ante_prob: float,
    max_ante_prob: float,
    min_conseq_prob: float,
    max_conseq_prob: float,
    min_state_prob: float,
    max_state_prob: float,
    dataset_len: int
):
    """ String description of dataset useful for saving files """
    return f"nv{num_vars}" + \
        f"_nr{min_num_rules}-{max_num_rules}" + \
        f"_ns{min_num_states}-{max_num_states}" + \
        f"_ap{min_ante_prob:.2f}-{max_ante_prob:.2f}" + \
        f"_bp{min_conseq_prob:.2f}-{max_conseq_prob:.2f}" + \
        f"_sp{max_state_prob:.2f}-{max_state_prob:.2f}" + \
        f"_len{dataset_len}"


def model_description(args):
    return f"{args.model_name}" + \
        f"_d{args.embed_dim}_L{args.num_layers}_H{args.num_heads}"

def train_dataset_description(args):
    return dataset_description(
        num_vars = args.train_num_vars,
        min_num_rules = args.train_min_num_rules,
        max_num_rules = args.train_max_num_rules,
        min_num_states = args.train_min_num_states,
        max_num_states = args.train_max_num_states,
        min_ante_prob = args.train_min_ante_prob,
        max_ante_prob = args.train_max_ante_prob,
        min_conseq_prob = args.train_min_conseq_prob,
        max_conseq_prob = args.train_max_conseq_prob,
        min_state_prob = args.train_min_state_prob,
        max_state_prob = args.train_max_state_prob,
        dataset_len = args.train_len,
    )


def eval_dataset_description(args):
    return dataset_description(
        num_vars = args.eval_num_vars,
        min_num_rules = args.eval_min_num_rules,
        max_num_rules = args.eval_max_num_rules,
        min_num_states = args.eval_min_num_states,
        max_num_states = args.eval_max_num_states,
        min_ante_prob = args.eval_min_ante_prob,
        max_ante_prob = args.eval_max_ante_prob,
        min_conseq_prob = args.eval_min_conseq_prob,
        max_conseq_prob = args.eval_max_conseq_prob,
        min_state_prob = args.eval_min_state_prob,
        max_state_prob = args.eval_max_state_prob,
        dataset_len = args.eval_len,
    )


def compute_saveto_file(args):
    model_desc = model_description(args)
    train_desc = train_dataset_description(args)
    eval_desc = eval_dataset_description(args)
    saveto_file = f"{model_desc}__{train_desc}__{eval_desc}.pt"
    saveto_file = str(Path(args.output_dir, saveto_file))
    return saveto_file


@torch.no_grad()
def run_eval(model, dataset, args):
    model.eval()
    if args.use_gpu:
        model.cuda()

    model_desc = model_description(args)
    train_desc = train_dataset_description(args)
    eval_desc = eval_dataset_description(args)
    saveto_file = compute_saveto_file(args)

    print(f"Model: {model_desc}")
    print(f"Train: {train_desc}")
    print(f"Eval:  {eval_desc}")
    print(f"Will save to:\n  {saveto_file}")

    num_dones, running_hits = 0, 0
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    pbar = tqdm(dataloader)
    for batch in pbar:
        tokens, labels = batch["tokens"], batch["labels"]
        if args.use_gpu:
            tokens, labels = tokens.cuda(), labels.cuda()
        out = model(tokens=tokens, labels=labels)
        preds = (out.logits > 0).long()

        num_dones += tokens.size(0)
        running_hits += (preds == labels).sum()

        acc = running_hits.item() / (num_dones * args.eval_num_vars)
        desc = f"Accuracy {acc:.3f}"
        pbar.set_description(desc)

    stats = {
        "accuracy" : acc
    }

    torch.save(stats, saveto_file)
    return stats


""" Main stuff """

if __name__ == "__main__":
    parser = HfArgumentParser(DistributionShiftExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    if args.syn_exp_name == "next_state":
        model = load_next_state_model_from_wandb(
            model_name = args.model_name,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            num_vars = args.train_num_vars,
            num_rules_range = (args.train_min_num_rules, args.train_max_num_rules),
            num_states_range = (args.train_min_num_states, args.train_max_num_states),
            ante_prob_range = (args.train_min_ante_prob, args.train_max_ante_prob),
            conseq_prob_range = (args.train_min_conseq_prob, args.train_max_conseq_prob),
            state_prob_range = (args.train_min_state_prob, args.train_max_state_prob),
            train_len = args.train_len,
            eval_len = args.train_eval_len,
            task_name="next_state"
        )
        dataset = NextStateTokensDataset(
            num_vars = args.eval_num_vars,
            num_rules_range = (args.eval_min_num_rules, args.eval_max_num_rules),
            num_states_range = (args.eval_min_num_states, args.eval_max_num_states),
            ante_prob_range = (args.eval_min_ante_prob, args.eval_max_ante_prob),
            conseq_prob_range = (args.eval_min_conseq_prob, args.eval_max_conseq_prob),
            state_prob_range = (args.eval_min_state_prob, args.eval_max_state_prob),
            dataset_len = args.eval_len
        )
    elif args.syn_exp_name == "autoreg_ksteps":
        model = load_next_state_model_from_wandb(
            model_name = args.model_name,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            num_vars = args.train_num_vars,
            num_rules_range = (args.train_min_num_rules, args.train_max_num_rules),
            ante_prob_range = (args.train_min_ante_prob, args.train_max_ante_prob),
            conseq_prob_range = (args.train_min_conseq_prob, args.train_max_conseq_prob),
            train_len = args.train_len,
            eval_len = args.train_eval_len,
            task_name="autoreg_ksteps",
            num_steps = args.train_num_steps,
            chain_len_range=(args.train_min_chain_len, args.train_max_chain_len)
        )
        dataset = AutoregKStepsTokensDataset(
            num_vars = args.eval_num_vars,
            num_rules_range = (args.eval_min_num_rules, args.eval_max_num_rules),
            ante_prob_range = (args.eval_min_ante_prob, args.eval_max_ante_prob),
            conseq_prob_range = (args.eval_min_conseq_prob, args.eval_max_conseq_prob),
            chain_len_range = (args.eval_min_chain_len, args.eval_max_chain_len),
            num_steps = args.eval_num_steps,
            dataset_len = args.eval_len)

    run_eval(model, dataset, args)


