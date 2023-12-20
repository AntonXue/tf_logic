import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser, AutoTokenizer, DataCollatorWithPadding
import wandb

""" Our imports """
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from models import AutoTFLModel
from my_datasets import *
from experiments_init import *
from evaluation_utils import *

""" Parser for Hugging Face """

@dataclass
class SyntheticExperimentArguments:
    """ This class captures ALL the possible synthetic experiments we may want to run """
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "synthetic_experiments")),
        metadata = {"help": "Output directory of synthetic experiments"}
    )

    syn_exp_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The experiment to run"}
    )

    """ Model details """

    model_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The seq2seq model to use"}
    )

    embed_dim: Optional[int] = field(
        default = None,
        metadata = {"help": "The model's embedding (i.e., hidden) dimension."}
    )

    ffwd_dim: Optional[int] = field(
        default = None,
        metadata = {"help": "The transformer's MLP dimension."}
    )

    num_layers: Optional[int] = field(
        default = None,
        metadata = {"help": "The model's number of transformer layers."}
    )

    num_heads : Optional[int] = field(
        default = None,
        metadata = {"help": "The model's number of attention heads."}
    )

    use_pretrained : Optional[bool] = field(
        default = False,
        metadata = {"help": "Weights from the pretrained model are loaded if True. Note that this restricts changes to the model (default num_heads, num_layers, etc. will be used)"}
    )

    """ Dataset details """

    num_rules: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of rules to use."}
    )

    min_num_rules: Optional[int] = field(
        default = None,
        metadata = {"help": "The minimum number of rules to use."}
    )

    max_num_rules: Optional[int] = field(
        default = None,
        metadata = {"help": "The maximum number of rules to use."}
    )

    num_vars: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of propositional variables to use."}
    )

    num_steps: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of steps; used for autoreg_ksteps."}
    )

    ante_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The probability of a variable in the antecedent being true."}
    )

    min_ante_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The minimum probability of a variable in the antecedent being true."}
    )

    max_ante_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The maximum probability of a variable in the antecedent being true."}
    )

    conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The probability of a variable in the consequent being true."}
    )

    min_conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The minimum probability of a variable in the consequent being true."}
    )

    max_conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The maximum probability of a variable in the consequent being true."}
    )

    state_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The probability of a variable in an initial state being true."}
    )

    min_state_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The minimum_probability of a variable in an initial state being true."}
    )

    max_state_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The maximum_probability of a variable in an initial state being true."}
    )

    chain_len: Optional[int] = field(
        default = 3,
        metadata = {"help": "The attmeped length of the deduction chain for a random dataset."}
    )

    min_num_rules: Optional[int] = field(
        default = 5,
        metadata = {"help": "The minimum number of rules to randomly generate."}
    )

    max_num_rules: Optional[int] = field(
        default = 10,
        metadata = {"help": "The maximum number of rules to randomly generate."}
    )

    min_num_states: Optional[int] = field(
        default = 5,
        metadata = {"help": "The minimum number of states to randomly generate."}
    )

    max_num_states: Optional[int] = field(
        default = 10,
        metadata = {"help": "The maximum number of states to randomly generate."}
    )

    train_len: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of elements in the training dataset."}
    )

    eval_len: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of elements in the eval (i.e., validation) dataset."}
    )

    """ Training details """

    train_batch_size: Optional[int] = field(
        default = 8,
        metadata = {"help": "The train batch size."}
    )

    eval_batch_size: Optional[int] = field(
        default = 8,
        metadata = {"help": "The eval (i.e., validation) batch size."}
    )

    auto_find_batch_size: Optional[bool] = field(
        default = True,
        metadata = {"help": "Automatically scale batch size if it encounters out-of-memory."}
    )

    num_epochs: Optional[int] = field(
        default = 100,
        metadata = {"help": "The number of epochs for training."}
    )

    seed: Optional[int] = field(
        default = 1234,
        metadata = {"help": "RNG seed"}
    )

    logging_steps: int = field(
        default = 10,
        metadata = {"help": "How often the HF's Trainer logs."}
    )


def synexp_args_to_wandb_run_name(args: SyntheticExperimentArguments):
    model_str = f"{args.model_name}" + \
                (f"_pt" if args.use_pretrained else "") + \
                (f"_d{args.embed_dim}" if args.embed_dim is not None else "") + \
                (f"_fd{args.ffwd_dim}" if args.ffwd_dim is not None else "") + \
                (f"_L{args.num_layers}" if args.num_layers is not None else "") + \
                (f"_H{args.num_heads}" if args.num_heads is not None else "")

    if args.syn_exp_name == "one_shot":
        return f"SynOS_{model_str}__" + \
            f"nv{args.num_vars}_nr{args.num_rules}" + \
            f"_ap{args.ante_prob:.2f}_bp{args.conseq_prob:.2f}_tp{args.theorem_prob:.2f}" + \
            f"_cl{args.chain_len}_ntr{args.train_len}_ntt{args.eval_len}"

    elif args.syn_exp_name == "one_shot_str":
        return f"SynOSstr__{model_str}__" + \
            f"nv{args.num_vars}_nr{args.num_rules}" + \
            f"_ap{args.ante_prob:.2f}_bp{args.conseq_prob:.2f}_tp{args.theorem_prob:.2f}" + \
            f"_cl{args.chain_len}_ntr{args.train_len}_ntt{args.eval_len}"

    elif args.syn_exp_name == "next_state":
        return f"SynNS_{model_str}__" + \
            f"nv{args.num_vars}_nr{args.num_rules}" + \
            f"_ap{args.ante_prob:.2f}_bp{args.conseq_prob:.2f}_sp{args.state_prob:.2f}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}"

    elif args.syn_exp_name == "next_state_from_tokens":
        return f"SynNSFT_{model_str}__" + \
            f"nv{args.num_vars}" + \
            f"_nr{args.min_num_rules}-{args.max_num_rules}" + \
            f"_ns{args.min_num_states}-{args.max_num_states}" + \
            f"_ap{args.min_ante_prob:.2f}-{args.max_ante_prob:.2f}" + \
            f"_bp{args.min_conseq_prob:.2f}-{args.max_conseq_prob:.2f}" + \
            f"_sp{args.max_state_prob:.2f}-{args.max_state_prob:.2f}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}"

    elif args.syn_exp_name == "autoreg_ksteps":
        return f"SynAR_{model_str}__" + \
            f"nv{args.num_vars}_nr{args.num_rules}" + \
            f"_ap{args.ante_prob:.2f}_bp{args.conseq_prob:.2f}_sp{args.state_prob:.2f}" + \
            f"_cl{args.chain_len}_ntr{args.train_len}_ntt{args.eval_len}"
    else:
        raise ValueError(f"Unrecognized syn_exp_name {args.syn_exp_name}")


def trainer_stats_for_wandb(
    args: SyntheticExperimentArguments,
    trainer: Trainer
):
    """ Make some statistics to report to wandb """
    if args.syn_exp_name == "one_shot":
        num_train_qeds = torch.tensor(0)
        for i in range(len(trainer.train_dataset)):
            num_train_qeds += trainer.train_dataset[i]["labels"]

        num_eval_qeds = torch.tensor(0)
        for i in range(len(trainer.eval_dataset)):
            num_eval_qeds += trainer.eval_dataset[i]["labels"]

        return {
            "train_len": len(trainer.train_dataset),
            "train_qeds": num_train_qeds.item(),
            "eval_len" : len(trainer.eval_dataset),
            "eval_qeds": num_eval_qeds.item()
        }

    elif args.syn_exp_name == "next_state":
        num_train_succ_diffs = torch.tensor(0)
        for i in range(len(trainer.train_dataset)):
            item = trainer.train_dataset[i]
            num_train_succ_diffs += (item["state"] - item["labels"]).sum().abs() > 0

        num_eval_succ_diffs = torch.tensor(0)
        for item in range(len(trainer.eval_dataset)):
            item = trainer.eval_dataset[i]
            num_eval_succ_diffs += (item["state"] - item["labels"]).sum().abs() > 0

        return {
            "train_len": len(trainer.train_dataset),
            "train_succ_diffs": num_train_succ_diffs.item(),
            "eval_len": len(trainer.eval_dataset),
            "eval_succ_diffs": num_eval_succ_diffs.item()
        }

    elif args.syn_exp_name == "next_state_from_tokens":
        return {
            "train_len": len(trainer.train_dataset),
            "eval_len": len(trainer.eval_dataset),
        }

    elif args.syn_exp_name == "autoreg_ksteps":
        num_train_kstep_diffs = torch.tensor(0)
        for i in range(len(trainer.train_dataset)):
            item = trainer.train_dataset[i]
            num_train_kstep_diffs += (item["state"] - item["labels"][-1]).sum().abs() > 0

        num_eval_kstep_diffs = torch.tensor(0)
        for item in range(len(trainer.eval_dataset)):
            item = trainer.eval_dataset[i]
            num_eval_kstep_diffs += (item["state"] - item["labels"][-1]).sum().abs() > 0

        return {
            "train_len": len(trainer.train_dataset),
            "train_kstep_diffs": num_train_kstep_diffs.item(),
            "eval_len": len(trainer.eval_dataset),
            "eval_kstep_diffs": num_eval_kstep_diffs.item()
        }

    else:
        raise ValueError(f"Unrecognized syn_exp_name {args.syn_exp_name}")


def make_trainer_for_synthetic(
    args: SyntheticExperimentArguments,
    report_to: str = "wandb"
):
    """ Make a Hugging Face Trainer object """

    if args.syn_exp_name == "one_shot":
        big_dataset = OneShotEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            chain_len = args.chain_len,
            dataset_len = args.train_len + args.eval_len)

        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [args.train_len, args.eval_len])

        tfl_model = AutoTFLModel.from_kwargs(
            task_name = "one_shot",
            num_vars = args.num_vars,
            model_name = args.model_name,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads)

        training_args = TrainingArguments(
            args.output_dir,
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = synexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.20,
            save_strategy = "no")

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = one_shot_metrics)

    elif args.syn_exp_name == "one_shot_str":
        # Get the tokenizer to create the dataset
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        big_dataset = OneShotStringDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            theorem_prob = args.theorem_prob,
            dataset_len = args.train_len + args.eval_len,
            tokenizer = tokenizer,
            padding = "longest" if args.use_pretrained else "max_length")

        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [args.train_len, args.eval_len])

        if args.use_pretrained:
            tfl_model = AutoTFLModel.from_pretrained(
                task_name = "one_shot_str",
                model_name = args.model_name)
            tfl_model.config.pad_token_id = tokenizer.pad_token_id
        else:
            tfl_model = AutoTFLModel.from_kwargs(
                task_name = "one_shot_str",
                num_vars = args.num_vars,
                model_name = args.model_name,
                embed_dim = args.embed_dim,
                num_layers = args.num_layers,
                num_heads = args.num_heads)
            tfl_model.seqcls_model.model.config.pad_token_id = tokenizer.pad_token_id

        training_args = TrainingArguments(
            args.output_dir,
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = synexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.20,
            save_strategy = "no")

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator,
            compute_metrics = one_shot_metrics)


    elif args.syn_exp_name == "next_state":
        big_dataset = NextStateEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            state_prob = args.state_prob,
            dataset_len = args.train_len + args.eval_len)

        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [args.train_len, args.eval_len])

        tfl_model = AutoTFLModel.from_kwargs(
            task_name = "next_state",
            num_vars = args.num_vars,
            model_name = args.model_name,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads)

        training_args = TrainingArguments(
            args.output_dir,
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = synexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.20,
            save_strategy = "no")

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = next_state_metrics)


    elif args.syn_exp_name == "next_state_from_tokens":
        big_dataset = NextStateFromTokensEmbedsDataset(
            num_vars = args.num_vars,
            num_rules_range = (args.min_num_rules, args.max_num_rules),
            num_states_range = (args.min_num_states, args.max_num_states),
            ante_prob_range = (args.min_ante_prob, args.max_ante_prob),
            conseq_prob_range = (args.min_conseq_prob, args.max_conseq_prob),
            state_prob_range = (args.min_state_prob, args.max_state_prob),
            dataset_len = args.train_len + args.eval_len)

        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [args.train_len, args.eval_len])

        tfl_model = AutoTFLModel.from_kwargs(
            task_name = "next_state_from_tokens",
            num_vars = args.num_vars,
            model_name = args.model_name,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads)

        training_args = TrainingArguments(
            args.output_dir,
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = synexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.20,
            save_strategy = "no")

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = next_state_metrics)


    elif args.syn_exp_name == "autoreg_ksteps":
        big_dataset = AutoRegKStepsEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            num_steps = args.num_steps,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            state_prob = args.state_prob,
            dataset_len = args.train_len + args.eval_len)

        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [args.train_len, args.eval_len])

        tfl_model = AutoTFLModel.from_kwargs(
            task_name = "autoreg_ksteps",
            num_vars = args.num_vars,
            num_steps = args.num_steps,
            model_name = args.model_name,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads)

        training_args = TrainingArguments(
            args.output_dir,
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = synexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.20,
            save_strategy = "no")

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = autoreg_ksteps_metrics)


    else:
        raise ValueError(f"Unrecognized exp_name {args.syn_exp_name}")


""" Main stuff """

if __name__ == "__main__":
    parser = HfArgumentParser(SyntheticExperimentArguments)
    args = parser.parse_args_into_dataclasses()[0]

    torch.manual_seed(args.seed)

    trainer = make_trainer_for_synthetic(args)

    # Log some preliminary training stats
    trainer_stats = trainer_stats_for_wandb(args, trainer)
    trainer.train()

    wandb.run.summary["trainer_stats"] = trainer_stats
    wandb.finish()
