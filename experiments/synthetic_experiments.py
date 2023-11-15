import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser
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

    num_layers: Optional[int] = field(
        default = None,
        metadata = {"help": "The model's number of transformer layers."}
    )

    num_heads : Optional[int] = field(
        default = None,
        metadata = {"help": "The model's number of attention heads."}
    )


    """ Dataset details """

    num_rules: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of rules to use."}
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

    conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The probability of a variable in the consequent being true."}
    )

    theorem_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The probability of a variable in the theorem being true."}
    )

    state_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The probability of a variable in an initial state being true."}
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
        default = 5,
        metadata = {"help": "How often the Hugging Face Trainer logs"}
    )



def synexp_args_to_wandb_run_name(args: SyntheticExperimentArguments):
    if args.syn_exp_name == "one_shot":
        return f"SynOneShot_nr{args.num_rules}_nv{args.num_vars}" + \
               f"_{args.model_name}_d{args.embed_dim}_L{args.num_layers}_H{args.num_heads}" + \
               f"_ap{args.ante_prob:.2f}_bp{args.conseq_prob}_tp{args.theorem_prob}" + \
               f"_ntr{args.train_len}_ntt{args.eval_len}"

    elif args.syn_exp_name == "next_state":
        return f"SynNextState_nr{args.num_rules}_nv{args.num_vars}" + \
               f"_{args.model_name}_d{args.embed_dim}_L{args.num_layers}_H{args.num_heads}" + \
               f"_ap{args.ante_prob:.2f}_bp{args.conseq_prob}_sp{args.state_prob}" + \
               f"_ntr{args.train_len}_ntt{args.eval_len}"

    elif args.syn_exp_name == "autoreg_ksteps":
        return f"SynARKSteps_nr{args.num_rules}_nv{args.num_vars}_ns{args.num_steps}" + \
               f"_{args.model_name}_d{args.embed_dim}_L{args.num_layers}_H{args.num_heads}" + \
               f"_ap{args.ante_prob:.2f}_bp{args.conseq_prob}_sp{args.state_prob}" + \
               f"_ntr{args.train_len}_ntt{args.eval_len}"

    else:
        raise ValueError(f"Unrecognized syn_exp_name {args.syn_exp_name}")


def make_trainer_for_synthetic(
    args: SyntheticExperimentArguments,
    report_to: str = "wandb"
):
    """ Make a Hugging Face Trainer object """

    if args.syn_exp_name == "one_shot":
        train_dataset = OneShotEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            theorem_prob = args.theorem_prob,
            dataset_len = args.train_len,
            seed = args.seed)

        eval_dataset = OneShotEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            theorem_prob = args.theorem_prob,
            dataset_len = args.eval_len,
            seed = args.seed)

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
            logging_steps = args.logging_steps)

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = one_shot_metrics)


    elif args.syn_exp_name == "next_state":
        train_dataset = NextStateEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            state_prob = args.state_prob,
            dataset_len = args.train_len,
            seed = args.seed)

        eval_dataset = NextStateEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            state_prob = args.state_prob,
            dataset_len = args.eval_len,
            seed = args.seed)

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
            logging_steps = args.logging_steps)

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = next_state_metrics)


    elif args.syn_exp_name == "autoreg_ksteps":
        train_dataset = AutoRegKStepsEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            num_steps = args.num_steps,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            state_prob = args.state_prob,
            dataset_len = args.train_len,
            seed = args.seed)

        eval_dataset = AutoRegKStepsEmbedsDataset(
            num_rules = args.num_rules,
            num_vars = args.num_vars,
            num_steps = args.num_steps,
            ante_prob = args.ante_prob,
            conseq_prob = args.conseq_prob,
            state_prob = args.state_prob,
            dataset_len = args.eval_len,
            seed = args.seed)

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
            logging_steps = args.logging_steps)

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
    synexp_args = parser.parse_args_into_dataclasses()[0]
    trainer = make_trainer_for_synthetic(synexp_args)
    trainer.train()
    wandb.finish()


