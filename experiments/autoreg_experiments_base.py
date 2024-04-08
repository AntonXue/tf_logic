import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser
import wandb
import json

""" Our imports """
from common import *    # Definitions and path inserts, particularly for WANDB
from models import *
from my_datasets import *
from utils.metrics import *
from utils.model_loader_utils import load_checkpoint_from_wandb

""" Parser for Hugging Face """

@dataclass
class AutoregExperimentsArguments:
    """ This class captures ALL the possible synthetic experiments we may want to run """
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "autoreg_experiments")),
        metadata = {"help": "Output directory of synthetic experiments."}
    )

    syn_exp_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The experiment to run."}
    )

    """ Model details """

    model_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The seq2seq model to use."}
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

    num_heads: Optional[int] = field(
        default = None,
        metadata = {"help": "The model's number of attention heads."}
    )

    use_pretrained : Optional[bool] = field(
        default = False,
        metadata = {"help": "Weights from the pretrained model are loaded if True. " + \
                    "Note that this restricts changes to the model " + \
                    "(default num_heads, num_layers, etc. will be used)."}
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
        default = 3,
        metadata = {"help": "The number of steps; used for autoreg_ksteps."}
    )

    ante_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The probability of a variable in the antecedent being true."}
    )

    min_ante_prob: Optional[float] = field(
        default = 0.25,
        metadata = {"help": "The minimum probability of a variable in the antecedent being true."}
    )

    max_ante_prob: Optional[float] = field(
        default = 0.25,
        metadata = {"help": "The maximum probability of a variable in the antecedent being true."}
    )

    conseq_prob: Optional[float] = field(
        default = None,
        metadata = {"help": "The probability of a variable in the consequent being true."}
    )

    min_conseq_prob: Optional[float] = field(
        default = 0.25,
        metadata = {"help": "The minimum probability of a variable in the consequent being true."}
    )

    max_conseq_prob: Optional[float] = field(
        default = 0.25,
        metadata = {"help": "The maximum probability of a variable in the consequent being true."}
    )

    chain_len: Optional[int] = field(
        default = 3,
        metadata = {"help": "The attmepted length of the deduction chain for a random dataset."}
    )

    min_chain_len: Optional[int] = field(
        default = 2,
        metadata = {"help": "The minimum attmepted length of the deduction chain for a random dataset."}
    )

    max_chain_len: Optional[int] = field(
        default = 5,
        metadata = {"help": "The maximum attmepted length of the deduction chain for a random dataset."}
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

    num_epochs: Optional[int] = field(
        default = 16,
        metadata = {"help": "The number of epochs for training."}
    )

    seed: Optional[int] = field(
        default = 1234,
        metadata = {"help": "RNG seed"}
    )

    logging_steps: int = field(
        default = 64,
        metadata = {"help": "How often the HF's Trainer logs."}
    )


def synexp_args_to_wandb_run_name(args: AutoregExperimentsArguments):
    model_str = f"{args.model_name}" + \
                (f"_pt" if args.use_pretrained else "") + \
                (f"_d{args.embed_dim}" if args.embed_dim is not None else "") + \
                (f"_fd{args.ffwd_dim}" if args.ffwd_dim is not None else "") + \
                (f"_L{args.num_layers}" if args.num_layers is not None else "") + \
                (f"_H{args.num_heads}" if args.num_heads is not None else "")

    if args.syn_exp_name == "autoreg_ksteps":
        return f"SynSAR_{model_str}_" + \
            f"_nv{args.num_vars}" + \
            f"_ns{args.num_steps}" + \
            f"_nr{args.min_num_rules}-{args.max_num_rules}" + \
            f"_ap{args.min_ante_prob:.2f}-{args.max_ante_prob:.2f}" + \
            f"_bp{args.min_conseq_prob:.2f}-{args.max_conseq_prob:.2f}" + \
            f"_cl{args.min_chain_len}-{args.max_chain_len}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}_bsz{args.batch_size}" + \
            f"_lr{args.learning_rate:.5f}" + \
            f"_seed{args.seed}"

    else:
        raise ValueError(f"Unrecognized syn_exp_name {args.syn_exp_name}")


def trainer_stats_for_wandb(
    args: AutoregExperimentsArguments,
    trainer: Trainer
):
    """ Make some statistics to report to wandb """
    if args.syn_exp_name in ["one_shot", "one_shot_str"]:
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

    elif args.syn_exp_name in ["autoreg_ksteps"]:
        return {
            "train_len": len(trainer.train_dataset),
            "eval_len": len(trainer.eval_dataset),
        }

    else:
        raise ValueError(f"Unrecognized syn_exp_name {args.syn_exp_name}")


def set_wandb_run_id(args: AutoregExperimentsArguments):
    """ Set the wandb run id for this experiment """
    run_id_map_file = Path(DUMP_DIR, "run_id_map.json")
    if not run_id_map_file.exists():
        run_id_map = {}
    else:
        run_id_map = json.load(open(str(run_id_map_file), "r"))

    exp_id = synexp_args_to_wandb_run_name(args)
    
    if exp_id in run_id_map.keys():
        run_id = run_id_map[exp_id]
        print(f"Found run_id {run_id} for {exp_id}")
    else:
        run_id = wandb.util.generate_id()
        run_id_map[exp_id] = run_id
        json.dump(run_id_map, open(str(Path(DUMP_DIR, "run_id_map.json")), "w"))
        print(f"Generated run_id {run_id} for {exp_id}")
    
    # Set the run_id to ensure resuming any previous run
    os.environ["WANDB_RUN_ID"] = run_id


def make_trainer_for_autoreg(
    args: AutoregExperimentsArguments,
    report_to: str = "wandb"
):
    """ Make a Hugging Face Trainer object """
    if args.syn_exp_name == "autoreg_ksteps" and args.model_name == "gpt2":
        train_dataset = AutoregKStepsTokensDataset(
            num_vars = args.num_vars,
            num_rules_range = (args.min_num_rules, args.max_num_rules),
            ante_prob_range = (args.min_ante_prob, args.max_ante_prob),
            conseq_prob_range = (args.min_conseq_prob, args.max_conseq_prob),
            chain_len_range = (args.min_chain_len, args.max_chain_len),
            num_prevs_range = (1, args.min_chain_len),
            num_steps = args.num_steps,
            dataset_len = args.train_len
        )

        eval_dataset = AutoregKStepsTokensDataset(
            num_vars = args.num_vars,
            num_rules_range = (args.min_num_rules, args.max_num_rules),
            ante_prob_range = (args.min_ante_prob, args.max_ante_prob),
            conseq_prob_range = (args.min_conseq_prob, args.max_conseq_prob),
            chain_len_range = (args.min_chain_len, args.max_chain_len),
            num_prevs_range = (1, args.min_chain_len),
            num_steps = args.num_steps,
            dataset_len = args.eval_len
        )

        seqcls_model = MyGPT2SeqClsModel(MyGPT2Config(
            input_dim = 2 * args.num_vars,
            num_vars = args.num_vars,
            embed_dim = args.embed_dim,
            num_heads = args.num_heads,
            num_layers = args.num_layers,
            use_positional_embedding = False
        ))

        task_model = AutoregKStepsTaskModel(
            seqcls_model = seqcls_model,
            num_steps = args.num_steps,
            train_supervision_mode = "all"
        )

        training_args = TrainingArguments(
            str(Path(args.output_dir, synexp_args_to_wandb_run_name(args))),
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = synexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            learning_rate = args.learning_rate,
            warmup_ratio = 0.10,
            save_strategy = "epoch",
            save_total_limit = 2
        )

        return Trainer(
            task_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = autoreg_ksteps_metrics
        )

    else:
        raise ValueError(f"Unrecognized exp_name {args.syn_exp_name}")


""" Main stuff """

if __name__ == "__main__":
    parser = HfArgumentParser(AutoregExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    trainer = make_trainer_for_autoreg(args)

    set_wandb_run_id(args)
    
    # Log some preliminary training stats
    trainer_stats = trainer_stats_for_wandb(args, trainer)

    experiment_id = synexp_args_to_wandb_run_name(args)
    checkpoint = load_checkpoint_from_wandb(
        experiment_out_dir=str(Path(args.output_dir, experiment_id)),
        experiment_id=experiment_id
    )
    if checkpoint is not None:
        print("Found checkpoint: ", checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("No checkpoint found. Training from scratch.")
        trainer.train()
    
    wandb.run.summary["trainer_stats"] = trainer_stats
    wandb.finish()


