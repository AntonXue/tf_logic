import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser, AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import wandb
import json

""" Our imports """
from common import *    # Critical definitions and path inserts
from models import AutoTaskModel
from my_datasets import *
from minecraft.dataset import MinecraftAutoregKStepsTokensDataset
from utils.metrics import *
from utils.model_loader_utils import load_checkpoint_from_wandb
from datasets import Dataset

""" Parser for Hugging Face """

@dataclass
class MinecraftExperimentsArguments:
    """ This class captures ALL the possible minecraft experiments we may want to run """
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "minecraft_experiments")),
        metadata = {"help": "Output directory of minecraft experiments."}
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

    use_pretrained : Optional[bool] = field(
        default = True,
        metadata = {"help": "Weights from the pretrained model are loaded if True. " + \
                    "Note that this restricts changes to the model " + \
                    "(default num_heads, num_layers, etc. will be used)."}
    )

    """ Dataset details """

    num_steps: Optional[int] = field(
        default = 3,
        metadata = {"help": "The number of steps; used for autoreg_ksteps."}
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


def minecraftexp_args_to_wandb_run_name(args: MinecraftExperimentsArguments):
    model_str = f"{args.model_name}" + \
                (f"_pt" if args.use_pretrained else "")

    if args.syn_exp_name == "one_shot":
        return f"MinecraftOS_{model_str}__" + \
            f"nv{args.num_vars}" + \
            f"_nr{args.min_num_rules}-{args.max_num_rules}" + \
            f"_ap{args.min_ante_prob:.2f}-{args.max_ante_prob:.2f}" + \
            f"_bp{args.min_conseq_prob:.2f}-{args.max_conseq_prob:.2f}" + \
            f"_cl{args.min_chain_len}-{args.max_chain_len}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}_seed{args.seed}"

    elif args.syn_exp_name == "one_shot_str":
        return f"MinecraftOSstr__{model_str}__" + \
            f"ns{args.num_steps}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}_seed{args.seed}"

    elif args.syn_exp_name == "next_state":
        return f"MinecraftNS_{model_str}__" + \
            f"nv{args.num_vars}" + \
            f"_nr{args.min_num_rules}-{args.max_num_rules}" + \
            f"_ns{args.min_num_states}-{args.max_num_states}" + \
            f"_ap{args.min_ante_prob:.2f}-{args.max_ante_prob:.2f}" + \
            f"_bp{args.min_conseq_prob:.2f}-{args.max_conseq_prob:.2f}" + \
            f"_sp{args.min_state_prob:.2f}-{args.max_state_prob:.2f}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}_seed{args.seed}"

    elif args.syn_exp_name == "autoreg_ksteps":
        return f"MinecraftAR_{model_str}__" + \
            f"nv{args.num_vars}" + \
            f"_ns{args.num_steps}" + \
            f"_nr{args.min_num_rules}-{args.max_num_rules}" + \
            f"_ap{args.min_ante_prob:.2f}-{args.max_ante_prob:.2f}" + \
            f"_bp{args.min_conseq_prob:.2f}-{args.max_conseq_prob:.2f}" + \
            f"_cl{args.min_chain_len}-{args.max_chain_len}" + \
            f"_ntr{args.train_len}_ntt{args.eval_len}_seed{args.seed}"

    else:
        raise ValueError(f"Unrecognized syn_exp_name {args.syn_exp_name}")


def trainer_stats_for_wandb(
    args: MinecraftExperimentsArguments,
    trainer: Trainer
):
    """ Make some statistics to report to wandb """
    if args.syn_exp_name in ["one_shot", "one_shot_str"]:
        num_train_qeds = torch.tensor(0)
        for i in range(len(trainer.train_dataset)):
            num_train_qeds += trainer.train_dataset[i]["label"]

        num_eval_qeds = torch.tensor(0)
        for i in range(len(trainer.eval_dataset)):
            num_eval_qeds += trainer.eval_dataset[i]["label"]

        return {
            "train_len": len(trainer.train_dataset),
            "train_qeds": num_train_qeds.item(),
            "eval_len" : len(trainer.eval_dataset),
            "eval_qeds": num_eval_qeds.item()
        }

    elif args.syn_exp_name == "next_state":
        return {
            "train_len": len(trainer.train_dataset),
            "eval_len": len(trainer.eval_dataset),
        }

    elif args.syn_exp_name == "autoreg_ksteps":
        return {
            "train_len": len(trainer.train_dataset),
            "eval_len": len(trainer.eval_dataset),
        }

    else:
        raise ValueError(f"Unrecognized syn_exp_name {args.syn_exp_name}")

# TODO: Refactor this later (move to common.py)
def set_wandb_run_id(args: MinecraftExperimentsArguments):
    """ Set the wandb run id for this experiment """
    run_id_map_file = Path(DUMP_DIR, "run_id_map.json")
    if not run_id_map_file.exists():
        run_id_map = {}
    else:
        run_id_map = json.load(open(str(run_id_map_file), "r"))

    exp_id = minecraftexp_args_to_wandb_run_name(args)
    
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


def make_trainer_for_synthetic(
    args: MinecraftExperimentsArguments,
    report_to: str = "wandb"
):
    """ Make a Hugging Face Trainer object """

    if args.syn_exp_name == "one_shot_str":
        print("Creating a trainer for one_shot_str")
        # Get the tokenizer to create the dataset
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        big_dataset = MinecraftAutoregKStepsTokensDataset(
            num_steps=args.num_steps,
            dataset_len=args.train_len + args.eval_len,
            tokenizer=tokenizer)
        
        train_len, eval_len = args.train_len, args.eval_len
        if len(big_dataset) < args.train_len + args.eval_len:
            train_len, eval_len = len(big_dataset) * (args.train_len / (args.train_len + args.eval_len)), len(big_dataset) * (args.eval_len / (args.train_len + args.eval_len))
            train_len, eval_len = int(train_len), len(big_dataset) - int(train_len)

        print(f"train_len: {train_len}, eval_len: {eval_len}")

        train_dataset, eval_dataset = \
            torch.utils.data.random_split(big_dataset, [train_len, eval_len])

        if args.use_pretrained:
            train_hf_dataset = Dataset.from_dict({
                "data": [train_dataset[i]['data'] for i in range(len(train_dataset))],
                "label": [train_dataset[i]['labels'] for i in range(len(train_dataset))],
            }).with_format("torch")

            eval_hf_dataset = Dataset.from_dict({
                "data": [eval_dataset[i]['data'] for i in range(len(eval_dataset))],
                "label": [eval_dataset[i]['labels'] for i in range(len(eval_dataset))],
            }).with_format("torch")

            def tokenize_function(item):
                return tokenizer(item["data"], truncation=True)

            train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_hf_dataset.map(tokenize_function, batched=True)

            tfl_model = AutoModelForSequenceClassification.from_pretrained(
                            args.model_name, num_labels=2
                        )
            tfl_model.config.pad_token_id = tokenizer.pad_token_id

        else:
            tfl_model = AutoTaskModel.from_kwargs(
                task_name = args.syn_exp_name,
                num_vars = args.num_vars,
                model_name = args.model_name,
                embed_dim = args.embed_dim,
                num_layers = args.num_layers,
                num_heads = args.num_heads)
            tfl_model.seqcls_model.model.config.pad_token_id = tokenizer.pad_token_id

        training_args = TrainingArguments(
            str(Path(args.output_dir, minecraftexp_args_to_wandb_run_name(args))),
            num_train_epochs = args.num_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            auto_find_batch_size = args.auto_find_batch_size,
            evaluation_strategy = "epoch",
            report_to = report_to,
            run_name = minecraftexp_args_to_wandb_run_name(args),
            logging_steps = args.logging_steps,
            warmup_ratio = 0.10,
            save_strategy = "no")

        return Trainer(
            tfl_model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator,
            compute_metrics = one_shot_metrics)

    else:
        raise ValueError(f"Unrecognized exp_name {args.syn_exp_name}")


""" Main stuff """

if __name__ == "__main__":
    parser = HfArgumentParser(MinecraftExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    trainer = make_trainer_for_synthetic(args)

    set_wandb_run_id(args)
    
    # Log some preliminary training stats
    trainer_stats = trainer_stats_for_wandb(args, trainer)

    experiment_id = minecraftexp_args_to_wandb_run_name(args)
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

