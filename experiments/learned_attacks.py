from argparse import ArgumentParser
import json
import itertools
import torch
import wandb

from common import *
from my_datasets.attack_datasets import AutoregKStepsTokensAttackDataset
from utils.model_loader_utils import load_next_state_model_from_wandb
from models.attack_models import ForceOutputWithAppendedAttackTokensWrapper, ForceOutputWithAppendedAttackSeqClsTokensWrapper
from transformers import TrainingArguments, Trainer
from utils.metrics import autoreg_ksteps_metrics

DEFAULT_DUMP_DIR = str(Path(DUMP_DIR, "learned_attack_experiments"))

def get_param_dict_list_from_config_file(config_file) -> list:
    """Returns a list of dictionaries of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config

def synexp_args_to_wandb_run_name(param_instance):
    if param_instance["syn_exp_name"] == "autoreg_ksteps":
        return (
            f"SynARAttack_{param_instance['model_name']}"
            + f"_d{param_instance['embed_dim']}"
            + f"_L{param_instance['num_layers']}"
            + f"_H{param_instance['num_heads']}"
            + f"_nv{param_instance['num_vars']}"
            + f"_ns{param_instance['num_steps']}"
            + f"_nr{param_instance['min_num_rules']}-{param_instance['max_num_rules']}"
            + f"_seed{param_instance['seed']}"
            + f"_atk-{param_instance['base_attack_model_name']}_nat{param_instance['num_attack_tokens']}"
            + f"_clamp{param_instance['clamp']}"
            + f"_rt{param_instance['repeat_tokens']}"
            + f"_atkntr{param_instance['attack_train_len']}_atkntt{param_instance['attack_eval_len']}"
        )

def run_attack(attacker_model, reasoner_model, train_dataset, eval_dataset, param_instance, output_dir=DEFAULT_DUMP_DIR):
    """Train the attacker model on the given dataset."""
    training_args = TrainingArguments(
        str(Path(output_dir, synexp_args_to_wandb_run_name(param_instance))),
        num_train_epochs = param_instance["num_attack_epochs"],
        per_device_train_batch_size = param_instance["attack_train_batch_size"],
        per_device_eval_batch_size = param_instance["attack_eval_batch_size"],
        auto_find_batch_size = True,
        evaluation_strategy = "epoch",
        report_to = "wandb",
        run_name = synexp_args_to_wandb_run_name(param_instance),
        logging_steps = 10,
        warmup_ratio = 0.10,
        save_strategy = "epoch",
        save_total_limit = 2,
        load_best_model_at_end = True)

    trainer = Trainer(
        attacker_model,
        training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics = autoreg_ksteps_metrics)
    
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/learned_attacks_config.json",
        help="Path to the config file with the reasoner / base model details"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_DUMP_DIR,
        help="Path to directory where the results will be dumped."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = get_param_dict_list_from_config_file(args.config_file)
    print("Running with config: ", config)

    for param_dict in config:
        print("Param dict:", param_dict)
        for param_tuple in itertools.product(*param_dict.values()):
            param_instance = {}
            for i, param in enumerate(param_tuple):
                param_instance[list(param_dict.keys())[i]] = param

            param_instance["repeat_tokens"] = False if "repeat_tokens" not in param_instance else (param_instance["repeat_tokens"] == 1)
            param_instance["clamp"] = True if "clamp" not in param_instance else (param_instance["clamp"] == 1)
            param_instance["rho"] = 0.001 if "rho" not in param_instance else param_instance["rho"]
            param_instance["base_attack_model_name"] = "gpt2" if "base_attack_model_name" not in param_instance else param_instance["base_attack_model_name"]

            print("Reasoner with params: ", param_instance)
            torch.manual_seed(param_instance["seed"])

            if "syn_exp_name" in param_instance:
                assert (
                    param_instance["syn_exp_name"] == "autoreg_ksteps"
                ), "syn_exp_name must be autoreg_ksteps for this experiment."
            else:
                raise ValueError("syn_exp_name must be specified in config file.")

            big_dataset = AutoregKStepsTokensAttackDataset(
                num_vars=param_instance["num_vars"],
                num_rules_range=(
                    param_instance["min_num_rules"],
                    param_instance["max_num_rules"] - param_instance["num_attack_tokens"],
                ),
                ante_prob_range=(
                    param_instance["min_ante_prob"],
                    param_instance["max_ante_prob"],
                ),
                conseq_prob_range=(
                    param_instance["min_conseq_prob"],
                    param_instance["max_conseq_prob"],
                ),
                chain_len_range=(
                    param_instance["min_chain_len"],
                    param_instance["max_chain_len"],
                ),
                num_steps=param_instance["num_steps"],
                dataset_len=param_instance["attack_train_len"] + param_instance["attack_eval_len"],
            )

            train_dataset, eval_dataset = torch.utils.data.random_split(
                big_dataset, [param_instance["attack_train_len"], param_instance["attack_eval_len"]]
            )

            reasoner_model = load_next_state_model_from_wandb(
                model_name=param_instance["model_name"],
                embed_dim=param_instance["embed_dim"],
                num_layers=param_instance["num_layers"],
                num_heads=param_instance["num_heads"],
                num_vars=param_instance["num_vars"],
                num_rules_range=(
                    param_instance["min_num_rules"],
                    param_instance["max_num_rules"],
                ),
                ante_prob_range=(
                    param_instance["min_ante_prob"],
                    param_instance["max_ante_prob"],
                ),
                conseq_prob_range=(
                    param_instance["min_conseq_prob"],
                    param_instance["max_conseq_prob"],
                ),
                train_len=param_instance["train_len"],
                eval_len=param_instance["eval_len"],
                task_name=param_instance["syn_exp_name"],
                num_steps=param_instance["num_steps"],
                chain_len_range=(
                    param_instance["min_chain_len"],
                    param_instance["max_chain_len"],
                ),
                include_seed_in_run_name=True,
                seed=param_instance["seed"],
                bsz=param_instance["train_batch_size"],
                include_train_batch_size_in_run_name=True,
            )

            attacker_model = ForceOutputWithAppendedAttackSeqClsTokensWrapper(
                seqcls_model=reasoner_model,
                num_attack_tokens=param_instance["num_attack_tokens"],
                base_attack_model_name=param_instance["base_attack_model_name"],
                rho=param_instance["rho"],
                clamp=param_instance["clamp"],
                repeat_tokens=param_instance["repeat_tokens"],
            )

            run_attack(attacker_model, reasoner_model, train_dataset, eval_dataset, param_instance, args.output_dir)

