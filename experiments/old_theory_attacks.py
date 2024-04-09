"""
Script to run the theory attack on learned models.

The theory attack attempts to make the model predict a pre-chosen binary vector s_tgt as follows:
1. Choose a binary vector s_tgt.
2. Scale it by a factor lambda_val to get lambda_adjusted_s_tgt = s_tgt - lambda_val * (1 - s_tgt).
3. Create the u_atk vector as [0 0n lambda_adjusted_s_tgt] where 0n is a vector of n 0s.
4. Replace the last token of the input sequence with u_atk.
5. Run the model on the new sequence and check if the prediction is s_tgt.

Usage:
    python theory_attacks.py --config_file <path_to_config_file> --lambda_list <list_of_lambda_values> --output_dir <path_to_output_dir>
"""
from argparse import ArgumentParser
import json
import itertools
from common import *
from my_datasets.task_datasets import AutoregKStepsTokensDataset
from utils.model_loader_utils import load_next_state_model_from_wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

LAMBDA = [10**i for i in range(-3, 10)]
LAMBDA.reverse()
DEFAULT_DUMP_DIR = str(Path(DUMP_DIR, "theory_attack_experiments_l1"))


def get_param_dict_list_from_config_file(config_file) -> list:
    """Returns a list of dictionaries of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config


def get_file_name(param_instance):
    """Returns the name of the file to dump the results in."""

    if param_instance["syn_exp_name"] == "autoreg_ksteps":
        return (
            f"{param_instance['model_name']}"
            + f"_nv{param_instance['num_vars']}"
            + f"_ns{param_instance['num_steps']}"
            + f"_nr{param_instance['min_num_rules']}-{param_instance['max_num_rules']}"
            + f"_ap{param_instance['min_ante_prob']:.2f}-{param_instance['max_ante_prob']:.2f}"
            + f"_bp{param_instance['min_conseq_prob']:.2f}-{param_instance['max_conseq_prob']:.2f}"
            + f"_cl{param_instance['min_chain_len']}-{param_instance['max_chain_len']}"
            + f"_ntr{param_instance['train_len']}_ntt{param_instance['eval_len']}_seed{param_instance['seed']}.json"
        )

    raise ValueError(f"syn_exp_name {param_instance['syn_exp_name']} not supported.")


@torch.no_grad()
def run_eval(
    model, dataset, param_instance, lambda_list=LAMBDA, output_dir=DEFAULT_DUMP_DIR
):
    """Runs the theory attack on the given model and dataset."""
    torch.manual_seed(param_instance["seed"])
    print("Running eval for param_instance: ", param_instance)
    model.eval()

    batch_size = param_instance["eval_batch_size"]
    num_vars = param_instance["num_vars"]

    lambda_metrics = {
        lambda_val: {
            "running_hits": 0,
            "states_hits": 0,
            "target_running_hits": 0,
            "target_states_hits": 0,
            "num_dones": 0,
        }
        for lambda_val in lambda_list
    }
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pbar = tqdm(dataloader)
    for batch in pbar:
        tokens, orig_labels = batch["tokens"], batch["labels"]

        # Create the adversarial target states
        # s_tgts shape: (batch_size, 1, num_vars)
        s_tgts = torch.randint(0, 2, (tokens.size(0), 1, num_vars))

        for lambda_val in lambda_list:
            lambda_adjusted_s_tgts = s_tgts - lambda_val * (1 - s_tgts)

            # u_atks shape: (batch_size, 2*num_vars + 1, 1)
            u_atks = torch.cat(
                (torch.zeros(tokens.size(0), 1, num_vars + 1), lambda_adjusted_s_tgts),
                dim=2,
            )

            # Replace the last token of tokens with u_atks
            # adv_tokens shape: (batch_size, num_rules, 2*num_vars + 1) == tokens.shape
            adv_tokens = torch.cat((tokens[:, :-1, :], u_atks), dim=1)

            out = model(adv_tokens, labels=s_tgts)
            preds = (out.logits > 0).long()

            lambda_metrics[lambda_val]["running_hits"] += (preds == s_tgts).sum()
            lambda_metrics[lambda_val]["states_hits"] += (
                torch.mean((preds == s_tgts).float(), axis=2) > 1 - 1e-5
            ).sum()
            lambda_metrics[lambda_val]["target_running_hits"] += (
                preds[:, -1] == s_tgts[:, -1]
            ).sum()
            lambda_metrics[lambda_val]["target_states_hits"] += (
                torch.mean((preds[:, -1] == s_tgts[:, -1]).float(), axis=1) > 1 - 1e-5
            ).sum()
            lambda_metrics[lambda_val]["num_dones"] += tokens.size(0)

            target_elems_acc = lambda_metrics[lambda_val]["running_hits"].item() / (
                lambda_metrics[lambda_val]["num_dones"] * num_vars
            )
            target_states_acc = (
                lambda_metrics[lambda_val]["target_states_hits"].item()
                / lambda_metrics[lambda_val]["num_dones"]
            )
            desc = f"Target Elems acc {target_elems_acc:.3f} | Target state acc {target_states_acc:.3f}"

            pbar.set_description(desc)

    acc_lambda_metrics = {
        lambda_val: {
            "ElemsAcc": lambda_metrics[lambda_val]["running_hits"].item()
            / (
                lambda_metrics[lambda_val]["num_dones"]
                * num_vars
                * param_instance["num_steps"]
            ),
            "StatesAcc": lambda_metrics[lambda_val]["states_hits"].item()
            / (lambda_metrics[lambda_val]["num_dones"] * param_instance["num_steps"]),
            "TargetElemsAcc": lambda_metrics[lambda_val]["running_hits"].item()
            / (lambda_metrics[lambda_val]["num_dones"] * num_vars),
            "TargetStatesAcc": lambda_metrics[lambda_val]["target_states_hits"].item()
            / lambda_metrics[lambda_val]["num_dones"],
        }
        for lambda_val in lambda_list
    }

    # Save the metrics in a json file
    out_file = get_file_name(param_instance)

    with open(os.path.join(output_dir, out_file), "w") as f:
        json.dump(acc_lambda_metrics, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="theory_attack_base_config.json",
        help="Path to JSON config file with the dataset and model details.",
    )
    parser.add_argument(
        "--lambda_list",
        type=list,
        default=LAMBDA,
        help="List of lambda values to use for the theory attack.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_DUMP_DIR,
        help="Path to directory where the results will be dumped.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = get_param_dict_list_from_config_file(args.config_file)
    print("Running with config: ", config)

    for param_dict in config:
        for param_tuple in itertools.product(*param_dict.values()):
            param_instance = {}
            for i, param in enumerate(param_tuple):
                param_instance[list(param_dict.keys())[i]] = param

            print("Running with param_instance: ", param_instance)
            torch.manual_seed(param_instance["seed"])

            if "syn_exp_name" in param_instance:
                assert (
                    param_instance["syn_exp_name"] == "autoreg_ksteps"
                ), "syn_exp_name must be autoreg_ksteps for this experiment."
            else:
                raise ValueError("syn_exp_name must be specified in config file.")

            dataset = AutoregKStepsTokensDataset(
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
                chain_len_range=(
                    param_instance["min_chain_len"],
                    param_instance["max_chain_len"],
                ),
                num_steps=param_instance["num_steps"],
                dataset_len=param_instance["eval_len"],
            )

            model = load_next_state_model_from_wandb(
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
            )

            run_eval(
                model,
                dataset,
                param_instance,
                lambda_list=args.lambda_list,
                output_dir=args.output_dir,
            )
