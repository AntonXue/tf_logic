import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import HfArgumentParser
from argparse import Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd


""" Our imports """
from common import *
from utils.model_loader_utils import *


@dataclass
class TheoryAttackExperimentsArguments:
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "theory_attacks")),
        metadata = {"help": "Output directory for theory attacks."}
    )

    attack_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The experiment to run."}
    )

    config_file: Optional[str] = field(
        default = None,
        metadata = {"help": "Where the config file is."}
    )

    experiment_seed: int = field(
        default = 1234,
        metadata = {"help": "The seed to use for initializing stuff."}
    )

    batch_size: int = field(
        default = 256,
        metadata = {"help": "The batch size for evaluating stuff."}
    )

    device: str = field(
        default = "cuda",
    )


def run_big_token_attack(args):
    config = Namespace(**json.load(open(args.config_file)))
    assert config.num_samples % args.batch_size == 0

    saveto_file = Path(args.output_dir, "theory_attack_big_token.csv")
    print(f"Will save to: {saveto_file}")

    df = pd.DataFrame(columns=[
        "train_seed", "num_vars", "embed_dim", "kappa_power", "elems_acc", "states_acc"
    ])

    df_idx = 0

    for train_seed in config.train_seeds:
        for nd_pair in config.nd_pairs:
            n, d = nd_pair
            model, dataset = load_model_and_dataset_from_big_grid(
                embed_dim = d,
                num_vars = n,
                num_steps = 3, # TODO: change
                seed = train_seed,
                dataset_len = config.num_samples,
            )

            model.eval().to(args.device)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            for kappa_power in config.kappa_powers:
                kappa = torch.tensor(10. ** kappa_power)
                num_dones, acc_elem_hits, acc_state_hits = 0, 0, 0
                pbar = tqdm(dataloader)

                for batch in pbar:
                    batch_tokens = batch["tokens"].to(args.device)
                    s_tgt = (torch.rand(args.batch_size, n) < 0.5).long().to(args.device)
                    kappa_token = s_tgt - kappa * (1 - s_tgt)
                    adv_token = torch.cat([torch.zeros_like(s_tgt), kappa_token.float()], dim=1)
                    all_tokens = torch.cat([batch_tokens, adv_token.view(-1,1,2*n)], dim=1)
                    out = model(all_tokens)
                    pred = (out.logits[:,0] > 0).long()
                    num_dones += batch_tokens.size(0)
                    acc_elem_hits += (s_tgt == pred).float().mean(dim=1).sum()
                    acc_state_hits += ((s_tgt == pred).sum(dim=1) == n).sum()
                    elems_acc = acc_elem_hits / num_dones
                    states_acc = acc_state_hits / num_dones

                    desc_str = f"n {n}, d {d}, log(k) {kappa_power:.2f}: "
                    desc_str += f"N {num_dones}, elems {elems_acc:.3f}, states {states_acc:.3f}"
                    pbar.set_description(desc_str)

                this_df = pd.DataFrame({
                    "train_seed": train_seed,
                    "num_vars": n,
                    "embed_dim": d,
                    "kappa_power": kappa_power,
                    "elems_acc": elems_acc.item(),
                    "states_acc": states_acc.item(),
                }, index=[df_idx])

                df_idx += 1

                df = pd.concat([df, this_df])
                df.to_csv(saveto_file)

            print(f"Done with: trseed {train_seed}, n {n}, d {d}")



def run_repeat_tokens_attack(model, dataset, num_repeats, num_samples):
    pass
    return {
        "num_samples": num_samples,
        "state_acc": None,
        "elems_acc": None,
    }



if __name__ == "__main__":
    parser = HfArgumentParser(TheoryAttackExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.attack_name == "big_token_attack":
        run_big_token_attack(args)
    elif args.attack_name == "repeat_token_attack":
        pass



