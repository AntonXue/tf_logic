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

sys.path.insert(0, PROJ_ROOT)
print(sys.path)
from my_datasets.utils.logic_utils import step_rules


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


@torch.no_grad()
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
                num_dones, cum_elem_hits, cum_state_hits = 0, 0, 0
                pbar = tqdm(dataloader)

                for batch in pbar:
                    batch_tokens = batch["tokens"].to(args.device)
                    # Slice out the first rule to make the length correct
                    batch_tokens = batch_tokens[:,1:]
                    tgt_state = (torch.rand(args.batch_size, n) < 0.5).long().to(args.device)
                    tgt_scaled = tgt_state - kappa * (1 - tgt_state)

                    adv_token = torch.cat([torch.zeros_like(tgt_scaled), tgt_scaled.float()], dim=1)
                    all_tokens = torch.cat([batch_tokens, adv_token.view(-1,1,2*n)], dim=1)
                    out = model(all_tokens)
                    pred = (out.logits[:,0] > 0).long()
                    num_dones += batch_tokens.size(0)
                    cum_elem_hits += (tgt_state == pred).float().mean(dim=1).sum()
                    cum_state_hits += ((tgt_state == pred).sum(dim=1) == n).sum()
                    elems_acc = cum_elem_hits / num_dones
                    states_acc = cum_state_hits / num_dones

                    desc_str = f"n {n}, d {d}, log(k) {kappa_power:.3f}: "
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


@torch.no_grad()
def run_repeat_token_attack(args):
    config = Namespace(**json.load(open(args.config_file)))
    assert config.num_samples % args.batch_size == 0

    saveto_file = Path(args.output_dir, "theory_attack_repeat_token.csv")
    print(f"Will save to: {saveto_file}")

    df = pd.DataFrame(columns=[
        "train_seed", "num_vars", "embed_dim", "elems_acc", "states_acc"
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
                max_test_seq_len = 2**16,
            )

            # We need to make the context length very big

            model.eval().to(args.device)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            for repeat_power in config.repeat_powers:
                num_repeats = 2 ** repeat_power
                num_dones, cum_elem_hits, cum_state_hits = 0, 0, 0
                pbar = tqdm(dataloader)

                for batch in pbar:
                    batch_tokens = batch["tokens"].to(args.device)
                    tgt_state = (torch.rand(args.batch_size, n) < 0.5).long().to(args.device)
                    tgt_repeated = tgt_state.view(-1,1,n).repeat(1,num_repeats,1)
                    adv_tokens = torch.cat([torch.zeros_like(tgt_repeated), tgt_repeated], dim=2)
                    all_tokens = torch.cat([batch_tokens, adv_tokens.float()], dim=1)
                    out = model(all_tokens)
                    pred = (out.logits[:,0] > 0).long()
                    num_dones += batch_tokens.size(0)
                    cum_elem_hits += (tgt_state == pred).float().mean(dim=1).sum()
                    cum_state_hits += ((tgt_state == pred).sum(dim=1) == n).sum()
                    elems_acc = cum_elem_hits / num_dones
                    states_acc = cum_state_hits / num_dones

                    desc_str = f"n {n}, d {d}, nr {num_repeats}: "
                    desc_str += f"N {num_dones}, elems {elems_acc:.3f}, states {states_acc:.3f}"
                    pbar.set_description(desc_str)

                this_df = pd.DataFrame({
                    "train_seed": train_seed,
                    "num_vars": n,
                    "embed_dim": d,
                    "repeat_power": int(repeat_power),
                    "elems_acc": elems_acc.item(),
                    "states_acc": states_acc.item(),
                }, index=[df_idx])

                df_idx += 1

                df = pd.concat([df, this_df])
                df.to_csv(saveto_file)

            print(f"Done with: trseed {train_seed}, n {n}, d {d}")


@torch.no_grad()
def run_suppress_rule_attack(args):
    config = Namespace(**json.load(open(args.config_file)))
    assert config.num_samples % args.batch_size == 0

    saveto_file = Path(args.output_dir, "theory_attack_suppress_rule.csv")
    print(f"Will save to: {saveto_file}")

    for train_seed in config.train_seeds:
        for nd_pair in config.nd_pairs:
            n, d = nd_pair
            model, res_dataset = load_model_and_dataset_from_big_grid(
                embed_dim = d,
                num_vars = n,
                num_steps = 3,
                seed = train_seed,
                dataset_len = config.num_samples
            )

            model.eval().to(args.device)
            supp_dataset = SuppressRuleDataset(res_dataset)

            num_dones = 0
            cum_elem_hits, cum_state_hits = 0, 0
            bcum_elem_hits, bcum_state_hits = 0, 0

            dataloader = DataLoader(supp_dataset, batch_size=args.batch_size, shuffle=True)
            pbar = tqdm(dataloader)
            for i, batch in enumerate(pbar):
                # The rule to suppress
                tokens, supp_rule, labels = batch["tokens"], batch["supp_rule"], batch["labels"]
                tokens, supp_rule, labels = \
                    tokens.to(args.device), supp_rule.to(args.device), labels.to(args.device)
                
                supp_ante, supp_conseq = supp_rule.chunk(2, dim=-1)
                adv_token = torch.cat([supp_ante, -1*supp_conseq], dim=-1)
                adv_input = torch.cat([adv_token.view(-1,1,2*n), tokens], dim=1).to(args.device)
                res_out = model(adv_input)
                res_pred = (res_out.logits > 0)

                # Test the would-have-been thing
                state = tokens[:,-1,n:]
                s, true_succs = state, []
                for _ in range(supp_dataset.num_steps):
                    s, _ = step_rules(adv_input, s)
                    true_succs.append(s)
                true_succs = torch.stack(true_succs, dim=1)

                
                num_dones += tokens.size(0)
                cum_elem_hits += (res_pred == labels).float().mean(dim=(1,2)).sum()
                cum_state_hits += ((res_pred == labels).sum(dim=2) == n).sum()
                elem_acc = cum_elem_hits / num_dones
                state_acc = cum_state_hits / num_dones
                
                bcum_elem_hits += (true_succs == labels).float().mean(dim=(1,2)).sum()
                bcum_state_hits += ((true_succs == labels).sum(dim=2) == n).float().mean(dim=-1).sum()
                belem_acc = bcum_elem_hits / num_dones
                bstate_acc = bcum_state_hits / num_dones

                desc_str = f"n {n}, d {d}: "
                desc_str += f"N {num_dones}, elems {elem_acc:.3f}, states {state_acc:.3f}, "
                desc_str += f"belems {belem_acc:.3f}, bstates {bstate_acc:.3f}, "
                pbar.set_description(desc_str)


if __name__ == "__main__":
    parser = HfArgumentParser(TheoryAttackExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.attack_name == "big_token":
        ret = run_big_token_attack(args)

    elif args.attack_name == "repeat_token":
        ret = run_repeat_token_attack(args)

    elif args.attack_name == "suppress_rule":
        ret = run_suppress_rule_attack(args)

    else:
        raise ValueError(f"Unknown attack name {args.attack_name}")



