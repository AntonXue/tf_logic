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

    config_file: Optional[str] = field(
        default = None,
        metadata = {"help": "Where the config file is."}
    )

    device: str = field(
        default = "cuda",
    )


@torch.no_grad()
def run_big_token_attack(config):
    assert config.num_samples % config.batch_size == 0
    saveto_file = Path(config.output_dir, "theory_attack_big_token.csv")
    print(f"Will save to: {saveto_file}")

    df_idx = 0
    df = pd.DataFrame(columns=[
        "train_seed", "num_vars", "embed_dim", "kappa_power", "elems_acc", "states_acc"
    ])

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

            model.eval().to(config.device)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

            for kappa_power in config.kappa_powers:
                kappa = torch.tensor(10. ** kappa_power)
                num_dones, cum_elem_hits, cum_state_hits = 0, 0, 0
                pbar = tqdm(dataloader)

                for batch in pbar:
                    batch_tokens = batch["tokens"].to(config.device)
                    # Slice out the first rule to make the length correct
                    batch_tokens = batch_tokens[:,1:]
                    tgt_state = (torch.rand(config.batch_size, n) < 0.5).long().to(config.device)
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
def run_repeat_token_attack(config):
    assert config.num_samples % config.batch_size == 0
    saveto_file = Path(config.output_dir, "theory_attack_repeat_token.csv")
    print(f"Will save to: {saveto_file}")

    df_idx = 0
    df = pd.DataFrame(columns=[
        "train_seed", "num_vars", "embed_dim", "elems_acc", "states_acc"
    ])

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

            model.eval().to(config.device)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

            for repeat_power in config.repeat_powers:
                num_repeats = 2 ** repeat_power
                num_dones, cum_elem_hits, cum_state_hits = 0, 0, 0
                pbar = tqdm(dataloader)

                for batch in pbar:
                    batch_tokens = batch["tokens"].to(config.device)
                    tgt_state = (torch.rand(config.batch_size, n) < 0.5).long().to(config.device)
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
def run_suppress_rule_attack(config):
    assert config.num_samples % config.batch_size == 0
    saveto_file = Path(config.output_dir, "theory_attack_suppress_rule.csv")
    print(f"Will save to: {saveto_file}")

    df_idx = 0
    df = pd.DataFrame(columns=[
        "train_seed", "num_vars", "embed_dim", "raw_acc", "adv_acc",
        "adv_top3", "adv_weight", "adv_rel_weight"
    ])


    for train_seed in config.train_seeds:
        for nd_pair in config.nd_pairs:
            n, embed_dim = nd_pair
            model, res_dataset = load_model_and_dataset_from_big_grid(
                embed_dim = embed_dim,
                num_vars = n,
                num_steps = 3,
                seed = train_seed,
                dataset_len = config.num_samples
            )

            model.eval().to(config.device)
            supp_dataset = SuppressRuleDataset(
                num_vars = n,
                num_rules = res_dataset.num_rules_range[1],
                dataset_len = len(res_dataset),
            )

            num_dones = 0
            cum_raw_hits = 0
            cum_adv_hits = 0
            cum_top3_hits = 0
            cum_adv_weight = 0

            dataloader = DataLoader(supp_dataset, batch_size=config.batch_size, shuffle=True)
            pbar = tqdm(dataloader)
            for i, batch in enumerate(pbar):
                raw_tokens, abcde = batch["tokens"].to(config.device), batch["abcde"].to(config.device)
                a, b, c, d = abcde[:,0], abcde[:,1], abcde[:,2], abcde[:,3]

                # Output of the regular token sequence
                raw_out = model(tokens=raw_tokens, output_attentions=True)
                raw_labels = torch.cat([
                    F.one_hot(a,n).view(-1,1,n),
                    (F.one_hot(a,n) + F.one_hot(b,n) + F.one_hot(c,n)).view(-1,1,n),
                    (F.one_hot(a,n) + F.one_hot(b,n) + F.one_hot(c,n) + F.one_hot(d,n)).view(-1,1,n),
                ], dim=1)

                # Now add the known adversarial rule form and run it through the model
                atk_ante = F.one_hot(a, num_classes=n).view(-1,1,n)
                atk_conseq = -1 * F.one_hot(b, num_classes=n).view(-1,1,n)
                atk_rule = torch.cat([atk_ante, atk_conseq], dim=-1)
                adv_tokens = torch.cat([atk_rule, raw_tokens], dim=1)
                adv_out = model(tokens=adv_tokens, output_attentions=True)
                adv_labels = batch["labels"].to(config.device)

                # Extract the relevant attention for the adversarial inputs
                _, adv_out2, _ = adv_out.all_seqcls_outputs
                adv_attn = adv_out2.attentions[0] # Extract from the 1-tuple; shape (N,1,L,L)
                adv_attn = adv_attn[:,0] # (N,L,L)
                top_attn_inds = adv_attn[:,-1].sort(dim=1, descending=True).indices # (N,L)

                # Now compute some metrics
                num_dones += raw_tokens.size(0)
                raw_pred = (raw_out.logits > 0).long()
                cum_raw_hits += ((raw_pred == raw_labels).sum(dim=(-1,-2)) == 3 * n).sum()
                raw_acc = cum_raw_hits / num_dones

                adv_pred = (adv_out.logits > 0).long()
                cum_adv_hits += ((adv_pred == adv_labels).sum(dim=(-1,-2)) == 3 * n).sum()
                adv_acc = cum_adv_hits / num_dones

                cum_top3_hits += (top_attn_inds[:,:3] == 0).sum()
                top3_acc = cum_top3_hits / num_dones

                cum_adv_weight += adv_attn[:,-1,0].sum()
                avg_adv_weight = cum_adv_weight / num_dones
                rel_adv_weight = avg_adv_weight * adv_tokens.size(1)

                desc_str = f"n {n}, d {embed_dim}, N {num_dones}: "
                desc_str += f"raw_acc {raw_acc:.3f}, adv_acc {adv_acc:.3f}, "
                desc_str += f"adv_top3 {top3_acc:.3f}, adv_wt {avg_adv_weight:.3f} "
                desc_str += f"({rel_adv_weight:.3f})"
                pbar.set_description(desc_str)

            this_df = pd.DataFrame({
                "train_seed": train_seed,
                "num_vars": n,
                "embed_dim": embed_dim,
                "raw_acc": raw_acc.item(),
                "adv_acc": adv_acc.item(),
                "adv_weight": avg_adv_weight.item(),
                "adv_rel_weight": rel_adv_weight.item(),
            }, index=[df_idx])

            df_idx += 1

            df = pd.concat([df, this_df])
            df.to_csv(saveto_file)

            df_idx += 1

        print(f"Done with: trseed {train_seed}, n {n}, d {embed_dim}")


if __name__ == "__main__":
    parser = HfArgumentParser(TheoryAttackExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = Namespace(**json.load(open(args.config_file)))

    # Set up the configs
    config.output_dir = args.output_dir
    config.device = args.device

    if config.attack_name == "big_token":
        ret = run_big_token_attack(config)

    elif config.attack_name == "repeat_token":
        ret = run_repeat_token_attack(config)

    elif config.attack_name == "suppress_rule":
        ret = run_suppress_rule_attack(config)

    else:
        raise ValueError(f"Unknown attack name {config.attack_name}")



