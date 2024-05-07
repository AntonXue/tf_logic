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
        default = "cpu",
    )



def load_model_and_dataset(
    num_vars: int,
    embed_dim: int,
    train_seed: int,
    reasoner_type: str,
    reasoner_num_steps: int = 3
):
    model, dataset = load_model_and_dataset_from_big_grid(
        embed_dim = embed_dim,
        num_vars = num_vars,
        seed = train_seed
    )

    model.num_steps = reasoner_num_steps

    if reasoner_type == "theory":
        model = TheoryAutoregKStepsModel(
            num_vars = num_vars,
            num_steps = reasoner_num_steps
        )

    model.eval()
    return model, dataset


@torch.no_grad()
def run_coerce_state_attack(config):
    assert config.num_samples % config.batch_size == 0
    saveto_file = Path(config.output_dir, "theory_attack_coerce_state.csv")
    print(f"Will save to: {saveto_file}")

    df = pd.DataFrame(columns=[
        "reasoner_type", "train_seed", "num_vars", "embed_dim", "kappa_power", "elems_acc", "states_acc"
    ])
    df_idx = 0

    for reasoner_type in config.reasoner_types:
        for train_seed in config.train_seeds:
            for nd in config.nd_pairs:
                n, d = nd

                model, res_dataset = load_model_and_dataset(
                    num_vars = n,
                    embed_dim = d,
                    train_seed = train_seed,
                    reasoner_type = reasoner_type,
                    reasoner_num_steps = 1
                )

                model.eval().to(config.device)

                dataset = CoerceStateDataset(res_dataset, num_attack_tokens=1, dataset_len=config.num_samples)

                dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

                for kappa_power in config.kappa_powers:
                    kappa = torch.tensor(10. ** kappa_power)
                    num_dones, cum_elem_hits, cum_state_hits = 0, 0, 0
                    pbar = tqdm(dataloader)

                    for batch in pbar:
                        batch_tokens = batch["tokens"].to(config.device)
                        tgt_state = batch["labels"].to(config.device)

                        adv_ante = torch.zeros_like(tgt_state)
                        adv_conseq = tgt_state - kappa * (1 - tgt_state)
                        adv_tokens = torch.cat([adv_ante, adv_conseq], dim=-1).view(-1,1,2*n)
                        all_tokens = torch.cat([adv_tokens, batch_tokens], dim=1)

                        out = model(all_tokens)
                        pred = (out.logits[:,0] > 0).long()

                        num_dones += batch_tokens.size(0)
                        cum_elem_hits += (tgt_state == pred).float().mean(dim=1).sum()
                        elems_acc = cum_elem_hits / num_dones
                        cum_state_hits += ((tgt_state == pred).sum(dim=1) == n).sum()
                        states_acc = cum_state_hits / num_dones

                        desc = f"{reasoner_type}, "
                        desc += f"n {n}, d {d}, log(kappa) {kappa_power:.3f}, "
                        desc += f"N {num_dones}, acc ({elems_acc:.3f}, {states_acc:.3f})"
                        pbar.set_description(desc)

                    this_df = pd.DataFrame({
                        "reasoner_type": reasoner_type,
                        "train_seed": train_seed,
                        "num_vars": n,
                        "embed_dim": d,
                        "kappa_power": kappa_power,
                        "elems_acc": elems_acc.item(),
                        "states_acc": states_acc.item(),
                    }, index=[df_idx])

                    df = pd.concat([df, this_df])
                    df.to_csv(saveto_file)
                    df_idx += 1

                print(f"Done with: trseed {train_seed}, n {n}, d {d}")


@torch.no_grad()
def run_suppress_rule_attack(config):
    assert config.num_samples % config.batch_size == 0
    saveto_file = Path(config.output_dir, "theory_attack_suppress_rule.csv")
    print(f"Will save to: {saveto_file}")

    df = pd.DataFrame(columns=[
        "reasoner_type",
        "train_seed", "num_vars", "embed_dim", "raw_state_acc", "adv_state_acc",
        "adv_top3", "adv_weight", "adv_rel_weight"
    ])
    df_idx = 0
    for reasoner_type in config.reasoner_types:
        for train_seed in config.train_seeds:
            for nd_pair in config.nd_pairs:
                n, embed_dim = nd_pair

                model, res_dataset = load_model_and_dataset(
                    num_vars = n,
                    embed_dim = embed_dim,
                    train_seed = train_seed,
                    reasoner_type = reasoner_type,
                    reasoner_num_steps = 1
                )

                model.eval().to(config.device)
                atk_dataset = SuppressRuleDataset(
                    reasoner_dataset = res_dataset,
                    dataset_len = config.num_samples,
                )

                num_dones = 0
                cum_raw_elems_hits, cum_raw_state_hits = 0, 0
                cum_adv_elems_hits, cum_adv_state_hits = 0, 0
                cum_top3_hits = 0
                cum_adv_weight = 0

                dataloader = DataLoader(atk_dataset, batch_size=config.batch_size, shuffle=True)
                pbar = tqdm(dataloader)
                for i, batch in enumerate(pbar):
                    adv_labels = batch["labels"].to(config.device)
                    raw_tokens = batch["tokens"].to(config.device)
                    abcde = batch["abcde"].to(config.device)
                    a, b, c, d, e = abcde.chunk(5, dim=-1)

                    # Output of the raw token sequence (i.e., without attacks)
                    raw_labels = torch.cat([
                        (F.one_hot(a,n) + F.one_hot(b,n) + F.one_hot(c,n)).view(-1,1,n),
                    ], dim=1)
                    raw_out = model(tokens=raw_tokens, output_attentions=True)
                    raw_pred = (raw_out.logits > 0).long()

                    # Now add the known adversarial rule form and run it through the model
                    atk_rule = torch.cat([F.one_hot(a,n), -1*F.one_hot(b,n)], dim=-1)
                    adv_tokens = torch.cat([atk_rule.view(-1,1,2*n), raw_tokens], dim=1)
                    adv_out = model(tokens=adv_tokens, output_attentions=True)
                    adv_pred = (adv_out.logits > 0).long()

                    # Extract the relevant attention for the adversarial inputs
                    if reasoner_type == "learned":
                        adv_out = adv_out.all_seqcls_outputs[0]
                        adv_attn = adv_out.attentions[0] # Extract from the 1-tuple; shape (N,1,L,L)
                        adv_attn = adv_attn[:,0] # (N,L,L)

                    elif reasoner_type == "theory":
                        adv_attn = adv_out.attentions[0]

                    top_attn_inds = adv_attn[:,-1].sort(dim=1, descending=True).indices # (N,L)

                    # Now compute some metrics
                    num_dones += raw_tokens.size(0)

                    cum_raw_elems_hits += (raw_pred == raw_labels).float().mean(dim=(1,2)).sum()
                    raw_elems_acc = cum_raw_elems_hits / num_dones

                    cum_raw_state_hits += ((raw_pred == raw_labels).float().mean(dim=(-1,-2)) > 0.999).sum()
                    raw_state_acc = cum_raw_state_hits / num_dones

                    cum_adv_elems_hits += (adv_pred == adv_labels).float().mean(dim=(1,2)).sum()
                    adv_elems_acc = cum_adv_elems_hits / num_dones

                    cum_adv_state_hits += ((adv_pred == adv_labels).float().mean(dim=(-1,-2)) > 0.999).sum()
                    adv_state_acc = cum_adv_state_hits / num_dones

                    cum_top3_hits += (top_attn_inds[:,:3] == 0).sum()
                    top3_acc = cum_top3_hits / num_dones

                    cum_adv_weight += adv_attn[:,-1,0].sum()
                    avg_adv_weight = cum_adv_weight / num_dones
                    rel_adv_weight = avg_adv_weight * adv_tokens.size(1)

                    desc = f"{reasoner_type}, "
                    desc += f"n {n}, d {embed_dim}, N {num_dones}: "
                    desc += f"raw ({raw_elems_acc:.3f}, {raw_state_acc:.3f}), "
                    desc += f"adv ({adv_elems_acc:.3f}, {adv_state_acc:.3f}), "
                    desc += f"adv_top3 {top3_acc:.3f}, adv_wt {avg_adv_weight:.3f} "
                    desc += f"({rel_adv_weight:.3f})"
                    pbar.set_description(desc)

                this_df = pd.DataFrame({
                    "reasoner_type": reasoner_type,
                    "train_seed": train_seed,
                    "num_vars": n,
                    "embed_dim": embed_dim,
                    "raw_state_acc": raw_state_acc.item(),
                    "adv_state_acc": adv_state_acc.item(),
                    "adv_top3": top3_acc.item(),
                    "adv_weight": avg_adv_weight.item(),
                    "adv_rel_weight": rel_adv_weight.item(),
                }, index=[df_idx])

                df = pd.concat([df, this_df])
                df.to_csv(saveto_file)
                df_idx += 1

            print(f"Done with: trseed {train_seed}, n {n}, d {embed_dim}")



@torch.no_grad()
def run_knowledge_amnesia_attack(config):
    assert config.num_samples % config.batch_size == 0
    saveto_file = Path(config.output_dir, "theory_attack_coerce_state.csv")
    print(f"Will save to: {saveto_file}")

    df = pd.DataFrame(columns=[
        "reasoner_type", "train_seed", "num_vars", "embed_dim",
        "raw_state_acc", "adv_state_acc",
    ])
    df_idx = 0
    for reasoner_type in config.reasoner_types:
        for train_seed in config.train_seeds:
            for nd_pair in config.nd_pairs:
                n, embed_dim = nd_pair

                model, res_dataset = load_model_and_dataset(
                    num_vars = n,
                    embed_dim = embed_dim,
                    train_seed = train_seed,
                    reasoner_type = reasoner_type,
                    reasoner_num_steps = 1
                )

                model.eval().to(config.device)
                atk_dataset = KnowledgeAmnesiaDataset(
                    reasoner_dataset = res_dataset,
                    dataset_len = config.num_samples,
                )

                num_dones = 0
                cum_raw_elems_hits, cum_raw_state_hits = 0, 0
                cum_adv_elems_hits, cum_adv_state_hits = 0, 0

                dataloader = DataLoader(atk_dataset, batch_size=config.batch_size, shuffle=True)
                pbar = tqdm(dataloader)
                for i, batch in enumerate(pbar):
                    adv_labels = batch["labels"].to(config.device)
                    raw_tokens = batch["tokens"].to(config.device)
                    abcde = batch["abcde"].to(config.device)
                    a, b, c, d, e = abcde.chunk(5, dim=-1)

                    # Output of the raw token sequence (i.e., without attacks)
                    raw_labels = torch.cat([
                        (F.one_hot(a,n) + F.one_hot(b,n) + F.one_hot(c,n)).view(-1,1,n),
                    ], dim=1)
                    raw_out = model(tokens=raw_tokens)
                    raw_pred = (raw_out.logits > 0).long()

                    # Now the adversarial stuff
                    atk_tokens = torch.cat([F.one_hot(a,n), -1*F.one_hot(a,n)], dim=-1).view(-1,1,2*n)
                    atk_tokens = atk_tokens.repeat(1,100,1)
                    adv_tokens = torch.cat([atk_tokens, raw_tokens], dim=1)
                    adv_out = model(tokens=adv_tokens)
                    adv_pred = (adv_out.logits > 0).long()

                    # Compute some statistics
                    num_dones += raw_tokens.size(0)
                    cum_raw_elems_hits += (raw_pred == raw_labels).float().mean(dim=(1,2)).sum()
                    raw_elems_acc = cum_raw_elems_hits / num_dones

                    cum_raw_state_hits += ((raw_pred == raw_labels).float().mean(dim=(-1,-2)) > 0.999).sum()
                    raw_state_acc = cum_raw_state_hits / num_dones

                    cum_adv_elems_hits += (adv_pred == adv_labels).float().mean(dim=(1,2)).sum()
                    adv_elems_acc = cum_adv_elems_hits / num_dones

                    cum_adv_state_hits += ((adv_pred == adv_labels).float().mean(dim=(-1,-2)) > 0.999).sum()
                    adv_state_acc = cum_adv_state_hits / num_dones

                    desc = f"{reasoner_type}, "
                    desc += f"n {n}, d {embed_dim}, N {num_dones}: "
                    desc += f"raw ({raw_elems_acc:.3f}, {raw_state_acc:.3f}), "
                    desc += f"adv ({adv_elems_acc:.3f}, {adv_state_acc:.3f}), "
                    pbar.set_description(desc)

                this_df = pd.DataFrame({
                    "reasoner_type": reasoner_type,
                    "train_seed": train_seed,
                    "num_vars": n,
                    "embed_dim": embed_dim,
                    "raw_state_acc": raw_state_acc.item(),
                    "adv_state_acc": adv_state_acc.item()
                }, index=[df_idx])

                df = pd.concat([df, this_df])
                df.to_csv(saveto_file)
                df_idx += 1


if __name__ == "__main__":
    parser = HfArgumentParser(TheoryAttackExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = Namespace(**json.load(open(args.config_file)))

    # Set up the configs
    config.output_dir = args.output_dir
    config.device = args.device

    if config.attack_name == "coerce_state":
        ret = run_coerce_state_attack(config)

    elif config.attack_name == "suppress_rule":
        ret = run_suppress_rule_attack(config)

    elif config.attack_name == "knowledge_amnesia":
        ret = run_knowledge_amnesia_attack(config)

    else:
        raise ValueError(f"Unknown attack name {config.attack_name}")



