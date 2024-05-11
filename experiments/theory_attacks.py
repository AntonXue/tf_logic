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

def hot(i, n):
    return F.one_hot(i, n)

def load_model_and_dataset(
    num_vars: int,
    embed_dim: int,
    train_seed: int,
    reasoner_type: str,
    num_reasoner_steps: int = 3,    # Default number of steps
):
    model, dataset = load_model_and_dataset_from_big_grid(
        embed_dim = embed_dim,
        num_vars = num_vars,
        seed = train_seed
    )

    if num_reasoner_steps is not None:
        model.num_steps = num_reasoner_steps

    if reasoner_type == "theory":
        model = TheoryAutoregKStepsModel(
            num_vars = num_vars,
            num_steps = model.num_steps,
        )

    model.eval()
    return model, dataset


@torch.no_grad()
def run_theory_attack_common(config):
    """ We can only do this because there's lots of similarities between the
        suppress rule and amnesia attacks
    """
    assert config.attack_name in ["suppress_rule", "knowledge_amnesia"]
    assert config.num_samples % config.batch_size == 0

    common_col_names = [
        "reasoner_type", "train_seed", "num_vars", "embed_dim", "num_repeats",
        "raw_state_acc",
        "adv_ns1_state_acc", "adv_ns2_state_acc", "adv_ns3_state_acc",
        "adv_ns1_attn_wts", "adv_ns2_attn_wts", "adv_ns3_attn_wts"
    ]

    if config.attack_name == "suppress_rule":
        saveto_file = Path(config.output_dir, "theory_attack_suppress_rule.csv")
        df = pd.DataFrame(columns=common_col_names + [
            "adv_ns1_attn_ratio", "adv_ns2_attn_ratio", "adv_ns3_attn_ratio"
        ])
    elif config.attack_name == "knowledge_amnesia":
        saveto_file = Path(config.output_dir, "theory_attack_knowledge_amnesia.csv")
        df = pd.DataFrame(columns=common_col_names)
    else:
        raise ValueError(f"Unknown config.attack_name {config.attack_name}")

    print(f"Will save to: {saveto_file}")

    for train_seed in config.train_seeds:
        for reasoner_type in config.reasoner_types:
            for (n, embed_dim) in config.nd_pairs:
                res_model, res_dataset = load_model_and_dataset(
                    num_vars = n,
                    embed_dim = embed_dim,
                    train_seed = train_seed,
                    reasoner_type = reasoner_type,
                )

                res_model.eval().to(config.device)

                if config.attack_name == "suppress_rule":
                    atk_dataset = SuppressRuleDataset(res_dataset, dataset_len=config.num_samples)
                elif config.attack_name == "knowledge_amnesia":
                    atk_dataset = KnowledgeAmnesiaDataset(res_dataset, dataset_len=config.num_samples)

                dataloader = DataLoader(atk_dataset, batch_size=config.batch_size, shuffle=True)

                for k in config.num_repeats:
                    num_dones = 0
                    raw_elems_hits, raw_state_hits = 0, 0
                    adv_ns1_elems_hits, adv_ns1_state_hits = 0, 0
                    adv_ns2_elems_hits, adv_ns2_state_hits = 0, 0
                    adv_ns3_elems_hits, adv_ns3_state_hits = 0, 0
                    adv_ns1_attn_wts, adv_ns2_attn_wts, adv_ns3_attn_wts = \
                        torch.tensor([]), torch.tensor([]), torch.tensor([])
                    adv_ns1_suppd_wts, adv_ns2_suppd_wts, adv_ns3_suppd_wts = \
                        torch.tensor([]), torch.tensor([]), torch.tensor([])

                    pbar = tqdm(dataloader)
                    for i, batch in enumerate(pbar):
                        adv_labels = batch["labels"].to(config.device)
                        raw_tokens = batch["tokens"].to(config.device)
                        infos = batch["infos"].to(config.device)
                        a, b, c, d, e, f, g, h = infos.chunk(8, dim=-1)

                        # Output of the raw token sequence (i.e., without attacks)
                        raw_labels = torch.cat([
                            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n),
                            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
                            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n) + hot(g,n)
                        ], dim=1)

                        raw_out = res_model(tokens=raw_tokens, output_attentions=True)
                        raw_pred = (raw_out.logits > 0).long()

                        # Now add the known adversarial rule form and run it through the model
                        if config.attack_name == "suppress_rule":
                            atk_rule = torch.cat([F.one_hot(a,n), -1*F.one_hot(b,n)], dim=-1)
                            atk_rules = atk_rule.view(-1,1,2*n).repeat(1,k,1)
                        elif config.attack_name == "knowledge_amnesia":
                            r = raw_tokens.size(1)
                            atk_rule = torch.cat([F.one_hot(a,n), -r*F.one_hot(a,n)], dim=-1)
                            atk_rules = atk_rule.view(-1,1,2*n).repeat(1,k,1)

                        adv_tokens = torch.cat([atk_rules, raw_tokens], dim=1)
                        adv_out = res_model(tokens=adv_tokens, output_attentions=True)
                        adv_pred = (adv_out.logits > 0).long()

                        # Now compute some metrics
                        num_dones += raw_tokens.size(0)
                        all_raw_hits = raw_pred == raw_labels   # (N,3,n)
                        raw_elems_hits += all_raw_hits.float().mean(dim=(1,2)).sum()
                        raw_state_hits += all_raw_hits.all(dim=-1).all(dim=-1).sum()
                        raw_elems_acc = raw_elems_hits / num_dones
                        raw_state_acc = raw_state_hits / num_dones

                        all_adv_hits = adv_pred == adv_labels   # (N,3,n)
                        adv_ns1_elems_hits += all_adv_hits[:,0:1].float().mean(dim=(1,2)).sum()
                        adv_ns1_state_hits += all_adv_hits[:,0:1].all(dim=-1).all(dim=-1).sum()
                        adv_ns1_elems_acc = adv_ns1_elems_hits / num_dones
                        adv_ns1_state_acc = adv_ns1_state_hits / num_dones

                        adv_ns2_elems_hits += all_adv_hits[:,0:2].float().mean(dim=(1,2)).sum()
                        adv_ns2_state_hits += all_adv_hits[:,0:2].all(dim=-1).all(dim=-1).sum()
                        adv_ns2_elems_acc = adv_ns2_elems_hits / num_dones
                        adv_ns2_state_acc = adv_ns2_state_hits / num_dones

                        adv_ns3_elems_hits += all_adv_hits[:,0:3].float().mean(dim=(1,2)).sum()
                        adv_ns3_state_hits += all_adv_hits[:,0:3].all(dim=-1).all(dim=-1).sum()
                        adv_ns3_elems_acc = adv_ns3_elems_hits / num_dones
                        adv_ns3_state_acc = adv_ns3_state_hits / num_dones

                        # Attention metrics
                        if reasoner_type == "learned":
                            adv_out1, adv_out2, adv_out3 = adv_out.all_seqcls_outputs
                            adv_attn1 = adv_out1.attentions[0][:,0] # (N, r+k, r+k)
                            adv_attn2 = adv_out2.attentions[0][:,0] # (N, r+k+1, r+k+1)
                            adv_attn3 = adv_out3.attentions[0][:,0] # (N, r+k+2, r+k+2)

                        elif reasoner_type == "theory":
                            adv_attn1, adv_attn2, adv_attn3 = adv_out.attentions

                        # Cumulative attention weight of the attack tokens
                        adv_ns1_attn_wts = torch.cat([adv_ns1_attn_wts, adv_attn1[:,-1,:k].sum(dim=-1)])
                        adv_ns2_attn_wts = torch.cat([adv_ns2_attn_wts, adv_attn2[:,-1,:k].sum(dim=-1)])
                        adv_ns3_attn_wts = torch.cat([adv_ns3_attn_wts, adv_attn3[:,-1,:k].sum(dim=-1)])

                        if config.attack_name == "suppress_rule":
                            supp_idx = batch["supp_idx"]
                            adv_ns1_suppd_wts = torch.cat([
                                adv_ns1_suppd_wts,
                                adv_attn1[:,-1].gather(1, k+supp_idx.view(-1,1)).view(-1)
                            ])

                            adv_ns2_suppd_wts = torch.cat([
                                adv_ns2_suppd_wts,
                                adv_attn2[:,-1].gather(1, k+supp_idx.view(-1,1)).view(-1)
                            ])

                            adv_ns3_suppd_wts = torch.cat([
                                adv_ns3_suppd_wts,
                                adv_attn3[:,-1].gather(1, k+supp_idx.view(-1,1)).view(-1)
                            ])

                        desc = f"{reasoner_type} "
                        desc += f"ndk ({n},{embed_dim},{k}), N {num_dones}: "
                        desc += f"raw ({raw_elems_acc:.2f},{raw_state_acc:.2f}), "
                        desc += f"adv ({adv_ns1_elems_acc:.2f},{adv_ns1_state_acc:.2f} # "
                        desc += f"{adv_ns2_elems_acc:.2f},{adv_ns2_state_acc:.2f} # "
                        desc += f"{adv_ns3_elems_acc:.2f},{adv_ns3_state_acc:.2f}), "
                        desc += f"attn "
                        desc += f"({adv_ns1_attn_wts.mean().item():.2f},"
                        desc += f"{adv_ns2_attn_wts.mean().item():.2f},"
                        desc += f"{adv_ns3_attn_wts.mean().item():.2f})"
                        pbar.set_description(desc)
                        # End inner for loop

                    # Things to save that are common to all attacks
                    save_dict = {
                        "reasoner_type": reasoner_type,
                        "train_seed": train_seed,
                        "num_vars": n,
                        "embed_dim": embed_dim,
                        "num_repeats": k,
                        "raw_state_acc": raw_state_acc.item(),
                        "adv_ns1_state_acc": adv_ns1_state_acc.item(),
                        "adv_ns2_state_acc": adv_ns2_state_acc.item(),
                        "adv_ns3_state_acc": adv_ns3_state_acc.item(),
                        "adv_ns1_attn_wts": adv_ns1_attn_wts.mean().item(),
                        "adv_ns2_attn_wts": adv_ns2_attn_wts.mean().item(),
                        "adv_ns3_attn_wts": adv_ns3_attn_wts.mean().item(),
                    }

                    # Attack-specific additions
                    if config.attack_name == "suppress_rule":
                        other_dict = {
                            "adv_ns1_attn_ratio": (adv_ns1_attn_wts/adv_ns1_suppd_wts).mean().item(),
                            "adv_ns2_attn_ratio": (adv_ns2_attn_wts/adv_ns2_suppd_wts).mean().item(),
                            "adv_ns3_attn_ratio": (adv_ns3_attn_wts/adv_ns3_suppd_wts).mean().item(),
                        }
                        save_dict = save_dict | other_dict

                    this_df = pd.DataFrame(save_dict, index=[0])
                    df = pd.concat([df, this_df], ignore_index=True)
                    df.to_csv(saveto_file, index=False)


@torch.no_grad()
def run_coerce_state_attack(config):
    assert config.num_samples % config.batch_size == 0
    saveto_file = Path(config.output_dir, "theory_attack_coerce_state.csv")
    print(f"Will save to: {saveto_file}")

    df = pd.DataFrame(columns=[
        "reasoner_type", "train_seed", "num_vars", "embed_dim", "kappa_power", "elems_acc", "states_acc"
    ])

    for train_seed in config.train_seeds:
        for reasoner_type in config.reasoner_types:
            for nd in config.nd_pairs:
                n, d = nd
                res_model, res_dataset = load_model_and_dataset(
                    num_vars = n,
                    embed_dim = d,
                    train_seed = train_seed,
                    reasoner_type = reasoner_type,
                    num_reasoner_steps = 1
                )

                res_model.eval().to(config.device)
                dataset = CoerceStateDataset(res_dataset, 1, config.num_samples)
                dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

                for kappa_power in config.kappa_powers:
                    kappa = torch.tensor(10. ** kappa_power)
                    num_dones, elem_hits, state_hits = 0, 0, 0
                    pbar = tqdm(dataloader)

                    for batch in pbar:
                        batch_tokens = batch["tokens"].to(config.device)
                        tgt_state = batch["labels"].to(config.device)

                        atk_ante = torch.zeros_like(tgt_state)
                        atk_conseq = tgt_state - kappa * (1 - tgt_state)
                        atk_rule = torch.cat([atk_ante, atk_conseq], dim=-1).view(-1,1,2*n)
                        all_tokens = torch.cat([atk_rule, batch_tokens], dim=1)

                        out = res_model(all_tokens)
                        pred = (out.logits[:,0] > 0).long()

                        num_dones += batch_tokens.size(0)
                        elem_hits += (tgt_state == pred).float().mean(dim=1).sum()
                        elems_acc = elem_hits / num_dones
                        state_hits += ((tgt_state == pred).sum(dim=1) == n).sum()
                        states_acc = state_hits / num_dones

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
                    })

                    df = pd.concat([df, this_df], ignore_index=True)
                    df.to_csv(saveto_file)


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

    elif config.attack_name in ["suppress_rule", "knowledge_amnesia"]:
        ret = run_theory_attack_common(config)

    else:
        raise ValueError(f"Unknown attack name {config.attack_name}")



