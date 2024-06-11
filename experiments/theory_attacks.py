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
        model = TheoryAutoregKStepsModel(num_vars=num_vars, num_steps=model.num_steps)

    model.eval()
    return model, dataset


@torch.no_grad()
def run_coerce_state_variance_attack(config):
    """ Coerce state attack gets its own thing """
    assert config.attack_name == "coerce_state_variance"
    device = config.device
    bsz = config.batch_size

    saveto_file = Path(config.output_dir, "theory_coerce_state_variance.csv")
    df = pd.DataFrame(columns=[
        "num_samples",
        "reasoner_type", "train_seed", "num_vars", "embed_dim", "num_copies",
        "variance", "variance_std"
    ])


    for train_seed in config.train_seeds:
        for reasoner_type in config.reasoner_types:
            for (n, embed_dim) in config.nd_pairs:
                res_model, res_dataset = load_model_and_dataset(
                    num_vars = n,
                    embed_dim = embed_dim,
                    train_seed = train_seed,
                    reasoner_type = reasoner_type,
                )

                res_model.eval().to(device)
                num_samples = config.num_samples
                atk_dataset = CoerceStateDataset(res_dataset, 4, num_samples)
                L = atk_dataset[0]["tokens"].size(0)
                hotp = atk_dataset.hot_prob

                for k in config.num_copies:
                    pbar = tqdm(range(num_samples))
                    variances = torch.tensor([]).to(device)
                    for idx in pbar:
                        item = atk_dataset[idx]

                        # Make the attack suffix
                        infos = item["infos"].to(device)
                        targets = item["hints"].to(device)
                        tgt1, tgt2, tgt3 = targets.chunk(3, dim=0) # (1,2n) each
                        tgt0 = (torch.rand(1,n) < hotp).long().to(device)

                        atk_rule1 = torch.cat([tgt0, 2*L*(tgt1-0.5)], dim=-1)
                        atk_rule2 = torch.cat([tgt1, 2*L*(tgt2-0.5)], dim=-1)
                        atk_rule3 = torch.cat([tgt2, 2*L*(tgt3-0.5)], dim=-1)
                        atk_rules = torch.cat([atk_rule1, atk_rule2, atk_rule3], dim=0).repeat(k,1)
                        init_token = torch.cat([torch.zeros_like(tgt0), 2*L*(tgt0-0.5)], dim=-1)
                        adv_suffix = torch.cat([atk_rules, init_token], dim=0)

                        # Make the other tokens and query the model
                        other_tokens = (torch.rand(bsz, *item["tokens"].shape) < hotp).long().to(device)
                        adv_tokens = torch.cat([
                            other_tokens,
                            adv_suffix.view(1,-1,2*n).repeat(bsz,1,1)
                        ], dim=1)

                        adv_out = res_model(tokens=adv_tokens)
                        adv_pred = (adv_out.logits > 0).long()

                        variances = torch.cat([variances, adv_pred.float().var(dim=0).mean().view(1)])

                        # Some stats
                        desc = f"{reasoner_type} ndk ({n},{embed_dim},{k}), var {variances.mean():.3e}"
                        pbar.set_description(desc)

                    # Done with this combination of (n,d,k), so save
                    save_dict = {
                        "num_samples": num_samples,
                        "reasoner_type": reasoner_type,
                        "train_seed": train_seed,
                        "num_vars": n,
                        "embed_dim": embed_dim,
                        "num_copies": k,
                        "variance": variances.mean().item(),
                        "variance_std": variances.var().sqrt().item()
                    }

                    this_df = pd.DataFrame(save_dict, index=[0])
                    df = pd.concat([df, this_df], ignore_index=True)
                    df.to_csv(saveto_file, index=False)


@torch.no_grad()
def run_theory_attack_common(config):
    """ We can only do this because there's lots of similarities between the
        suppress rule and amnesia attacks
    """
    assert config.attack_name in ["suppress_rule", "fact_amnesia", "coerce_state"]
    assert config.num_samples % config.batch_size == 0

    common_col_names = [
        "num_samples",
        "reasoner_type", "train_seed", "num_vars", "embed_dim", "num_attack_tokens",
        "raw_state_acc",
        "adv_ns1_state_acc", "adv_ns2_state_acc", "adv_ns3_state_acc",
        "adv_ns1_atk_wts", "adv_ns2_atk_wts", "adv_ns3_atk_wts",
        "adv_ns1_atk_wts_std", "adv_ns2_atk_wts_std", "adv_ns3_atk_wts_std"
    ]

    if config.attack_name == "suppress_rule":
        saveto_file = Path(config.output_dir, "theory_suppress_rule.csv")
        df = pd.DataFrame(columns=common_col_names + [
            "adv_ns1_suppd_wts", "adv_ns2_suppd_wts", "adv_ns3_suppd_wts",
            "adv_ns1_other_wts", "adv_ns2_other_wts", "adv_ns3_other_wts",
            "adv_ns1_suppd_wts_std", "adv_ns2_suppd_wts_std", "adv_ns3_suppd_wts_std",
            "adv_ns1_other_wts_std", "adv_ns2_other_wts_std", "adv_ns3_other_wts_std",
        ])

    elif config.attack_name == "fact_amnesia":
        saveto_file = Path(config.output_dir, "theory_fact_amnesia.csv")
        df = pd.DataFrame(columns=common_col_names)

    elif config.attack_name == "coerce_state":
        saveto_file = Path(config.output_dir, "theory_coerce_state.csv")
        df = pd.DataFrame(columns=common_col_names)

    else:
        raise ValueError(f"Unknown config.attack_name {config.attack_name}")

    device = config.device

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

                res_model.eval().to(device)

                for k in config.num_attack_tokens:
                    # We may repeat k-1 times
                    if config.attack_name == "suppress_rule":
                        atk_dataset = SuppressRuleDataset(res_dataset, k, config.num_samples)
                    elif config.attack_name == "fact_amnesia":
                        atk_dataset = FactAmnesiaDataset(res_dataset, k, config.num_samples)
                    elif config.attack_name == "coerce_state":
                        atk_dataset = CoerceStateDataset(res_dataset, 4, config.num_samples)

                    dataloader = DataLoader(atk_dataset, batch_size=config.batch_size)

                    raw_elems_hits, raw_state_hits = torch.tensor([]).to(device), torch.tensor([]).to(device)
                    adv_ns1_elems_hits, adv_ns1_state_hits = torch.tensor([]).to(device), torch.tensor([]).to(device)
                    adv_ns2_elems_hits, adv_ns2_state_hits = torch.tensor([]).to(device), torch.tensor([]).to(device)
                    adv_ns3_elems_hits, adv_ns3_state_hits = torch.tensor([]).to(device), torch.tensor([]).to(device)

                    adv_ns1_atk_wts, adv_ns2_atk_wts, adv_ns3_atk_wts = \
                        torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
                    adv_ns1_suppd_wts, adv_ns2_suppd_wts, adv_ns3_suppd_wts = \
                        torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
                    adv_ns1_other_wts, adv_ns2_other_wts, adv_ns3_other_wts = \
                        torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)

                    pbar = tqdm(dataloader)
                    for i, batch in enumerate(pbar):
                        raw_tokens = batch["tokens"].to(device)
                        N, L = raw_tokens.size(0), raw_tokens.size(1)

                        adv_labels = batch["labels"].to(device)
                        if adv_labels.size(1) == 4:
                            adv_labels = adv_labels[:,1:]
                        assert adv_labels.shape == (N,3,n)  # (N,3,n)

                        infos = batch["infos"].to(device)
                        a, b, c, d, e, f, g, h = infos.chunk(infos.size(-1), dim=-1)

                        # Output of the raw token sequence (i.e., without attacks)
                        raw_labels = torch.cat([
                            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n),
                            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
                            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n) + hot(g,n)
                        ], dim=1)

                        raw_out = res_model(tokens=raw_tokens, output_attentions=True)
                        raw_pred = (raw_out.logits > 0).long()

                        # Now add the known adversarial rule form and run it through the model
                        init_token = torch.cat([torch.zeros(N,1,n).to(device), hot(a,n)], dim=-1) # (N,2n)

                        if config.attack_name == "suppress_rule":
                            # Try to suppress the rule c,d -> f
                            atk_rule = torch.cat([hot(c,n) + hot(d,n), -1*hot(f,n)], dim=-1)
                            adv_suffix = torch.cat([
                                atk_rule.view(N,1,2*n).repeat(1,k-1,1),
                                init_token.view(N,1,-1)
                            ], dim=1)

                        elif config.attack_name == "fact_amnesia":
                            atk_rule = torch.cat([torch.zeros(N,1,n).to(device), -L*hot(a,n)], dim=-1)
                            adv_suffix = torch.cat([
                                atk_rule.view(N,1,-1).repeat(1,k-1,1),
                                init_token.view(N,1,-1)
                            ], dim=1)

                        elif config.attack_name == "coerce_state":
                            tgt1, tgt2, tgt3 = batch["hints"].chunk(3, dim=1)
                            tgt0 = hot(a,n).view(-1,1,n)
                            atk_rule1 = torch.cat([tgt0, L*(tgt1 - 0.5)], dim=-1)
                            atk_rule2 = torch.cat([tgt1, L*(tgt2 - 0.5)], dim=-1)
                            atk_rule3 = torch.cat([tgt2, L*(tgt3 - 0.5)], dim=-1)

                            adv_suffix = torch.cat([
                                atk_rule1,
                                atk_rule2,
                                atk_rule3,
                                init_token.view(N,1,-1)
                            ], dim=1)


                        adv_tokens = torch.cat([raw_tokens, adv_suffix], dim=1)
                        adv_out = res_model(tokens=adv_tokens, output_attentions=True)
                        adv_pred = (adv_out.logits > 0).long()

                        # Now compute some metrics
                        all_raw_hits = raw_pred == raw_labels   # (N,3,n)
                        raw_elems_hits = torch.cat([raw_elems_hits, all_raw_hits.float().mean(dim=(1,2))])
                        raw_state_hits = torch.cat([raw_state_hits, all_raw_hits.all(dim=-1)]).float()

                        adv_hits = adv_pred == adv_labels   # (N,3,n)
                        adv_ns1_elems_hits = torch.cat([adv_ns1_elems_hits, adv_hits[:,0:1].float().mean(dim=(1,2))])
                        adv_ns2_elems_hits = torch.cat([adv_ns2_elems_hits, adv_hits[:,0:2].float().mean(dim=(1,2))])
                        adv_ns3_elems_hits = torch.cat([adv_ns3_elems_hits, adv_hits[:,0:3].float().mean(dim=(1,2))])

                        adv_ns1_state_hits = torch.cat([adv_ns1_state_hits, adv_hits[:,0:1].all(dim=-1).all(dim=-1)])
                        adv_ns2_state_hits = torch.cat([adv_ns2_state_hits, adv_hits[:,0:2].all(dim=-1).all(dim=-1)])
                        adv_ns3_state_hits = torch.cat([adv_ns3_state_hits, adv_hits[:,0:3].all(dim=-1).all(dim=-1)])

                        # Attention metrics
                        if reasoner_type == "learned":
                            adv_out1, adv_out2, adv_out3 = adv_out.all_seqcls_outputs
                            adv_attn1 = adv_out1.attentions[0][:,0] # (N, r+k, r+k)
                            adv_attn2 = adv_out2.attentions[0][:,0] # (N, r+k+1, r+k+1)
                            adv_attn3 = adv_out3.attentions[0][:,0] # (N, r+k+2, r+k+2)

                        elif reasoner_type == "theory":
                            adv_attn1, adv_attn2, adv_attn3 = adv_out.attentions

                        # Cumulative attention weight of the k attack tokens
                        adv_ns1_atk_wts = torch.cat([adv_ns1_atk_wts, adv_attn1[:,-1,L:L+k]])
                        adv_ns2_atk_wts = torch.cat([adv_ns2_atk_wts, adv_attn2[:,-1,L:L+k]])
                        adv_ns3_atk_wts = torch.cat([adv_ns3_atk_wts, adv_attn3[:,-1,L:L+k]])

                        if config.attack_name == "suppress_rule":
                            # Try to suppress the rule c,d -> f
                            suppd_idx = batch["cdf_index"].to(device)
                            other_idx = batch["bce_index"].to(device)
                            adv_ns1_suppd_wts = torch.cat([adv_ns1_suppd_wts, adv_attn1[:,-1].gather(1, suppd_idx.view(-1,1))])
                            adv_ns2_suppd_wts = torch.cat([adv_ns2_suppd_wts, adv_attn2[:,-1].gather(1, suppd_idx.view(-1,1))])
                            adv_ns3_suppd_wts = torch.cat([adv_ns3_suppd_wts, adv_attn3[:,-1].gather(1, suppd_idx.view(-1,1))])

                            adv_ns1_other_wts = torch.cat([adv_ns1_other_wts, adv_attn1[:,-1].gather(1, other_idx.view(-1,1))])
                            adv_ns2_other_wts = torch.cat([adv_ns2_other_wts, adv_attn2[:,-1].gather(1, other_idx.view(-1,1))])
                            adv_ns3_other_wts = torch.cat([adv_ns3_other_wts, adv_attn3[:,-1].gather(1, other_idx.view(-1,1))])

                        # Compute stats
                        desc = f"{reasoner_type} ndk ({n},{embed_dim},{k}): "
                        desc += f"raw ({raw_elems_hits.float().mean():.2f},{raw_state_hits.float().mean():.2f}), "

                        desc += f"adv ({adv_ns1_elems_hits.float().mean():.2f},{adv_ns1_state_hits.float().mean():.2f} # " \
                            + f"{adv_ns2_elems_hits.float().mean():.2f},{adv_ns2_state_hits.float().mean():.2f} # " \
                            + f"{adv_ns3_elems_hits.float().mean():.2f},{adv_ns3_state_hits.float().mean():.2f}), "

                        desc += f"atk (" \
                            + f"{adv_ns1_atk_wts.float().mean():.2f}," \
                            + f"{adv_ns2_atk_wts.float().mean():.2f}," \
                            + f"{adv_ns3_atk_wts.float().mean():.2f}), " \

                        if config.attack_name == "suppress_rule":
                            desc += f"suppd ({adv_ns1_suppd_wts.float().mean():.2f}," \
                                + f"{adv_ns2_suppd_wts.float().mean():.2f}," \
                                + f"{adv_ns3_suppd_wts.float().mean():.2f}), "

                            desc += f"other ({adv_ns1_other_wts.float().mean():.2f}," \
                                + f"{adv_ns2_other_wts.float().mean():.2f}," \
                                + f"{adv_ns3_other_wts.float().mean():.2f}), "

                        pbar.set_description(desc)
                        # End inner for loop

                    # Things to save that are common to all attacks
                    save_dict = {
                        "num_samples": config.num_samples,
                        "reasoner_type": reasoner_type,
                        "train_seed": train_seed,
                        "num_vars": n,
                        "embed_dim": embed_dim,
                        "num_attack_tokens": k,
                        "raw_state_acc": raw_state_hits.float().mean().item(),
                        "adv_ns1_state_acc": adv_ns1_state_hits.float().mean().item(),
                        "adv_ns2_state_acc": adv_ns2_state_hits.float().mean().item(),
                        "adv_ns3_state_acc": adv_ns3_state_hits.float().mean().item(),
                        "adv_ns1_atk_wts": adv_ns1_atk_wts.mean().item(),
                        "adv_ns2_atk_wts": adv_ns2_atk_wts.mean().item(),
                        "adv_ns3_atk_wts": adv_ns3_atk_wts.mean().item(),
                        "adv_ns1_atk_wts_std": adv_ns1_atk_wts.var().sqrt().item(),
                        "adv_ns2_atk_wts_std": adv_ns2_atk_wts.var().sqrt().item(),
                        "adv_ns3_atk_wts_std": adv_ns3_atk_wts.var().sqrt().item(),
                    }

                    # Attack-specific additions
                    if config.attack_name == "suppress_rule":
                        other_dict = {
                            "adv_ns1_suppd_wts": adv_ns1_suppd_wts.mean().item(),
                            "adv_ns2_suppd_wts": adv_ns2_suppd_wts.mean().item(),
                            "adv_ns3_suppd_wts": adv_ns3_suppd_wts.mean().item(),
                            "adv_ns1_other_wts": adv_ns1_other_wts.mean().item(),
                            "adv_ns2_other_wts": adv_ns2_other_wts.mean().item(),
                            "adv_ns3_other_wts": adv_ns3_other_wts.mean().item(),

                            "adv_ns1_suppd_wts_std": adv_ns1_suppd_wts.var().sqrt().item(),
                            "adv_ns2_suppd_wts_std": adv_ns2_suppd_wts.var().sqrt().item(),
                            "adv_ns3_suppd_wts_std": adv_ns3_suppd_wts.var().sqrt().item(),
                            "adv_ns1_other_wts_std": adv_ns1_other_wts.var().sqrt().item(),
                            "adv_ns2_other_wts_std": adv_ns2_other_wts.var().sqrt().item(),
                            "adv_ns3_other_wts_std": adv_ns3_other_wts.var().sqrt().item(),
                        }
                        save_dict = save_dict | other_dict

                    this_df = pd.DataFrame(save_dict, index=[0])
                    df = pd.concat([df, this_df], ignore_index=True)
                    df.to_csv(saveto_file, index=False)


if __name__ == "__main__":
    parser = HfArgumentParser(TheoryAttackExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = Namespace(**json.load(open(args.config_file)))

    # Set up the configs
    config.output_dir = args.output_dir
    config.device = args.device

    if config.attack_name == "coerce_state_variance":
        ret = run_coerce_state_variance_attack(config)

    elif config.attack_name in ["suppress_rule", "fact_amnesia", "coerce_state"]:
        ret = run_theory_attack_common(config)

    else:
        raise ValueError(f"Unknown attack name {config.attack_name}")



