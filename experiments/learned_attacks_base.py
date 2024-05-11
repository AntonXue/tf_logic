import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from transformers import Trainer, TrainingArguments, HfArgumentParser
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
import json

""" Our imports """
from common import *    # Definitions and path inserts, particularly for WANDB
from models import *
from my_datasets import *
from utils.model_loader_utils import *

""" Parser for Hugging Face """

@dataclass
class LearnedAttackExperimentsArguments:
    """ This class captures ALL the possible synthetic experiments we may want to run """
    output_dir: str = field(
        default = str(Path(DUMP_DIR, "learned_attacks")),
        metadata = {"help": "Output directory of synthetic experiments."}
    )

    attack_name: Optional[str] = field(
        default = None,
        metadata = {"help": "The name of the attack."}
    )

    """ Model details """

    num_vars: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of propositional variables to use."}
    )

    embed_dim: Optional[int] = field(
        default = None,
        metadata = {"help": "The reasoner model's embedding (i.e., hidden) dimension."}
    )

    num_attack_tokens: Optional[int] = field(
        default = None,
        metadata = {"help": "The reasoner model's embedding (i.e., hidden) dimension."}
    )

    attack_tokens_style: Optional[str] = field(
        default = "repeat",
        metadata = {"help": "The reasoner model's embedding (i.e., hidden) dimension."}
    )

    token_range: Optional[str] = field(
        default = None,
        metadata = {"help": "Unbounded or clamped to (0,1). We also binarize accordingly."}
    )

    """ Attack training details """

    train_len: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of elements in the training dataset."}
    )

    eval_len: Optional[int] = field(
        default = None,
        metadata = {"help": "The number of elements in the eval (i.e., validation) dataset."}
    )

    num_epochs: Optional[int] = field(
        default = 16,
        metadata = {"help": "The number of epochs for training."}
    )

    warmup_ratio: Optional[float] = field(
        default = 0.1,
        metadata = {"help": "Warmup ratio for training."}
    )

    learning_rate: Optional[float] = field(
        default = 1e-4,
        metadata = {"help": "Learning rate."}
    )

    batch_size: Optional[int] = field(
        default = 64,
        metadata = {"help": "The train batch size."}
    )

    reasoner_seed: Optional[int] = field(
        default = None,
        metadata = {"help": "The seed that the reasoner model was trained with"}
    )

    attacker_seed: Optional[int] = field(
        default = 1234,
        metadata = {"help": "RNG seed"}
    )

    logging_steps: int = field(
        default = 16,
        metadata = {"help": "How often the HF's Trainer logs."}
    )

    device: str = field(
        default = "cuda",
        metadata = {"help": "The device we run things on."}
    )

    reasoner_type: str = field(
        default = "learned",
        metadata = {"help": "The type (learned/theory) of reasoner that we use."}
    )

    only_run_eval: int = field(
        default = 0,
        metadata = {"help": (
            "Whether to only dump the statistics without training. "
            )
        }
    )


def get_info_strings(args: LearnedAttackExperimentsArguments):
    """ Figure out where to save things """
    run_name = f"{args.attack_name}"
    run_name += f"_{args.reasoner_type}"
    run_name += f"_n{args.num_vars}_d{args.embed_dim}"
    run_name += f"_k{args.num_attack_tokens}_{args.attack_tokens_style}"

    last_saveto = str(Path(args.output_dir, run_name + "_last.pt"))
    best_saveto = str(Path(args.output_dir, run_name + "_best.pt"))
    csv_saveto = str(Path(args.output_dir, f"learned_{args.attack_name}.csv"))
    return {
        "run_name": run_name,
        "last_saveto": last_saveto,
        "best_saveto": best_saveto,
        "csv_saveto": csv_saveto
    }


def append_to_csv(csv_file: str, row_dict: dict):
    """ Make a new CSV/DataFrame if doesn't exist, and append to it """
    # Python 3.7+ preserves dict keys by order of insertion
    # https://stackoverflow.com/a/60862/2704964
    col_names = list(row_dict.keys())

    if Path(csv_file).is_file():
        df = pd.read_csv(csv_file)
        assert sorted(list(col_names)) == sorted(list(df.columns.values)) # Columns match?
    # Otherwise need to make new file
    else:
        df = pd.DataFrame(columns=col_names)

    new_df = pd.DataFrame(row_dict, index=[0])
    df = pd.concat([df, new_df], ignore_index=True).drop_duplicates(
        subset = ["reasoner_type", "reasoner_seed", "num_vars", "embed_dim", "num_repeats"],
        keep = "last"
    )
    df.to_csv(csv_file, index=False)
    return df


def train_one_epoch(
    atk_model,
    dataloader,
    optimizer,
    lr_scheduler,
    args: LearnedAttackExperimentsArguments
):
    """ The train function for all attacks basically looks the same """
    optimizer.zero_grad()
    atk_model.train()
    device = next(atk_model.parameters()).device
    num_dones, all_losses = 0, []
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        tokens, infos, labels = \
            batch["tokens"].to(device), batch["infos"].to(device), batch["labels"].to(device)
        out = atk_model(tokens=tokens, infos=infos, labels=labels)

        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # Track stats
        num_dones += tokens.size(0)
        all_losses.append(loss.item())
        sma_loss = torch.tensor(all_losses[-16:]).mean().item() # Smooth-moving average
        lr = lr_scheduler.get_last_lr()[0]

        desc = "[train] "
        desc += f"N {num_dones}, lr {lr:.6f}, loss {sma_loss:.4f} (first {all_losses[0]:.4f}"
        pbar.set_description(desc)

    return {
        "loss": sma_loss,
        "all_losses": all_losses
    }


@torch.no_grad()
def eval_one_epoch(
    atk_model,
    dataloader,
    args: LearnedAttackExperimentsArguments,
    do_save: bool = False
):
    atk_model.eval()
    res_model = atk_model.reasoner_model
    device = next(atk_model.parameters()).device
    n, embed_dim = args.num_vars, args.embed_dim

    num_dones = 0
    raw_elems_hits, raw_state_hits = 0, 0
    adv_ns1_elems_hits, adv_ns1_state_hits = 0, 0
    adv_ns2_elems_hits, adv_ns2_state_hits = 0, 0
    adv_ns3_elems_hits, adv_ns3_state_hits = 0, 0
    adv_ns1_attn_wts, adv_ns2_attn_wts, adv_ns3_attn_wts = \
        torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
    adv_ns1_suppd_wts, adv_ns2_suppd_wts, adv_ns3_suppd_wts = \
        torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)

    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        raw_tokens, infos, adv_labels = \
            batch["tokens"].to(device), batch["infos"].to(device), batch["labels"].to(device)
        a, b, c, d, e, f, g, h = infos.chunk(8, dim=-1)

        # Check the raw stats
        raw_labels = torch.cat([
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n) + hot(g,n),
        ], dim=1).to(device)
        raw_out = res_model(tokens=raw_tokens, output_attentions=True)
        raw_pred = (raw_out.logits > 0).long()

        # Check the adv states
        atk_out = atk_model(tokens=raw_tokens, infos=infos, labels=adv_labels)
        adv_tokens = torch.cat([atk_out.logits, raw_tokens], dim=1)
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
        if args.reasoner_type == "learned":
            adv_out1, adv_out2, adv_out3 = adv_out.all_seqcls_outputs
            adv_attn1 = adv_out1.attentions[0][:,0] # (N, r+k, r+k)
            adv_attn2 = adv_out2.attentions[0][:,0] # (N, r+k+1, r+k+1)
            adv_attn3 = adv_out3.attentions[0][:,0] # (N, r+k+2, r+k+2)

        elif args.reasoner_type == "theory":
            adv_attn1, adv_attn2, adv_attn3 = adv_out.attentions

        # Cumulative attention weight of the attack tokens
        k = atk_model.num_attack_tokens
        adv_ns1_attn_wts = torch.cat([adv_ns1_attn_wts, adv_attn1[:,-1,:k].sum(dim=-1)])
        adv_ns2_attn_wts = torch.cat([adv_ns2_attn_wts, adv_attn2[:,-1,:k].sum(dim=-1)])
        adv_ns3_attn_wts = torch.cat([adv_ns3_attn_wts, adv_attn3[:,-1,:k].sum(dim=-1)])

        if args.attack_name == "suppress_rule":
            supp_idx = batch["supp_idx"].to(device)
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

        desc = f"[eval] "
        desc += f"nd ({n},{embed_dim}), N {num_dones}: "
        desc += f"raw ({raw_elems_acc:.2f},{raw_state_acc:.2f}), "
        desc += f"adv ({adv_ns1_elems_acc:.2f},{adv_ns1_state_acc:.2f} # "
        desc += f"{adv_ns2_elems_acc:.2f},{adv_ns2_state_acc:.2f} # "
        desc += f"{adv_ns3_elems_acc:.2f},{adv_ns3_state_acc:.2f}), "
        desc += f"attn "
        desc += f"({adv_ns1_attn_wts.mean().item():.2f},"
        desc += f"{adv_ns2_attn_wts.mean().item():.2f},"
        desc += f"{adv_ns3_attn_wts.mean().item():.2f})"
        pbar.set_description(desc)

    row_dict = {
        "reasoner_type": args.reasoner_type,
        "reasoner_seed": args.reasoner_seed,
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
    if args.attack_name == "suppress_rule":
        other_dict = {
            "adv_ns1_attn_ratio": (adv_ns1_attn_wts/adv_ns1_suppd_wts).mean().item(),
            "adv_ns2_attn_ratio": (adv_ns2_attn_wts/adv_ns2_suppd_wts).mean().item(),
            "adv_ns3_attn_ratio": (adv_ns3_attn_wts/adv_ns3_suppd_wts).mean().item(),
        }
        row_dict = row_dict | other_dict

    if do_save:
        csv_saveto = get_info_strings(args)["csv_saveto"]
        append_to_csv(csv_saveto, row_dict)

    # Return the dict of the things to save
    return row_dict


def run_learned_attack(args: LearnedAttackExperimentsArguments):
    torch.manual_seed(args.attacker_seed)

    reasoner_model, reasoner_dataset = load_model_and_dataset_from_big_grid(
        num_vars = args.num_vars,
        embed_dim = args.embed_dim,
        seed = args.reasoner_seed,
    )

    if args.reasoner_type == "theory":
        reasoner_model = TheoryAutoregKStepsModel(
            num_vars = args.num_vars,
            num_steps = reasoner_model.num_steps
        )

    if args.attack_name == "suppress_rule":
        atk_model = SuppressRuleWrapperModel(
            reasoner_model = reasoner_model,
            num_attack_tokens = args.num_attack_tokens,
            attack_tokens_style = args.attack_tokens_style,
        )
        atk_model.eval()
        atk_model.to(args.device)
        train_dataset = SuppressRuleDataset(reasoner_dataset, args.train_len)
        eval_dataset = SuppressRuleDataset(reasoner_dataset, args.eval_len)

    elif args.attack_name == "knowledge_amnesia":
        atk_model = KnowledgeAmnesiaWrapperModel(
            reasoner_model = reasoner_model,
            num_attack_tokens = args.num_attack_tokens,
            attack_tokens_style = args.attack_tokens_style,
        )
        atk_model.eval()
        atk_model.to(args.device)
        train_dataset = KnowledgeAmnesiaDataset(reasoner_dataset, args.train_len)
        eval_dataset = KnowledgeAmnesiaDataset(reasoner_dataset, args.eval_len)

    else:
        raise ValueError(f"Unrecognized attack_name {args.attack_name}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)


    optimizer = AdamW(atk_model.parameters(), lr=args.learning_rate)

    train_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(train_steps * args.warmup_ratio)
    decay_steps = train_steps - warmup_steps
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers = [
            LinearLR(optimizer, start_factor=0.01, end_factor=1.00, total_iters=warmup_steps),
            LinearLR(optimizer, start_factor=1.00, end_factor=0.01, total_iters=decay_steps)
        ],
        milestones = [warmup_steps]
    )

    # Figure out where to save things
    info_strs = get_info_strings(args)
    print(f"run name: {info_strs['run_name']}")

    # Do one eval at the start just for reference
    eval_one_epoch(atk_model, eval_dataloader, args)
    all_losses, best_loss = [], None
    print(f"{args.reasoner_type}, n {args.num_vars}, d {args.embed_dim}")
    for epoch in range(1, args.num_epochs+1):
        print(f"epoch {epoch}/{args.num_epochs}, lr {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = train_one_epoch(atk_model, train_dataloader, optimizer, lr_scheduler, args)
        this_loss = train_stats["loss"]
        all_losses.extend(train_stats["all_losses"])

        eval_stats = eval_one_epoch(atk_model, eval_dataloader, args)

        save_dict = {
            "epoch": epoch,
            "train_loss": this_loss,
            "model_state_dict": {k: v.cpu() for (k,v) in atk_model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "all_losses": all_losses
        }

        torch.save(save_dict, info_strs["last_saveto"])

        if best_loss is None or this_loss < best_loss:
            best_save_dict = save_dict
            delta = 0. if best_loss is None else (best_loss - this_loss)
            print(f"New best {this_loss:.4f}, delta {delta:.4f}")
            best_loss = this_loss
            torch.save(save_dict, info_strs["best_saveto"])

    # Do a CSV dump on the last one
    eval_stats = eval_one_epoch(atk_model, eval_dataloader, args, do_save=True)

    # eval_one_epoch(atk_model, eval_dataloader, args)
    return best_save_dict


if __name__ == "__main__":
    parser = HfArgumentParser(LearnedAttackExperimentsArguments)
    args = parser.parse_args_into_dataclasses()[0]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.attacker_seed)

    if args.attack_name == "coerce_state":
        # trainer = make_coerce_state_trainer(args)
        pass

    elif args.attack_name in ["suppress_rule", "knowledge_amnesia"]:
        run_learned_attack(args)

    else:
        raise ValueError(f"Unknown attack name {args.attack_name}")



