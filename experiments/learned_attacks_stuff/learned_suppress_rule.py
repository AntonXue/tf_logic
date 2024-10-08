"""
    Custom trainer for suppress rule attack because HuggingFace trainer is annoying
"""

import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

sys.path.append(str(Path(Path(__file__).parent.parent.parent.resolve()))) # Project root
from experiments.utils.model_loader_utils import load_model_and_dataset_from_big_grid
from models.task_models import *
from models.attack_models import SuppressRuleWrapperModel
from my_datasets import SuppressRuleDataset

@dataclass
class LearnedSuppressRuleConfig:
    num_vars: int
    embed_dim: int
    num_attack_tokens: int
    attack_tokens_style: str
    train_len: int
    eval_len: int
    batch_size: int
    num_epochs: int
    reasoner_seed: int
    attacker_seed: int
    reasoner_type: str = "learned"
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.1
    device: str = "cuda"
    output_dir: str = None


def hot(i,n):
    return F.one_hot(i, n)


def train_one_epoch(atk_model, dataloader, optimizer, lr_scheduler, config):
    optimizer.zero_grad()
    atk_model.train()
    num_dones, all_losses = 0, []
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        tokens = batch["tokens"].to(config.device)
        infos = batch["infos"].to(config.device)
        adv_labels = batch["labels"].to(config.device)

        out = atk_model(tokens=tokens, infos=infos, labels=adv_labels)

        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # Track stats
        num_dones += tokens.size(0)
        all_losses.append(loss.item())
        first_loss = all_losses[0]
        sma_loss = torch.tensor(all_losses[-16:]).mean().item() # Smooth-moving average
        lr = lr_scheduler.get_last_lr()[0]

        desc = "[train] "
        desc += f"N {num_dones}, lr {lr:.6f}, loss {sma_loss:.4f} (first {first_loss:.4f}"
        pbar.set_description(desc)

    return {
        "loss": sma_loss,
        "all_losses": all_losses
    }


@torch.no_grad()
def eval_one_epoch(atk_model, dataloader, config):
    atk_model.eval()
    res_model = atk_model.reasoner_model

    n = config.num_vars

    num_dones = 0
    cum_raw_elems_hits, cum_raw_state_hits = 0, 0
    cum_adv_elems_hits, cum_adv_state_hits = 0, 0
    cum_top3_hits = 0
    cum_adv_weight = 0

    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        raw_tokens = batch["tokens"].to(config.device)
        adv_labels = batch["labels"].to(config.device)
        infos = batch["infos"].to(config.device)
        a, b, c, d, e, f, g, h = infos.chunk(8, dim=-1)

        # Check the raw stats
        raw_labels = torch.cat([
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n) + hot(g,n),
        ], dim=1)
        raw_out = res_model(tokens=raw_tokens, output_attentions=True)
        raw_pred = (raw_out.logits > 0).long()

        # Check the adv states
        atk_out = atk_model(tokens=raw_tokens, infos=infos, labels=adv_labels)
        atk_token = atk_out.logits
        adv_tokens = torch.cat([
            atk_out.logits.view(-1, config.num_attack_tokens, 2*n),
            raw_tokens
        ], dim=1)

        adv_out = res_model(tokens=adv_tokens, output_attentions=True)
        adv_pred = (adv_out.logits > 0).long()

        # Extract the relevant information
        if config.reasoner_type == "learned":
            adv_out, *_ = adv_out.all_seqcls_outputs # This is 3 steps; get the seocnd step
            adv_attn = adv_out.attentions[0] # Extract from the 1-tuple; shape (N,1,L,L)
            adv_attn = adv_attn[:,0] # (N,L,L)
        elif config.reasoner_type == "theory":
            adv_attn, *_ = adv_out.attentions
        else:
            raise ValueError(f"Unrecognized reasoner_type {config.reasoner_type}")
        top_attn_inds = adv_attn[:,-1].sort(dim=1, descending=True).indices # (N,L)

        # Do some metrics
        num_dones += raw_tokens.size(0)

        cum_raw_elems_hits += (raw_pred == raw_labels).float().mean(dim=(1,2)).sum()
        raw_elems_acc = cum_raw_elems_hits / num_dones

        cum_raw_state_hits += (raw_pred == raw_labels).all(dim=-1).all(dim=-1).sum()
        raw_state_acc = cum_raw_state_hits / num_dones

        cum_adv_elems_hits += (adv_pred == adv_labels).float().mean(dim=(1,2)).sum()
        adv_elems_acc = cum_adv_elems_hits / num_dones

        cum_adv_state_hits += (adv_pred == adv_labels).all(dim=-1).all(dim=-1).sum()
        adv_state_acc = cum_adv_state_hits / num_dones

        cum_top3_hits += (top_attn_inds[:,:3] == 0).sum()
        top3_acc = cum_top3_hits / num_dones

        cum_adv_weight += adv_attn[:,-1,0].sum()
        adv_weight = cum_adv_weight / num_dones
        rel_adv_weight = adv_weight * adv_tokens.size(1)

        desc = "[eval]  "
        desc += f"raw_acc ({raw_elems_acc:.3f}, {raw_state_acc:.3f}), "
        desc += f"adv_acc ({adv_elems_acc:.3f}, {adv_state_acc:.3f}), "
        desc += f"adv_top3 {top3_acc:.3f}, "
        desc += f"adv_wt {adv_weight:.3f} (rel {rel_adv_weight:.3f}), "
        pbar.set_description(desc)

    return {
        "raw_elems_acc": raw_elems_acc,
        "raw_state_acc": raw_state_acc,
        "adv_elems_acc": raw_elems_acc,
        "adv_state_acc": raw_state_acc,
        "adv_top3_acc": top3_acc,
        "adv_wt": adv_weight,
    }



def run_learned_suppress_rule(config: LearnedSuppressRuleConfig):
    torch.manual_seed(config.attacker_seed)

    reasoner_model, reasoner_dataset = load_model_and_dataset_from_big_grid(
        num_vars = config.num_vars,
        embed_dim = config.embed_dim,
        seed = config.reasoner_seed,
    )

    if config.reasoner_type == "theory":
        reasoner_model = TheoryAutoregKStepsModel(
            num_vars = config.num_vars,
            num_steps = reasoner_model.num_steps
        )

    atk_model = SuppressRuleWrapperModel(
        reasoner_model = reasoner_model,
        num_attack_tokens = config.num_attack_tokens,
        attack_tokens_style = config.attack_tokens_style,
    )
    atk_model.to(config.device)

    train_dataloader = DataLoader(
        SuppressRuleDataset(
            reasoner_dataset = reasoner_dataset,
            dataset_len = config.train_len
        ),
        batch_size = config.batch_size,
        shuffle = True
    )

    eval_dataloader = DataLoader(
        SuppressRuleDataset(
            reasoner_dataset = reasoner_dataset,
            dataset_len = config.eval_len
        ),
        batch_size = config.batch_size,
        shuffle = True
    )

    optimizer = AdamW(atk_model.parameters(), lr=config.learning_rate)
        
    train_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(train_steps * config.warmup_ratio)
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
    run_name = f"suppress_rules"
    run_name += f"_{config.reasoner_type}"
    run_name += f"_n{config.num_vars}_d{config.embed_dim}"
    run_name += f"_k{config.num_attack_tokens}_{config.attack_tokens_style}"
    last_saveto = str(Path(config.output_dir, run_name + "_last.pt"))
    best_saveto = str(Path(config.output_dir, run_name + "_best.pt"))
    print(f"run name: {run_name}")

    # Do one eval at the start just for reference
    eval_one_epoch(atk_model, eval_dataloader, config)

    all_losses = []
    best_loss = None

    print(f"{config.reasoner_type}, n {config.num_vars}, d {config.embed_dim}")
    for epoch in range(1, config.num_epochs+1):
        print(f"epoch {epoch}/{config.num_epochs}, lr {lr_scheduler.get_last_lr()[0]:.6f}")
        train_stats = train_one_epoch(atk_model, train_dataloader, optimizer, lr_scheduler, config)
        this_loss = train_stats["loss"]
        all_losses.extend(train_stats["all_losses"])

        eval_stats = eval_one_epoch(atk_model, eval_dataloader, config)

        save_dict = {
            "epoch": epoch,
            "train_loss": this_loss,
            "model_state_dict": {k: v.cpu() for (k,v) in atk_model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "all_losses": all_losses
        }

        torch.save(save_dict, last_saveto)

        if best_loss is None or this_loss < best_loss:
            best_save_dict = save_dict
            delta = 0. if best_loss is None else (best_loss - this_loss)
            print(f"New best {this_loss:.4f}, delta {delta:.4f}")
            best_loss = this_loss
            torch.save(save_dict, best_saveto)

    # eval_one_epoch(atk_model, eval_dataloader, config)
    return best_save_dict



