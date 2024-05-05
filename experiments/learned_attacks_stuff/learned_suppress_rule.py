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
    train_len: int
    eval_len: int
    batch_size: int
    num_epochs: int
    reasoner_seed: int
    attacker_seed: int
    reasoner_type: str = "learned"
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    device: str = "cuda"


def train_one_epoch(atk_model, dataloader, optimizer, lr_scheduler, config):
    optimizer.zero_grad()
    atk_model.train()
    num_dones, cum_loss = 0, 0.
    first_loss = 0.
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        tokens = batch["tokens"].to(config.device)
        abcde = batch["abcde"].to(config.device)
        adv_labels = batch["labels"].to(config.device)

        out = atk_model(tokens=tokens, abcde=abcde, labels=adv_labels)

        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # Track stats
        if i == 0:
            first_loss = loss.item()

        num_dones += tokens.size(0)
        cum_loss += loss.item() * tokens.size(0)
        avg_loss = cum_loss / num_dones
        lr = lr_scheduler.get_last_lr()[0]
        desc = "[train] "
        desc += f"N {num_dones}, lr {lr:.6f}, loss {avg_loss:.4f} (first {first_loss:.4f})"
        pbar.set_description(desc)


@torch.no_grad()
def eval_one_epoch(atk_model, dataloader, config):
    atk_model.eval()
    res_model = atk_model.reasoner_model

    n = config.num_vars

    num_dones = 0
    cum_raw_elems_hits = 0
    cum_raw_state_hits = 0
    cum_adv_elems_hits = 0
    cum_adv_state_hits = 0
    cum_top3_hits = 0
    cum_adv_weight = 0

    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        raw_tokens = batch["tokens"].to(config.device)
        adv_labels = batch["labels"].to(config.device)
        abcde = batch["abcde"].to(config.device)
        a, b, c, d, e = abcde.chunk(5, dim=-1)

        # Check the raw stats
        raw_labels = torch.cat([
            F.one_hot(a,n).view(-1,1,n),
            (F.one_hot(a,n) + F.one_hot(b,n) + F.one_hot(c,n)).view(-1,1,n),
            (F.one_hot(a,n) + F.one_hot(b,n) + F.one_hot(c,n) + F.one_hot(d,n)).view(-1,1,n),
        ], dim=1)
        raw_out = res_model(tokens=raw_tokens, output_attentions=True)
        raw_pred = (raw_out.logits > 0).long()

        # Check the adv states
        atk_out = atk_model(tokens=raw_tokens, abcde=abcde, labels=adv_labels)
        atk_token = atk_out.logits
        adv_tokens = torch.cat([
            atk_out.logits.view(-1,1,2*n),
            raw_tokens
        ], dim=1)

        adv_out = res_model(tokens=adv_tokens, output_attentions=True)
        adv_pred = (adv_out.logits > 0).long()

        # Extract the relevant information
        if config.reasoner_type == "learned":
            _, adv_out2, _ = adv_out.all_seqcls_outputs # This is 3 steps; get the seocnd step
            adv_attn = adv_out2.attentions[0] # Extract from the 1-tuple; shape (N,1,L,L)
            adv_attn = adv_attn[:,0] # (N,L,L)
        elif config.reasoner_type == "theory":
            _, adv_attn, _ = adv_out.attentions
        else:
            raise ValueError(f"Unrecognized reasoner_type {config.reasoner_type}")
        top_attn_inds = adv_attn[:,-1].sort(dim=1, descending=True).indices # (N,L)

        # Do some metrics
        num_dones += raw_tokens.size(0)
        cum_raw_elems_hits += (raw_pred == raw_labels).float().mean(dim=(1,2)).sum()
        raw_elems_acc = cum_raw_elems_hits / num_dones

        cum_raw_state_hits += ((raw_pred == raw_labels).sum(dim=(1,2)) == 3*n).sum()
        raw_state_acc = cum_raw_state_hits / num_dones

        cum_adv_elems_hits += (adv_pred == adv_labels).float().mean(dim=(1,2)).sum()
        adv_elems_acc = cum_adv_elems_hits / num_dones

        cum_adv_state_hits += ((adv_pred == adv_labels).sum(dim=(1,2)) == 3*n).sum()
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
            num_steps = 3,
        )

    atk_model = SuppressRuleWrapperModel(reasoner_model=reasoner_model)
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

    # Do one eval at the start just for reference
    eval_one_epoch(atk_model, eval_dataloader, config)

    for epoch in range(1, config.num_epochs+1):
        print(f"epoch: {epoch}/{config.num_epochs}, lr {lr_scheduler.get_last_lr()[0]:.6f}")
        train_one_epoch(atk_model, train_dataloader, optimizer, lr_scheduler, config)

        if epoch % 2 == 0:
            eval_one_epoch(atk_model, eval_dataloader, config)

    eval_one_epoch(atk_model, eval_dataloader, config)




