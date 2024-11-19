import torch
from models import AutoregTheoryModel
from my_datasets import AutoregDataset
from train_utils import init_optimizer_and_lr_scheduler
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

saveto_dir = str(Path(Path(__file__).parent.parent, "_saved_models/autoreg_grid"))


def train_model(model, dataloader, optimizer, lr_scheduler):
    model.train().to(device)
    n, d = model.input_dim // 2, model.embed_dim
    losses = []
    e1_accs, e2_accs, e3_accs = [], [], []
    s1_accs, s2_accs, s3_accs = [], [], []
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        tokens, labels = batch["tokens"].to(device), batch["labels"].to(device)
        out = model(tokens, labels=labels)
        loss = out.loss
        loss.backward();
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

        # Track stuff
        pred = (out.logits > 0).long()
        loss = loss.detach().cpu().item()
        losses.append(loss)

        e1_accs.append((pred == labels)[:,:1].float().mean().item())
        e2_accs.append((pred == labels)[:,:2].float().mean().item())
        e3_accs.append((pred == labels)[:,:3].float().mean().item())

        s1_accs.append((pred == labels)[:,:1].all(dim=-1).all(dim=-1).float().mean().item())
        s2_accs.append((pred == labels)[:,:2].all(dim=-1).all(dim=-1).float().mean().item())
        s3_accs.append((pred == labels)[:,:3].all(dim=-1).all(dim=-1).float().mean().item())

        pbar.set_description(
            f"n {n}, d {d}, loss {loss:.3f}, elems {e3_accs[-1]:.3f}, state {s3_accs[-1]:.3f}"
        )

    return {
        "model_state_dict": model.eval().cpu().state_dict(),
        "losses": losses,
        "e1_accs": e1_accs, "e2_accs": e2_accs, "e3_accs": e3_accs,
        "s1_accs": s1_accs, "s2_accs": s2_accs, "s3_accs": s3_accs,
    }


def init_and_train(num_props, embed_dim, batch_size, num_optim_steps, lr, seed):
    torch.manual_seed(seed)
    model = AutoregTheoryModel(num_props=num_props, num_steps=3, embed_dim=embed_dim, do_layer_norm=False)
    model.train()
    dataset = AutoregDataset(num_props, batch_size*num_optim_steps)
    dataloader = DataLoader(dataset, batch_size)
    optimizer, lr_scheduler = init_optimizer_and_lr_scheduler(model, num_optim_steps, lr)
    ret = train_model(model, dataloader, optimizer, lr_scheduler)
    return ret


def train_theory(
    nd_pairs,
    batch_size = 512,
    num_optim_steps = 10000,
    lr = 1e-3,
    seed = 101,
):
    for (n, d) in nd_pairs:
        ret = init_and_train(n, d, batch_size, num_optim_steps, lr, seed)
        saveto = f"theory_n{n}_d{d}_bsz{batch_size}_ns{num_optim_steps}_lr{lr:.4f}_seed{seed}.pt"
        torch.save(ret, saveto_dir + "/" + saveto)


all_embed_dims = [128, 112, 96, 80, 64, 48, 32]
all_num_props = [64, 56, 48, 40, 32, 24, 16]

n16_pairs = [(16, d) for d in all_embed_dims]
n24_pairs = [(24, d) for d in all_embed_dims]
n32_pairs = [(32, d) for d in all_embed_dims]
n40_pairs = [(40, d) for d in all_embed_dims]
n48_pairs = [(48, d) for d in all_embed_dims]
n56_pairs = [(56, d) for d in all_embed_dims]
n64_pairs = [(64, d) for d in all_embed_dims]

d128_pairs = [(n, 128) for n in all_num_props]
d112_pairs = [(n, 112) for n in all_num_props]
d96_pairs = [(n, 96) for n in all_num_props]
d80_pairs = [(n, 80) for n in all_num_props]
d64_pairs = [(n, 64) for n in all_num_props]
d48_pairs = [(n, 48) for n in all_num_props]
d32_pairs = [(n, 32) for n in all_num_props]

