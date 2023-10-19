import os
import sys
import argparse
from tqdm import tqdm

import torch

from models import *
from logic import *


def generate_logic_stuff(config):
    N, r, n = config["batch_size"], config["num_rules"], config["num_vars"]
    pa, pb, pt = config["ante_prob"], config["conseq_prob"], config["theorem_prob"]
    rules = random_rules(batch_size=N, num_rules=r, num_vars=n, ante_prob=pa, conseq_prob=pb)
    theorem = (torch.rand(N, n) < pt).long()
    return rules, theorem


def run_one_epoch(model, optimizer, config, phase):
    """ Run one epoch of either training or testing
    """
    assert phase in ["train", "test"]
    device = next(model.parameters()).device
    _ = model.train() if phase == "train" else model.eval()

    loss_fn = torch.nn.BCELoss(reduction="sum")
    num_dones = num_hits = num_qed_true = num_qed_pred = 0
    cum_loss, cum_qed_pred, cum_grads2 = 0., 0., 0.
    pbar = tqdm(range(config["num_batches"]))
    for i in pbar:
        rules, thm = generate_logic_stuff(config)
        rules, thm = rules.to(device), thm.to(device)
        qed_true = prove_theorem(rules, thm)["qed"]
        with torch.set_grad_enabled(phase == "train"):
            if phase == "train":
                optimizer.zero_grad()

            qed_pred = model(rules, thm)
            loss = loss_fn(qed_pred, qed_true.float())

            if phase == "train":
                loss.backward()
                grads2 = sum([(p.grad ** 2).sum() for p in model.parameters()])
                optimizer.step()

        num_dones += rules.size(0)
        num_qed_true += qed_true.sum()
        num_qed_pred += (qed_pred > 0.5).sum()
        cum_qed_pred += qed_pred.sum()
        cum_grads2 += grads2 if phase == "train" else 0.
    
        num_hits += ((qed_pred > 0.5) == qed_true).sum()
        cum_loss += loss

        # Dump some data
        desc = "[train] " if phase == "train" else "[test]  "
        desc += f"dones {num_dones}, "
        desc += f"qtrue {(num_qed_true/num_dones):.3f}, "
        desc += f"qpred {(num_qed_pred/num_dones):.3f} ({(cum_qed_pred/num_dones):.3f}), "
        desc += f"acc {(num_hits/num_dones):.3f}, "
        desc += f"loss {(cum_loss/num_dones):.3f} (g2 {(cum_grads2/num_dones):.3f}), "
        pbar.set_description(desc)

    return {
        "model" : model,
        "num_dones" : num_dones,
        "acc" : num_hits / num_dones,
        "loss" : cum_loss / num_dones,
    }



def train(model, optimizer, scheduler, config, device="cuda"):
    model.to(device)
    best_test_loss = 1e5
    train_losses, test_losses = [], []
    todo_epochs = range(1, config["num_epochs"]+1)
    for epoch in todo_epochs:
        last_lr = scheduler.get_last_lr()[0]
        print(f"Running epoch {epoch}/{todo_epochs[-1]}, lr {last_lr:g}")
        train_stats = run_one_epoch(model, optimizer, config, phase="train")
        test_stats = run_one_epoch(model, optimizer, config, phase="test")
        scheduler.step()

        train_loss = train_stats["loss"] / train_stats["num_dones"]
        test_loss = test_stats["loss"] / test_stats["num_dones"]

        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))

        save_dict = {
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "scheduler_state_dict" : scheduler.state_dict(),
            "epoch" : epoch,
            "train_losses" : train_losses,
            "test_losses" : test_losses,
            "config" : config,
        }

        prefix = config_to_saveto_prefix(config)
        last_saveto = os.path.join(config["saveto_dir"], prefix + "_last.pt")
        torch.save(save_dict, last_saveto)
        if test_loss < best_test_loss:
            best_saveto = os.path.join(config["saveto_dir"], prefix + "_best.pt")
            torch.save(save_dict, best_saveto)
            loss_delta = best_test_loss - test_loss
            print(f"New best {test_loss:.3f}, old best {best_test_loss:.3f}, delta {loss_delta:.3f}")
            print(f"Saved to {best_saveto}")
            best_test_loss = test_loss

    return model


# Make the big string
def config_to_saveto_prefix(config):
    name = config["saveto_name"]
    n, r = config["num_vars"], config["num_rules"]
    d, fw, fd = config["model_dim"], config["ffwd_width"], config["ffwd_depth"]
    nh, nb = config["num_heads"], config["num_blocks"]
    pa, pb, pt = config["ante_prob"], config["conseq_prob"], config["theorem_prob"]
    return f"{name}_n{n}_r{r}_d{d}_fw{fw}_fd{fd}_nh{nh}_nb{nb}__pa{pa}_pb{pb}_pt{pt}"


def config_to_train_stuff(config):
    encoder = ProverEncoder(num_vars = config["num_vars"],
                            num_rules = config["num_rules"],
                            model_dim = config["model_dim"],
                            seq_len = config["seq_len"])

    tf = MyTransformer(model_dim = config["model_dim"],
                       ffwd_width = config["ffwd_width"],
                       ffwd_depth = config["ffwd_depth"],
                       num_heads = config["num_heads"],
                       num_blocks = config["num_blocks"])

    decoder = CheckQedDecoder(model_dim = config["model_dim"])

    model = LogicTransformer(encoder, tf, decoder)

    
    # optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])
    optimizer = torch.optim.SGD(model.parameters(), lr = config["lr"])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = config["step_every"],
                                                gamma = config["step_gamma"])

    return model, optimizer, scheduler


def parse_args_to_config():
    r, n, nb, nh = 20, 16, 5, 4
    d = r * n * 8
    ffw, ffd, seq_len = 2*d, 3, n+r+n
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-vars", type=int, default=n)
    parser.add_argument("--num-rules", type=int, default=r)
    parser.add_argument("--ante-prob", type=float, default=0.2)
    parser.add_argument("--conseq-prob", type=float, default=0.2)
    parser.add_argument("--theorem-prob", type=float, default=0.3)

    parser.add_argument("--model-dim", type=int, default=d)
    parser.add_argument("--ffwd-width", type=int, default=ffw)
    parser.add_argument("--ffwd-depth", type=int, default=ffd)
    parser.add_argument("--num-heads", type=int, default=nh)
    parser.add_argument("--num-blocks", type=int, default=nb)
    parser.add_argument("--seq-len", type=int, default=seq_len)

    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--step-every", type=int, default=5)
    parser.add_argument("--step-gamma", type=float, default=0.9)

    parser.add_argument("--saveto-dir", type=str, default="saved_models/")
    parser.add_argument("--saveto-name", type=str, default="logictf")
    parser.add_argument("--go", action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    return dict(args._get_kwargs())



if __name__ == "__main__":
    config = parse_args_to_config()
    model, optimizer, scheduler = config_to_train_stuff(config)

    if config["go"]:
        train(model, optimizer, scheduler, config)


