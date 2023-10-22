import os
import sys
import argparse
from tqdm import tqdm

import torch

from models import *
from logic import *

from train_utils import *


def run_one_epoch(model, loss_and_stats, optimizer, stats_to_str, config, phase):
    """ Run one epoch of either training or testing
    """
    assert phase in ["train", "test"]
    _ = model.train() if phase == "train" else model.eval()
    device = next(model.parameters()).device
    stats = {}
    pbar = tqdm(range(config["num_batches"]))
    for i in pbar:
        rules, thm = generate_logic_stuff(config)
        rules, thm = rules.to(device), thm.to(device)
        with torch.set_grad_enabled(phase == "train"):
            if phase == "train":
                optimizer.zero_grad()
                loss, stats = loss_and_stats(model, rules, thm, stats)
                loss.backward()
                optimizer.step()
            else:
                loss, stats = loss_and_stats(model, rules, thm, stats)

        desc = "[train] " if phase == "train" else "[test]  "
        desc += stats_to_str(stats) if callable(stats_to_str) else ""
        pbar.set_description(desc)

    return {
        "model" : model,
        "loss" : stats["loss"],
        "stats" : stats,
    }


def train(train_stuff, config, device="cuda"):
    """ train_stuff has the models, optimizer, etc. config contains other info
    """
    model = train_stuff["model"]
    loss_and_stats = train_stuff["loss_and_stats"]
    optimizer = train_stuff["optimizer"]
    scheduler = train_stuff["scheduler"]
    stats_to_str = train_stuff["stats_to_str"]

    model.to(device)
    saveto_prefix = config_to_saveto_prefix(config)
    last_saveto = os.path.join(config["saveto_dir"], saveto_prefix + "_last.pt")
    best_saveto = os.path.join(config["saveto_dir"], saveto_prefix + "_best.pt")

    best_test_loss, train_losses, test_losses = 1e5, [], []
    todo_epochs = range(1, config["num_epochs"]+1)
    print(f"Training on task {config['task']}")
    print(f"Wlll save to: {best_saveto}")
    for epoch in todo_epochs:
        last_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{todo_epochs[-1]}, lr {last_lr:g}")
        train_stats = run_one_epoch(model, loss_and_stats, optimizer, stats_to_str, config, "train")
        test_stats = run_one_epoch(model, loss_and_stats, optimizer, stats_to_str, config, "test")

        train_loss, test_loss = train_stats["loss"], test_stats["loss"]
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

        torch.save(save_dict, last_saveto)
        if test_loss < best_test_loss or epoch == todo_epochs[0]:
            torch.save(save_dict, best_saveto)
            delta = best_test_loss - test_loss
            print(f"Saving, new best {test_loss:.3f}, old best {best_test_loss:.3f}, delta {delta:.3f}")
            best_test_loss = test_loss
        print("")

    return train_stuff


def parse_args_to_config():
    # Some default numbers to tweak with
    # r, n, nb, nh = 96, 28, 5, 8
    r, n, nb, nh = 8, 10, 4, 4
    d = r * n * 8
    ffw, ffd, seq_len = 2*d, 5, n+r+n

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
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step-every", type=int, default=5)
    parser.add_argument("--step-gamma", type=float, default=0.9)

    parser.add_argument("--task", choices=["check-qed", "check-step"], default="check-qed")
    parser.add_argument("--saveto-dir", type=str, default="saved_models/")
    parser.add_argument("--saveto-name", type=str, default="logictf")

    parser.add_argument("--go", action="store_true", default=False)
    args, unknown = parser.parse_known_args()
    return dict(args._get_kwargs())



if __name__ == "__main__":
    config = parse_args_to_config()
    train_stuff = config_to_train_stuff(config)

    if config["go"]:
        train(train_stuff, config)


