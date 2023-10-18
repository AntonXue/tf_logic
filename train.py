import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.optim

from models import *
from logic import *


def generate_logic_stuff(config):
    N, r, n = config["batch_size"], config["num_rules"], config["num_vars"]
    pa, pb, pc = config["ante_prob"], config["conseq_prob", config["theorem_prob"]
    rules = random_rules(batch_size=N, num_rules=r, num_vars=n, ante_prob=pa, conseq_prob=pb)
    theorem = (torch.rand(N, n) < pc).long()
    return rules, theorem


def run_one_epoch(model, loss_fn, optimizer, train_config, logic_config):
    assert phase in ["train", "test"]
    _ = model.train().to(device) if phase == "train" else model.eval().to(device)
    num_batches, loss_fn = train_config["num_batches"], train_config["loss_fn"]
    num_dones, running_loss = 0, 0.
    pbar = tqdm(range(num_batches))
    for i in pbar:
        rules, thm = generate_logic_stuff(logic_config)
        rules, thm = rules.to(device), thm.to(device)
        with torch.set_grad_enabled(phase == "train"):
            if phase == "train":
                optimizer.zero_grad()
                loss = loss_fn(model, rules, thm)
                loss.backward()
                optimizer.step()
            else:
                loss = loss_fn(model, rules, thm)

        num_dones += rules.size(0)
        running_loss += loss
        desc = "[train]" if phase == "train" else "[test] "
        desc += f"dones {num_dones}, loss {(running_loss / num_dones):.3f}"
        pbar.set_description(desc)

    return {
        "model" : model,
        "num_dones" : num_dones,
        "loss" : running_loss
    }





def train(configs):
    model = configs["model"]

    best_test_loss = 0.0
    train_losses, test_losses = [], []
    for epoch in todo_epochs:
        print(f"Running epoch {epoch}/{todo_epochs[-1]}")
        train_stats = run_one_epoch(configs, phase="train")
        test_stats = run_one_epoch(configs, phase="test")
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
        }

        if test_loss < best_test_loss:
            loss_delta = best_test_loss - test_loss
            print(f"New best {test_loss:.3f}, old best {best_test_loss:.3f}")
            print(f"Saved to {best_saveto}")
            best_test_loss = test_loss

    pass


# Make the big string
def config_to_saveto_prefix(configs):
    pass

def make_configs(args):

    logic_configs = {
        "num_vars" : args.num_vars,
        "num_rules" : args.num_rules,
        "ante_prob" : args.ante_prob,
        "conseq_prob" : args.conseq_prob,
        "theorem_prob" : args.theorem_prob
    }

    model_dim = args.model_dim
    ffwd_width = args.ffwd_width
    ffwd_depth = args.ffwd_depth
    num_heads = args.num_heads
    num_blocks = args.num_blocks
    activ = args.activ

    tf_model = MyTransformer(model_dim=model_dim)

    return train_configs, logic_configs




def parse_args():
    _SAVETO_DIR = "saved_models/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-vars", type=int, default=20)
    parser.add_argument("--num-rules", type=int, default=50)
    parser.add_argument("--ante-prob", type=float, default=0.2)
    parser.add_argument("--conseq-prob", type=float, default=0.2)
    parser.add_argument("--theorem-prob", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--activ", type=str, default="relu")
    parser.add_argument("--saveto-dir", type=str, default=_SAVETO_DIR)
    args, unknown = parser.parse_known_args()

