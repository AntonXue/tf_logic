import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers

from models import *
from logic import *

def generate_logic_stuff(config):
    N, r, n = config["batch_size"], config["num_rules"], config["num_vars"]
    pa, pb, pt = config["ante_prob"], config["conseq_prob"], config["theorem_prob"]
    rules = random_rules(batch_size=N, num_rules=r, num_vars=n, ante_prob=pa, conseq_prob=pb)
    theorem = (torch.rand(N, n) < pt).long()
    return rules, theorem


def check_qed_loss_and_stats(model, rules, theorem, stats):
    qed_true = prove_theorem(rules, theorem)["qed"]
    qed_pred = model(rules, theorem)
    loss = F.binary_cross_entropy(qed_pred, qed_true.float(), reduction="sum")

    stats["num_dones"] = stats.get("num_dones", 0) + qed_pred.size(0)
    stats["cum_loss"] = stats.get("cum_loss", 0.) + loss
    stats["cum_hits"] = stats.get("cum_hits", 0) + ((qed_pred > 0.5) == qed_true).sum()
    stats["cum_qed_true_hot"] = stats.get("cum_qed_true_hot", 0) + qed_true.sum()
    stats["cum_qed_pred_hot"] = stats.get("cum_qed_pred_hot", 0) + (qed_pred > 0.5).sum()
    stats["cum_qed_pred"] = stats.get("cum_qed_pred", 0.) + qed_pred.sum()

    num_dones = stats["num_dones"] # Update the above first
    stats["acc"] = stats["cum_hits"] / num_dones
    stats["loss"] = stats["cum_loss"] / num_dones
    stats["qed_true_hot%"] = stats["cum_qed_true_hot"] / num_dones
    stats["qed_pred_hot%"] = stats["cum_qed_pred_hot"] / num_dones
    stats["avg_qed_pred"] = stats["cum_qed_pred"] / num_dones
    return loss, stats


def check_qed_stats_to_str(stats):
    return f"dones {stats['num_dones']}, " + \
            f"thot% {stats['qed_true_hot%']:.3f}, " + \
            f"phot% {stats['qed_pred_hot%']:.3f} ({stats['avg_qed_pred']:.3f}), " + \
            f"acc {stats['acc']:.3f}, loss {stats['loss']:.3f}"


def check_step_loss_and_stats(model, rules, theorem, stats):
    _, n = theorem.shape
    succ_true, _ = step_rules(rules, torch.zeros_like(theorem))
    succ_pred = model(rules, torch.zeros_like(theorem))
    loss = F.binary_cross_entropy(succ_pred, succ_true.float(), reduction="sum")

    stats["num_dones"] = stats.get("num_dones", 0) + succ_pred.size(0)
    stats["cum_loss"] = stats.get("cum_loss", 0.) + loss
    stats["cum_hits"] = stats.get("cum_hits", 0) + ((succ_pred > 0.5) == succ_true).sum()
    stats["cum_succ_pred_hots"] = stats.get("cum_succ_pred_hots", 0) + (succ_pred > 0.5).sum()

    num_dones = stats["num_dones"]
    stats["succ_pred_hot%"] = stats["cum_succ_pred_hots"] / (num_dones * n)
    stats["acc"] = stats["cum_hits"] / (num_dones * n)
    stats["loss"] = stats["cum_loss"] / num_dones
    return loss, stats


def check_step_stats_to_str(stats):
    return f"dones {stats['num_dones']}, " + \
            f"phot% {stats['succ_pred_hot%']:.3f}, " + \
            f"acc {stats['acc']:.3f}, loss {stats['loss']:.3f}"


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

    if config["task"] == "check-qed":
        decoder = QedDecoder(model_dim=config["model_dim"])
    elif config["task"] == "check-step":
        decoder = StateDecoder(model_dim=config["model_dim"], num_vars=config["num_vars"])
    else:
        raise NotImplementedError()

    model = LogicTransformer(encoder, tf, decoder)

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    elif config["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    else:
        raise NotImplementedError()

    scheduler = schedulers.StepLR(optimizer, step_size=config["step_every"], gamma=config["step_gamma"])

    if config["task"] == "check-qed":
        loss_and_stats = check_qed_loss_and_stats
        stats_to_str = check_qed_stats_to_str
    elif config["task"] == "check-step":
        loss_and_stats = check_step_loss_and_stats
        stats_to_str = check_step_stats_to_str
    else:
        raise NotImplementedError()

    return {
        "model" : model,
        "loss_and_stats" : loss_and_stats,
        "optimizer" : optimizer,
        "scheduler" : scheduler,
        "stats_to_str" : stats_to_str
    }


# Make the big string
def config_to_saveto_prefix(config):
    name = config["saveto_name"]
    n, r = config["num_vars"], config["num_rules"]
    d, fw, fd = config["model_dim"], config["ffwd_width"], config["ffwd_depth"]
    nh, nb = config["num_heads"], config["num_blocks"]
    pa, pb, pt = config["ante_prob"], config["conseq_prob"], config["theorem_prob"]
    task = config["task"]
    return f"{name}_{task}_n{n}_r{r}_d{d}_fw{fw}_fd{fd}_nh{nh}_nb{nb}__pa{pa}_pb{pb}_pt{pt}"

