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

def generate_logic_stuff(configs):
    N, r, n = configs["batch_size"], configs["num_rules"], configs["num_vars"]
    pa, pb, pt = configs["ante_prob"], configs["conseq_prob"], configs["theorem_prob"]
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


def configs_to_train_stuff(configs):
    encoder = ProverEncoder(num_vars = configs["num_vars"],
                            num_rules = configs["num_rules"],
                            model_dim = configs["model_dim"],
                            seq_len = configs["seq_len"])

    tf = MyTransformer(model_dim = configs["model_dim"],
                       ffwd_width = configs["ffwd_width"],
                       ffwd_depth = configs["ffwd_depth"],
                       num_heads = configs["num_heads"],
                       num_blocks = configs["num_blocks"])

    if configs["task"] == "check-qed":
        decoder = QedDecoder(model_dim=configs["model_dim"])
    elif configs["task"] == "check-step":
        decoder = StateDecoder(model_dim=configs["model_dim"], num_vars=configs["num_vars"])
    else:
        raise NotImplementedError()

    model = LogicTransformer(encoder, tf, decoder)

    if configs["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=configs["lr"], weight_decay=1e-4)
    elif configs["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=configs["lr"], weight_decay=1e-4)
    elif configs["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=configs["lr"])
    else:
        raise NotImplementedError()

    scheduler = schedulers.StepLR(optimizer, step_size=configs["step_every"], gamma=configs["step_gamma"])

    if configs["task"] == "check-qed":
        loss_and_stats = check_qed_loss_and_stats
        stats_to_str = check_qed_stats_to_str
    elif configs["task"] == "check-step":
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
def configs_to_saveto_prefix(configs):
    name = configs["saveto_name"]
    n, r = configs["num_vars"], configs["num_rules"]
    d, fw, fd = configs["model_dim"], configs["ffwd_width"], configs["ffwd_depth"]
    nh, nb = configs["num_heads"], configs["num_blocks"]
    pa, pb, pt = configs["ante_prob"], configs["conseq_prob"], configs["theorem_prob"]
    task = configs["task"]
    return f"{name}_{task}_n{n}_r{r}_d{d}_fw{fw}_fd{fd}_nh{nh}_nb{nb}__pa{pa}_pb{pb}_pt{pt}"

