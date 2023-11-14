import os
import wandb
import torch
import numpy as np

os.environ["WANDB_PROJECT"] = "transformer_friends"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def one_shot_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        logits = eval_preds.predictions[0]
    else:
        logits = eval_preds.predictions
    preds = np.argmax(logits, axis=1)
    avg_ones = np.mean(preds)
    acc = np.mean(preds == eval_preds.label_ids)
    try:
        # In case the eval metrics also include losses, just return the mean loss
        loss = np.mean(eval_preds.losses)
        return {
            "Accuracy": acc,
            "Avg Ones": avg_ones,
            "Loss": loss
        }
    except:
        return {"Accuracy" : acc, "AvgOnes" : avg_ones}


def next_state_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        logits = eval_preds.predictions[0]
    else:
        logits = eval_preds.predictions
    preds = (logits > 0).astype(np.int64)
    avg_ones = np.mean(preds)
    acc = np.mean(preds == eval_preds.label_ids)
    return {"Accuracy" : acc, "AvgOnes" : avg_ones}


def autoreg_ksteps_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        logits = eval_preds.predictions[0]
    else:
        logits = eval_preds.predictions
    preds = (logits > 0).astype(np.int64)
    avg_ones = np.mean(preds)
    acc = np.mean(preds == eval_preds.label_ids)
    return {"Accuracy" : acc, "AvgOnes" : avg_ones}

