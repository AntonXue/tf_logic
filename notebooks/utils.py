import sys
sys.path.append("..")

import torch
import numpy as np

def qed_compute_metrics(eval_preds):
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
        return {"Accuracy" : acc, "Avg Ones" : avg_ones}

def succ_compute_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        logits = eval_preds.predictions[0]
    else:
        logits = eval_preds.predictions
    preds = (logits > 0).astype(np.int64)
    avg_ones = np.mean(preds)
    acc = np.mean(preds == eval_preds.label_ids)
    return {"Accuracy" : acc, "Avg Ones" : avg_ones}

def arsteps_compute_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        logits = eval_preds.predictions[0]
    else:
        logits = eval_preds.predictions
    preds = (logits > 0).astype(np.int64)
    avg_ones = np.mean(preds)
    acc = np.mean(preds == eval_preds.label_ids)
    return {"Accuracy" : acc, "Avg Ones" : avg_ones}

