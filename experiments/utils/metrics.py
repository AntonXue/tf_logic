import os
import torch
import numpy as np

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
            "AvgOnes": avg_ones,
            "Loss": loss
        }
    except:
        return {"Accuracy": acc, "AvgOnes": avg_ones}


def small_succ_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        logits = eval_preds.predictions[0]
    else:
        logits = eval_preds.predictions
    preds = (logits > 0).astype(np.int64)
    avg_ones = np.mean(preds)
    elems_acc = np.mean(preds == eval_preds.label_ids)
    states_acc = np.mean(np.mean(preds == eval_preds.label_ids, axis=1) > 1 - 1e-5)
    return {
        "ElemsAcc": elems_acc,
        "StatesAcc": states_acc,
        "AvgOnes": avg_ones
    }


