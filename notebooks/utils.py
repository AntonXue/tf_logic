import sys
sys.path.append("..")

import torch
import numpy as np

def qed_compute_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        preds = eval_preds.predictions[0] > 0
    else:
        preds = eval_preds.predictions > 0
    avg_ones = np.mean(preds)
    acc = np.mean(preds == eval_preds.label_ids)
    return {"Accuracy" : acc, "Avg Ones" : avg_ones}

def succ_compute_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        preds = eval_preds.predictions[0] > 0
    else:
        preds = eval_preds.predictions > 0
    avg_ones = np.mean(preds)
    acc = np.mean(preds == eval_preds.label_ids)
    return {"Accuracy" : acc, "Avg Ones" : avg_ones}

def arsteps_compute_metrics(eval_preds):
    if isinstance(eval_preds.predictions, tuple):
        preds = eval_preds.predictions[0] > 0
    else:
        preds = eval_preds.predictions > 0
    avg_ones = np.mean(preds)
    acc = np.mean(preds == eval_preds.label_ids)
    return {"Accuracy" : acc, "Avg Ones" : avg_ones}

