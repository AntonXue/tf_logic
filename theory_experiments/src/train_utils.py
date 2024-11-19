# from transformers.optimization import AdamW
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_scheduler


def init_optimizer_and_lr_scheduler(
    model: nn.Module,
    num_steps: int,
    lr: float,
    warmup_ratio: float = 0.1,
    no_decay: str = ["bias", "LayerNorm.weight"],
):
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    warmup_steps = int(warmup_ratio*num_steps)
    lr_scheduler = get_scheduler(
        name = "linear",
        optimizer = optimizer,
        num_warmup_steps = warmup_steps,
        num_training_steps = num_steps,
    )

    return optimizer, lr_scheduler

