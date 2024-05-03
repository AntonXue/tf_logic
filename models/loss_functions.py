from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput
from transformers import GPT2Config, GPT2Model


""" Some different loss functions """

class MarginLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, logits, labels):
        labels = 2 * labels - 1
        loss = F.relu(self.delta - logits.float() * labels.float()).mean()
        return loss

class RegCatLoss(nn.Module):
    def __init__(self, rho: float = None):
        super().__init__()
        self.rho = rho

    def forward(self, logits, labels):
        labels = 2 * labels - 1
        loss = -(logits.float() * labels.float()).mean()

        _, n = logits.shape
        rho = (0.5 * n**2) if self.rho is None else rho
        reg = rho * F.mse_loss(logits, torch.zeros_like(logits))
        return loss + reg


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        labels = 2 * labels - 1
        loss = F.mse_loss(logits, labels)
        return loss




