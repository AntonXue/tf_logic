from typing import Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def default(value, default):
    return default if value is None else value


str_to_activ_module = {
    "relu" : nn.ReLU(),
    "gelu" : nn.GELU()
}

def get_activ(activ_str: str):
    return str_to_activ_module[activ_str]


def e(i, d):
    return torch.eye(d)[:,i]


@dataclass
class TaskConfig: pass


class TaskModel(nn.Module):
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config
        # Also copy over all the rules
        for k, v in asdict(config).items():
            self.__setattr__(k, v)


