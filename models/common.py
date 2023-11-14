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
    return torch.eye(d)[:,i].long()


@dataclass
class TFLConfig: pass


class TFLModel(nn.Module):
    def __init__(self, seqcls_model: nn.Module, config: TFLConfig):
        super().__init__()
        self.seqcls_model = seqcls_model
        assert hasattr(self.seqcls_model, "model_name")
        assert hasattr(self.seqcls_model, "embed_dim")
        assert hasattr(self.seqcls_model, "num_labels")

        self.embed_dim = seqcls_model.embed_dim

        # transformer.Trainer with wandb checks whether the model has a config attr,
        # and will force us to implement a to_dict function if so.
        # self.config = config
        for k, v in asdict(config).items():
            self.__setattr__(k, v)

