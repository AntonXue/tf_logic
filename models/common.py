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
        assert hasattr(seqcls_model, "model_name")
        assert hasattr(seqcls_model, "embed_dim")
        assert hasattr(seqcls_model, "num_labels")
        self.seqcls_model = seqcls_model

        # HF's transformer.Trainer with wandb checks whether the model has a
        # config attr and will force us to implement a to_dict function if so.
        # self.config = config
        for k, v in asdict(config).items():
            self.__setattr__(k, v)

    @property
    def model_name(self):
        return self.seqcls_model.model_name

    @property
    def embed_dim(self):
        return self.seqcls_model.embed_dim

    @property
    def num_labels(self):
        return self.seqcls_model.num_labels


""" Sequence classification models abstract class """

class SeqClsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def model_name(self):
        raise NotImplementedError()

    @property
    def input_dim(self):
        raise NotImplementedError()

    @property
    def embed_dim(self):
        raise NotImplementedError()

    @property
    def num_labels(self):
        raise NotImplementedError()


