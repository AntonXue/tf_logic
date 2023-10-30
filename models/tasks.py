from typing import Optional
from dataclasses import dataclass

import torch

from .utils import *

@dataclass
class TaskConfig:
    num_vars: int
    num_rules: int
    seq2seq_model: MySeq2SeqModel


class BaseTaskModel(nn.Module):
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config


""" One-shot QED """

@dataclass
class QedTaskConfig(TaskConfig):
    pass


class QedTaskModel(BaseTaskModel):
    """ One-shot QED task (!!! ambitious !!!)
    """
    def __init__(self, config: QedTaskConfig):
        super().__init__(config)


    def forward(self, rules, theorem, targets=None):
        pass


""" One-step tasks """

@dataclass
class OneStepTaskConfig(TaskConfig):
    pass

class StepTaskModel(BaseTaskModel):
    """ Check whether we can get s' = one_step(rules, s)
    """
    def __init__(self, config: OneStepTaskConfig):
        super().__init__(config)

    def forward(self, rules, theorem, targets=None):
        pass

""" Autoregressive stuff """

@dataclass
class AutoRegStepTaskConfig(TaskConfig):
    num_steps: Optional[int] = None


class AutoRegStepTaskModel(BaseTaskModel):
    """ Autoregressive task stuff
    """
    def __init__(self, config: AutoRegStepTaskConfig):
        super().__init__(config)

    def forward(self, rules, theorem, targets=None):
        pass


