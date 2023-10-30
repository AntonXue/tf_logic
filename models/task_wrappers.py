from typing import Optional

import torch

from .model_utils import *


class TaskWrapper(nn.Module):
    def __init__(self, seq2seq_model: MySeq2SeqModel):
        super().__init__()
        self.seq2seq_model = seq2seq_model



class WrapperForQedTask(TaskWrapper):
    """ One-shot QED task (!!! ambitious !!!)
    """
    def __init__(self, seq2seq_model: MySeq2SeqModel):
        super().__init__(seq2seq_model)


    def forward(self, rules, theorem, targets=None):
        pass


class WrapperForOneStepTask(TaskWrapper):
    """ Check whether we can get s' = one_step(rules, s)
    """
    def __init__(self, seq2seq_model: MySeq2SeqModel):
        super().__init__(self, seq2seq_model)

    def forward(self, rules, theorem, targets=None):
        pass

