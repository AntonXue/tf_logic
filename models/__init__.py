from .common import default, get_activ, MySeq2SeqModel
from .my_models import AFBlock, \
        MyTfConfig, MyTfModel, \
        MyTfForSeqClsConfig, MyTfForSeqCls
from .hf_models import MyGPT2Config, MyGPT2Model
from .task_models import *


def get_seq2seq_model(model_type: str, config=None, **kwargs):
    """ Based on a model_type, generate the configs from **kwargs """

    if model_type == "mytf":
        config = MyTfConfig(**kwargs) if config is None else config
        return MyTfModel(config)

    elif model_type == "gpt2":
        config = MyGPT2Config(**kwargs) if config is None else config
        return MyGPT2Model(config)

    else:
        raise NotImplementedError()


def get_seqcls_model(model_type: str, config=None, **kwargs):
    """ Based on a model_type, generate the configs from **kwargs """
    pass


