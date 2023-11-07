from .common import default, get_activ, MySeq2SeqModel
from .my_models import AFBlock, MyTfConfig, MyTfModel, MyTfForSeqCls
from .hf_models import MyGPT2Config, MyGPT2Model, MyGPT2ForSeqCls
from .task_models import *


def get_seq2seq_model(model_type: str, config=None, **kwargs):
    """ Based on a model_type, generate the configs from **kwargs """

    if model_type == "mytf":
        config = default(config, MyTfConfig(**kwargs))
        return MyTfModel(config)

    elif model_type == "gpt2":
        config = default(config, MyGPT2Config(**kwargs))
        return MyGPT2Model(config)

    else:
        raise NotImplementedError()


def get_seqcls_model(
        model_type: str,
        config = None,
        num_labels: Optional[int] = None,
        problem_type: Optional[str] = None,
        **kwargs):
    """ Based on a model_type, generate the configs from **kwargs """

    if model_type == "mytf":
        config = default(config, MyTfConfig(num_labels=num_labels, problem_type=problem_type, **kwargs))
        return MyTfForSeqCls(config)

    elif model_type == "gpt2":
        config = default(config, MyGPT2Config(num_labels=num_labels, problem_type=problem_type,
                pad_token_id=0, **kwargs))
        return MyGPT2ForSeqCls(config)

    else:
        raise NotImplementedError()



