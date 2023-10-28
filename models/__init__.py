from .my_models import *
from .hf_models import *


def get_transformer(model_type: str, config=None, **kwargs):
    """ Based on a model_type, generate the configs from **kwargs
    """

    if model_type == "mytf":
        config = MyTfConfig(**kwargs) if config is None else config
        return MyTfModel(config)

    elif model_type == "gpt2":
        config = MyGPT2Config(**kwargs) if config is None else config
        return MyGPT2Model(config)

    else:
        raise NotImplementedError()


