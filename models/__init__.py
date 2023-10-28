from .my_models import *
from .hf_models import *


def get_transformer(model_type: str, **kwargs):
    """ Based on a model_type, generate the configs from **kwargs
    """

    if model_type == "mytf":
        config = MyTransformerConfig(**kwargs)
        return MyTransformer(config)

    elif model_type == "gpt2":
        config = MyGPT2Config(**kwargs)
        return MyGPT2Model(config)

    else:
        raise NotImplementedError()



