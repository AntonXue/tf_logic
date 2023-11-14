from typing import Optional
from .common import default, get_activ
from .my_models import AFBlock, MyTfConfig, MyTfModel, MyTfSeqClsModel
from .hf_models import HFSeqClsConfig, HFSeqClsModel

from .tfl_models.one_shot_models import *
from .tfl_models.next_state_models import *
from .tfl_models.autoreg_ksteps_models import *


class AutoSeqClsModel:
    @classmethod
    def from_config(cls, config):
        if isinstance(config, MyTfConfig):
            return MyTfSeqClsModel(config)
        elif isinstance(config, HFSeqClsConfig):
            return HFSeqClsModel(config)
        else:
            raise ValueError(f"Unrecognized config {config}")

    @classmethod
    def from_kwargs(cls, model_name: str, **kwargs):
        if model_name == "mytf":
            return MyTfSeqClsModel(MyTfConfig(**kwargs))
        else:
            return HFSeqClsModel(HFSeqClsConfig(model_name=model_name, **kwargs))


class AutoTFLModel:
    @classmethod
    def from_config(cls, config):
        pass

    @classmethod
    def from_kwargs(
        cls,
        task_name: str,
        model_name: str,
        input_mode: str = "embeds",  # "embeds" or "string"
        **kwargs
    ):
        # The order of popping and inserting into the kwargs matters!
        if task_name == "one_shot":
            num_vars = kwargs.pop("num_vars")

            kwargs["problem_type"] = "single_label_classification"
            kwargs["num_labels"] = 2

            seqcls_model = AutoSeqClsModel.from_kwargs(model_name, **kwargs)
            config = OneShotTFLConfig(num_vars=num_vars)
            return OneShotEmbedsTFLModel(seqcls_model, config)

        elif task_name == "next_state":
            num_vars = kwargs.pop("num_vars")

            kwargs["problem_type"] = "multi_label_classification"
            kwargs["num_labels"] = num_vars

            seqcls_model = AutoSeqClsModel.from_kwargs(model_name, **kwargs)
            config = NextStateTFLConfig(num_vars=num_vars)
            return NextStateEmbedsTFLModel(seqcls_model, config)

        elif task_name == "autoreg_ksteps":
            num_vars = kwargs.pop("num_vars")
            num_steps = kwargs.pop("num_steps")

            kwargs["problem_type"] = "multi_label_classification"
            kwargs["num_labels"] = num_vars

            seqcls_model = AutoSeqClsModel.from_kwargs(model_name, **kwargs)
            config = AutoRegKStepsTFLConfig(num_vars=num_vars, num_steps=num_steps)
            return AutoRegKStepsEmbedsTFLModel(seqcls_model, config)

        else:
            raise ValueError(f"Unrecognized task_name {task_name}")


