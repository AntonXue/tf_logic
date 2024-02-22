from typing import Optional
from transformers import AutoModelForSequenceClassification

from .common import *
from .seqcls_models import *
from .task_models import *
from .attack_models import *


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
            return HFSeqClsModel(HFSeqClsConfig(model_name, **kwargs))

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        if model_name == "mytf":
            raise ValueError(f"No pretraining for mytf yet")
        else:
            return HFSeqClsModel(HFSeqClsConfig(model_name, use_pretrained=True, **kwargs))


class AutoTaskModel:
    @classmethod
    def from_config(cls, config):
        pass

    @classmethod
    def from_kwargs(
        cls,
        task_name: str,
        model_name: str,
        input_mode: str = "tokens",  # "tokens" or "string"
        **kwargs
    ):
        assert not (model_name == "mytf" and input_mode == "string")

        # The order of popping and inserting into the kwargs matters!
        if task_name == "one_shot":
            num_vars = kwargs.pop("num_vars")
            kwargs["problem_type"] = "single_label_classification"
            kwargs["num_labels"] = 2
            seqcls_model = AutoSeqClsModel.from_kwargs(model_name, **kwargs)
            return OneShotTaskModel(seqcls_model)

        elif task_name == "one_shot_str":
            num_vars = kwargs.pop("num_vars")
            kwargs["problem_type"] = "single_label_classification"
            kwargs["num_labels"] = 2
            seqcls_model = AutoSeqClsModel.from_kwargs(model_name, **kwargs)
            return OneShotStringTaskModel(seqcls_model)

        elif task_name == "autoreg_ksteps":
            num_vars = kwargs.pop("num_vars")
            num_steps = kwargs.pop("num_steps")
            kwargs["problem_type"] = "multi_label_classification"
            kwargs["num_labels"] = num_vars
            seqcls_model = AutoSeqClsModel.from_kwargs(model_name, **kwargs)
            return AutoregKStepsTaskModel(seqcls_model, num_steps=num_steps, train_supervision_mode="all")

        elif task_name == "sf_autoreg_ksteps":
            num_vars = kwargs.pop("num_vars")
            num_steps = kwargs.pop("num_steps")
            kwargs["problem_type"] = "multi_label_classification"
            kwargs["num_labels"] = num_vars
            seqcls_model = AutoSeqClsModel.from_kwargs(model_name, **kwargs)
            return AutoregKStepsTaskModel(seqcls_model, num_steps=num_steps, train_supervision_mode="first")

        else:
            raise NotImplementedError(f"{task_name} not supported")

    @classmethod
    def from_pretrained(
        cls,
        task_name: str,
        model_name: str,
        **kwargs
    ):
        # The order of popping and inserting into the kwargs matters!
        if task_name == "one_shot":
            num_vars = kwargs.pop("num_vars")
            kwargs["problem_type"] = "single_label_classification"
            kwargs["num_labels"] = 2
            seqcls_model = AutoSeqClsModel.from_pretrained(model_name, **kwargs)
            return OneShotTaskModel(seqcls_model)

        elif task_name == "one_shot_str":
            num_vars = kwargs.pop("num_vars")
            kwargs["problem_type"] = "single_label_classification"
            kwargs["num_labels"] = 2
            seqcls_model = AutoSeqClsModel.from_pretrained(model_name, **kwargs)
            return OneShotStringTaskModel(seqcls_model)

        elif task_name == "autoreg_ksteps":
            num_vars = kwargs.pop("num_vars")
            num_steps = kwargs.pop("num_steps")
            kwargs["problem_type"] = "multi_label_classification"
            kwargs["num_labels"] = num_vars
            seqcls_model = AutoSeqClsModel.from_pretrained(model_name, **kwargs)
            return AutoregKStepsTaskModel(seqcls_model, num_steps=num_steps, train_supervision_mode="all")

        elif task_name == "sf_autoreg_ksteps":
            num_vars = kwargs.pop("num_vars")
            num_steps = kwargs.pop("num_steps")
            kwargs["problem_type"] = "multi_label_classification"
            kwargs["num_labels"] = num_vars
            seqcls_model = AutoSeqClsModel.from_pretrained(model_name, **kwargs)
            return AutoregKStepsTaskModel(seqcls_model, num_steps=num_steps, train_supervision_mode="first")

        else:
            raise NotImplementedError(f"{task_name} not supported")


