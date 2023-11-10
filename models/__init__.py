from typing import Optional
from .common import default, get_activ
from .my_models import AFBlock, MyTfConfig, MyTfModel, MyTfSeqClsModel
from .hf_models import HFSeqClsConfig, HFSeqClsModel

from .task_models.one_shot_qed_models import *
from .task_models.one_step_state_models import *
from .task_models.autoreg_fixed_steps_models import *


def get_seq2seq_model(model_name: str, config=None, **kwargs):
    """ Based on a model_name, generate the configs from **kwargs """
    if model_name == "mytf":
        config = default(config, MyTfConfig(**kwargs))
        return MyTfModel(config)

    else:
        raise NotImplementedError()


def get_seqcls_model(model_name: str, config=None, **kwargs):
    """ Based on a model_name, generate the configs from **kwargs """
    if model_name == "mytf":
        config = default(config, MyTfConfig(**kwargs))
        return MyTfSeqClsModel(config)

    else:
        config = default(config, HFSeqClsConfig(model_name=model_name, **kwargs))
        return HFSeqClsModel(config)


def get_task_model(
    task_name: Optional[str] = None,
    task_config: Optional[TaskConfig] = None,
    num_vars: Optional[int] = None, # This is num_labels for succ and autoreg
    num_steps: Optional[int] = None,
    model_name: Optional[str] = None,
    **kwargs
):
    """ Instantiate a model_name for a particular task """
    # If the task_config is provided, then that's easy for us and we don't even need the task_name string
    if task_config is not None:
        if isinstance(task_config, OneShotQedTaskConfig):
            return OneShotQedEmbedsTaskModel(task_config)

        elif isinstance(task_config, OneStepStateTaskConfig):
            return OneStepStateEmbedsTaskModel(task_config)

        elif isinstance(task_config, AutoRegFixedStepsTaskConfig):
            return AutoRegFixedStepsEmbedsTaskModel(task_config)

        else:
            raise ValueError(f"Unrecognized task_config {task_config}")

    # Otherwise we need to make the task_config ourselves

    # Set up the seqcls_model
    if task_name == "oneshot_qed":
        problem_type = "single_label_classification"
    elif task_name in ["predict_successor", "autoreg_fixed_steps"]:
        problem_type = "multi_label_classification"
    else:
        raise ValueError(f"Unsupported task_name {task_name}")

    kwargs = {"problem_type" : problem_type, **kwargs}  # May overwrite problem_type

    # Shortcut for num_vars and num_labels, if applicable
    if "num_labels" not in kwargs and task_name == "oneshot_qed":
        kwargs = {"num_labels" : 2, **kwargs}
    elif "num_labels" not in kwargs:
        kwargs = {"num_labels" : num_vars, **kwargs}

    seqcls_model = get_seqcls_model(model_name, **kwargs)

    if task_name == "oneshot_qed":
        assert num_vars is not None
        task_config = OneShotQedTaskConfig(
            num_vars = num_vars,
            seqcls_model = seqcls_model)
        return OneShotQedEmbedsTaskModel(task_config)

    elif task_name == "predict_successor":
        assert num_vars is not None
        task_config = OneStepStateTaskConfig(
            num_vars = num_vars,
            seqcls_model = seqcls_model)
        return OneStepStateEmbedsTaskModel(task_config)

    elif task_name == "autoreg_fixed_steps":
        assert num_vars is not None
        assert num_steps is not None
        task_config = AutoRegFixedStepsTaskConfig(
            num_vars = num_vars,
            num_steps = num_steps,
            seqcls_model = seqcls_model)
        return AutoRegFixedStepsEmbedsTaskModel(task_config)

    else:
        raise ValueError(f"Unrecognized task_name {task_name}")


