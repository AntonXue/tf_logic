import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve))

from .synthetic import *


class AutoTrainer:
    @classmethod
    def from_synthetic_experiment_args(
        cls,
        args: SyntheticExperimentArguments,
        **kwargs
    ):
        return make_trainer_for_synthetic(args, **kwargs)


