from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
from transformers import Trainer, TrainingArguments

from .common import *

from models import *
from my_datasets import *


class SyntheticArguments: pass


@dataclass
class SyntheticOneShotArguments(SyntheticArguments):
    num_rules: int
    num_vars: int

    model_name: str
    embed_dim: int
    num_layers: int
    num_heads: int

    train_len: int
    test_len: int
    ante_prob: float
    conseq_prob: float
    theorem_prob: float
    seed: int = 1234

    num_epochs: int = 100

    @property
    def wandb_run_name(self):
        return f"SynOneShot_nr{self.num_rules}_nv{self.num_vars}" + \
               f"_{self.model_name}_d{self.embed_dim}_L{self.num_layers}_H{self.num_heads}" + \
               f"_ap{self.ante_prob:.2f}_bp{self.conseq_prob}_tp{self.theorem_prob}" + \
               f"_ntr{self.train_len}_ntt{self.test_len}"


@dataclass
class SyntheticNextStateArguments(SyntheticArguments):
    num_rules: int
    num_vars: int

    model_name: str
    embed_dim: int
    num_layers: int
    num_heads: int

    train_len: int
    test_len: int
    ante_prob: float
    conseq_prob: float
    state_prob: float
    seed: int = 1234

    num_epochs: int = 100

    @property
    def wandb_run_name(self):
        return f"SynNextState_nr{self.num_rules}_nv{self.num_vars}" + \
               f"_{self.model_name}_d{self.embed_dim}_L{self.num_layers}_H{self.num_heads}" + \
               f"_ap{self.ante_prob:.2f}_bp{self.conseq_prob}_sp{self.state_prob}" + \
               f"_ntr{self.train_len}_ntt{self.test_len}"


@dataclass
class SyntheticAutoRegKStepsArguments(SyntheticArguments):
    num_rules: int
    num_vars: int
    num_steps: int

    model_name: str
    embed_dim: int
    num_layers: int
    num_heads: int

    train_len: int
    test_len: int
    ante_prob: float
    conseq_prob: float
    state_prob: float
    seed: int = 1234

    num_epochs: int = 100

    @property
    def wandb_run_name(self):
        return f"SynARKSteps_nr{self.num_rules}_nv{self.num_vars}_ns{self.num_steps}" + \
               f"_{self.model_name}_d{self.embed_dim}_L{self.num_layers}_H{self.num_heads}" + \
               f"_ap{self.ante_prob:.2f}_bp{self.conseq_prob}_sp{self.state_prob}" + \
               f"_ntr{self.train_len}_ntt{self.test_len}"


class AutoTrainer:
    @classmethod
    def for_synthetic(
        cls,
        args: SyntheticArguments,
        output_dir: str = "trainer-test",
        report_to: str = "wandb"
    ):
        """ Make a Hugging Face Trainer object """
        assert hasattr(args, "wandb_run_name")

        if isinstance(args, SyntheticOneShotArguments):
            train_dataset = OneShotEmbedsDataset(
                num_rules = args.num_rules,
                num_vars = args.num_vars,
                ante_prob = args.ante_prob,
                conseq_prob = args.conseq_prob,
                theorem_prob = args.theorem_prob,
                dataset_len = args.train_len,
                seed = args.seed)

            test_dataset = OneShotEmbedsDataset(
                num_rules = args.num_rules,
                num_vars = args.num_vars,
                ante_prob = args.ante_prob,
                conseq_prob = args.conseq_prob,
                theorem_prob = args.theorem_prob,
                dataset_len = args.test_len,
                seed = args.seed)

            task_model = AutoTFLModel.from_kwargs(
                task_name = "one_shot",
                num_vars = args.num_vars,
                model_name = args.model_name,
                embed_dim = args.embed_dim,
                num_layers = args.num_layers,
                num_heads = args.num_heads)

            training_args = TrainingArguments(
                output_dir,
                num_train_epochs = args.num_epochs,
                evaluation_strategy = "epoch",
                report_to = report_to,
                run_name = args.wandb_run_name,
                logging_steps = 5)

            return Trainer(
                task_model,
                training_args,
                train_dataset = train_dataset,
                eval_dataset = test_dataset,
                compute_metrics = one_shot_metrics)


        elif isinstance(args, SyntheticNextStateArguments):
            train_dataset = NextStateEmbedsDataset(
                num_rules = args.num_rules,
                num_vars = args.num_vars,
                ante_prob = args.ante_prob,
                conseq_prob = args.conseq_prob,
                state_prob = args.state_prob,
                dataset_len = args.train_len,
                seed = args.seed)

            test_dataset = NextStateEmbedsDataset(
                num_rules = args.num_rules,
                num_vars = args.num_vars,
                ante_prob = args.ante_prob,
                conseq_prob = args.conseq_prob,
                state_prob = args.state_prob,
                dataset_len = args.test_len,
                seed = args.seed)

            task_model = AutoTFLModel.from_kwargs(
                task_name = "next_state",
                num_vars = args.num_vars,
                model_name = args.model_name,
                embed_dim = args.embed_dim,
                num_layers = args.num_layers,
                num_heads = args.num_heads)

            training_args = TrainingArguments(
                output_dir,
                num_train_epochs = args.num_epochs,
                evaluation_strategy = "epoch",
                report_to = report_to,
                run_name = args.wandb_run_name,
                logging_steps = 5)

            return Trainer(
                task_model,
                training_args,
                train_dataset = train_dataset,
                eval_dataset = test_dataset,
                compute_metrics = next_state_metrics)


        elif isinstance(args, SyntheticAutoRegKStepsArguments):
            train_dataset = AutoRegKStepsEmbedsDataset(
                num_rules = args.num_rules,
                num_vars = args.num_vars,
                num_steps = args.num_steps,
                ante_prob = args.ante_prob,
                conseq_prob = args.conseq_prob,
                state_prob = args.state_prob,
                dataset_len = args.train_len,
                seed = args.seed)

            test_dataset = AutoRegKStepsEmbedsDataset(
                num_rules = args.num_rules,
                num_vars = args.num_vars,
                num_steps = args.num_steps,
                ante_prob = args.ante_prob,
                conseq_prob = args.conseq_prob,
                state_prob = args.state_prob,
                dataset_len = args.test_len,
                seed = args.seed)

            task_model = AutoTFLModel.from_kwargs(
                task_name = "autoreg_ksteps",
                num_vars = args.num_vars,
                num_steps = args.num_steps,
                model_name = args.model_name,
                embed_dim = args.embed_dim,
                num_layers = args.num_layers,
                num_heads = args.num_heads)

            training_args = TrainingArguments(
                output_dir,
                num_train_epochs = args.num_epochs,
                evaluation_strategy = "epoch",
                report_to = report_to,
                run_name = args.wandb_run_name,
                logging_steps = 5)

            return Trainer(
                task_model,
                training_args,
                train_dataset = train_dataset,
                eval_dataset = test_dataset,
                compute_metrics = autoreg_ksteps_metrics)

        else:
            raise ValueError(f"Unrecognized args {args}")






