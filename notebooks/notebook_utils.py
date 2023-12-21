import os
import sys
sys.path.insert(0, "..")
import torch
import wandb
from safetensors import safe_open

from models import *
from my_datasets import *
from experiments import *

def load_model_from_wandb(
    model_name: str,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    num_vars: int,
    num_rules_range: tuple[int, int] = (16, 64),
    num_states_range: tuple[int, int] = (8, 32),
    ante_prob_range: tuple[float, float] = (0.3, 0.5),
    conseq_prob_range: tuple[float, float] = (0.2, 0.3),
    state_prob_range: tuple[float, float] = (0.5, 0.5),
    num_trains: int = 32768,
    num_evals: int = 4096,
    tag: str = "v0",
    wandb_project: str = "transformer_friends/transformer_friends",
    quiet: bool = False,
):

    model = AutoTaskModel.from_kwargs(
        task_name = "next_state",
        num_vars = num_vars,
        model_name = model_name,
        input_dim = 1 + 2 * num_vars,
        embed_dim = embed_dim,
        num_layers = num_layers,
        num_heads = num_heads)

    artifact_id = f"model-SynNS_{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}" + \
            f"__nv{num_vars}_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ns{num_states_range[0]}-{num_states_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_sp{state_prob_range[0]:.2f}-{state_prob_range[1]:.2f}" + \
            f"_ntr{num_trains}_ntt{num_evals}" + f":{tag}"

    if not quiet:
        print(f"Querying id: {artifact_id}")

    api = wandb.Api()
    artifact = api.artifact(os.path.join(wandb_project, artifact_id), type="model")

    if not quiet:
        print(f"Downloading: {artifact}")

    artifact_dir = artifact.download()
    artifact_path = os.path.join(artifact_dir, "model.safetensors")
    with safe_open(artifact_path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model.load_state_dict(tensors)
    model.eval()
    return model


def quickload_next_state_model_and_dataset(
    model_name: str = "gpt2",
    embed_dim: int = 1024,
    num_layers: int = 2,
    num_heads: int = 4,
    num_vars: int = 128,
    num_rules_range: tuple[int, int] = (16, 64),
    num_states_range: tuple[int, int] = (8, 32),
    ante_prob_range: tuple[float, float] = (0.3, 0.5),
    conseq_prob_range: tuple[float, float] = (0.2, 0.3),
    state_prob_range: tuple[float, float] = (0.5, 0.5),
    dataset_len = 8192
):
    model = load_model_from_wandb(
        model_name = model_name,
        embed_dim = embed_dim,
        num_layers = num_layers,
        num_heads = num_heads,
        num_vars = num_vars,
        num_rules_range = num_rules_range,
        num_states_range = num_states_range,
        ante_prob_range = ante_prob_range,
        conseq_prob_range = conseq_prob_range,
        state_prob_range = state_prob_range,
        num_trains = 32768,
        num_evals = 8192,
        quiet = True)

    dataset = NextStateTokensDataset(
        num_vars = num_vars,
        num_rules_range = num_rules_range,
        num_states_range = num_states_range,
        ante_prob_range = ante_prob_range,
        conseq_prob_range = conseq_prob_range,
        state_prob_range = state_prob_range,
        dataset_len = dataset_len)

    return model, dataset
