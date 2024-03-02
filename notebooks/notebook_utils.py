import sys
sys.path.insert(0, "..")

from pathlib import Path
import torch
import wandb
from safetensors import safe_open

from models import *
from my_datasets import *
from experiments import *

NOTEBOOKS_DIR = str(Path(__file__).resolve().parent)


def load_model_from_wandb(
    embed_dim: int,
    num_vars: int,
    num_steps: int,
    model_name: str = "gpt2",
    num_rules_range: tuple[int, int] = (32, 64),
    ante_prob_range: tuple[float, float] = (0.2, 0.3),
    conseq_prob_range: tuple[float, float] = (0.2, 0.3),
    chain_len_range: tuple[int, int] = (4, 7),
    num_trains: int = 65536,
    num_evals: int = 32768,
    batch_size: int = 1024,
    seed: int = 201,
    tag: str = "v0",
    wandb_project: str = "transformer_friends/transformer_friends",
    quiet: bool = False,
    overwrite: bool = False
):

    model = AutoTaskModel.from_kwargs(
        task_name = "autoreg_ksteps",
        num_vars = num_vars,
        model_name = model_name,
        input_dim = 2 * num_vars,
        embed_dim = embed_dim,
        num_layers = 1,
        num_heads = 1,
        num_steps = num_steps
    )

    artifact_id = f"model-SynAR_{model_name}_d{embed_dim}_L1_H1" + \
            f"__nv{num_vars}_ns{num_steps}" + \
            f"_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_cl{chain_len_range[0]}-{chain_len_range[1]}" + \
            f"_ntr{num_trains}_ntt{num_evals}_bsz{batch_size}_seed{seed}" + f":{tag}"

    artifact_dir = Path(NOTEBOOKS_DIR, "artifacts", artifact_id)
    if not artifact_dir.is_dir() or overwrite:
        if not quiet:
            print(f"Querying id: {artifact_id}")

        api = wandb.Api()
        artifact = api.artifact(str(Path(wandb_project, artifact_id)), type="model")

        if not quiet:
            print(f"Downloading: {artifact}")

        artifact_dir = artifact.download()

    artifact_file = Path(artifact_dir, "model.safetensors")

    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
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
