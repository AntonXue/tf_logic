import sys
from pathlib import Path
import torch
import wandb
from safetensors import safe_open

BASE_DIR = str(Path(__file__).parent.parent.parent.resolve())
sys.path.insert(0, BASE_DIR)

from models import *
from my_datasets import *
from experiments import *

DUMP_DIR = str(Path(BASE_DIR, "_dump"))


def load_next_state_model_from_wandb(
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
    train_len: int = 32768,
    eval_len: int = 4096,
    tag: str = "v0",
    wandb_project: str = "transformer_friends/transformer_friends",
    quiet: bool = False,
    overwrite: bool = False
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
            f"_ntr{train_len}_ntt{eval_len}" + f":{tag}"

    artifact_dir = Path(DUMP_DIR, "artifacts", artifact_id)
    if not artifact_dir.is_dir() or overwrite:
        if not quiet:
            print(f"Querying id: {artifact_id}")

        api = wandb.Api()
        artifact = api.artifact(str(Path(wandb_project, artifact_id)), type="model")

        if not quiet:
            print(f"Downloading: {artifact}")

        artifact_dir = artifact.download(artifact_dir)

    artifact_file = Path(artifact_dir, "model.safetensors")

    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model.load_state_dict(tensors)
    model.eval()
    return model
