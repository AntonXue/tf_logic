import sys
from pathlib import Path
import torch
import wandb
from safetensors import safe_open
import os

BASE_DIR = str(Path(__file__).parent.parent.parent.resolve())
sys.path.insert(0, BASE_DIR)

from models import *
from my_datasets import *
from experiments import *

DUMP_DIR = str(Path(BASE_DIR, "_dump"))

def download_artifact(
    artifact_id: str,
    artifact_dir: str,
    wandb_project: str = "transformer_friends/transformer_friends",
    quiet: bool = False,
    overwrite: bool = False,
    artifact_type: str = "model",
    raise_exception_if_not_found: str = True
):
    if not artifact_dir.is_dir() or overwrite:
        if not quiet:
            print(f"Querying id: {artifact_id}")

        try:
            api = wandb.Api()
            artifact = api.artifact(str(Path(wandb_project, artifact_id)), type=artifact_type)

            if not quiet:
                print(f"Downloading: {artifact}")
            artifact_dir = artifact.download(artifact_dir)
        except Exception as e:
            if raise_exception_if_not_found:
                raise Exception(e)
            artifact_dir = None
    return artifact_dir

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
    overwrite: bool = False,
    task_name: str = "next_state",
    num_steps: int = 3,
    chain_len_range: tuple[int, int] = (2, 5)
):

    if task_name == "next_state":
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
        
    elif task_name == "autoreg_ksteps":
        model = AutoTaskModel.from_kwargs(
            task_name = "autoreg_ksteps",
            num_vars = num_vars,
            model_name = model_name,
            input_dim = 1 + 2 * num_vars,
            embed_dim = embed_dim,
            num_layers = num_layers,
            num_heads = num_heads,
            num_steps = num_steps)
        artifact_id = f"model-SynAR_{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}" + \
            f"__nv{num_vars}_ns{num_steps}" + \
            f"_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_cl{chain_len_range[0]}-{chain_len_range[1]}" + \
            f"_ntr{train_len}_ntt{eval_len}" + f":{tag}"
        
    else:
        raise Exception(f"Unknown Task: {task_name}")

    artifact_dir = Path(DUMP_DIR, "artifacts", artifact_id)
    artifact_dir = download_artifact(artifact_id=artifact_id, 
                                     artifact_dir=artifact_dir,
                                     wandb_project=wandb_project,
                                     quiet=quiet,
                                     overwrite=overwrite)

    artifact_file = Path(artifact_dir, "model.safetensors")

    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model.load_state_dict(tensors)
    model.eval()
    return model

def load_checkpoint_from_wandb(
        experiment_out_dir: str,
        experiment_id: str
):
    if os.path.isdir(experiment_out_dir):
        # First try fetching checkpoint from local logs
        checkpoints = os.listdir(experiment_out_dir)
        checkpoints = sorted([int(name.split("checkpoint-")[-1]) for name in checkpoints if name.startswith("checkpoint")])
        if len(checkpoints) > 0:
            # pick the latest checkpoint
            latest_checkpoint = f"checkpoint-{checkpoints[-1]}"
            return str(Path(experiment_out_dir, latest_checkpoint))
    
    # download the latest checkpoint
    # This may not always give the latest epoch
    checkpoint_id = "checkpoint-" + experiment_id + ":latest"
    artifact_dir = Path(DUMP_DIR, "artifacts", checkpoint_id)
    checkpoint = download_artifact(artifact_id=checkpoint_id, artifact_dir=artifact_dir, raise_exception_if_not_found=False)
    return checkpoint

def load_stats_from_wandb(
    model_name: str,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    num_vars: int,
    num_steps: int = 3,
    chain_len_range: tuple[int, int] = (2, 5),
    num_rules_range: tuple[int, int] = (16, 64),
    num_states_range: tuple[int, int] = (8, 32),
    ante_prob_range: tuple[float, float] = (0.3, 0.5),
    conseq_prob_range: tuple[float, float] = (0.2, 0.3),
    state_prob_range: tuple[float, float] = (0.5, 0.5),
    train_len: int = 32768,
    eval_len: int = 4096,
    wandb_project: str = "transformer_friends/transformer_friends",
    syn_exp_name: str = "next_state",
    seed: int = 101,
    return_first: bool = True
):
    if syn_exp_name == "next_state":
        run_name = f"SynNS_{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}" + \
            f"__nv{num_vars}_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ns{num_states_range[0]}-{num_states_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_sp{state_prob_range[0]:.2f}-{state_prob_range[1]:.2f}" + \
            f"_ntr{train_len}_ntt{eval_len}_seed{seed}"
        
    elif syn_exp_name == "autoreg_ksteps":
        run_name = f"SynAR_{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}" + \
            f"__nv{num_vars}_ns{num_steps}" + \
            f"_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_cl{chain_len_range[0]}-{chain_len_range[1]}" + \
            f"_ntr{train_len}_ntt{eval_len}_seed{seed}"   
    else:
        raise Exception(f"Unknown Task: {syn_exp_name}")

    api = wandb.Api()

    runs = api.runs(wandb_project, filters={"config.run_name": run_name})

    if len(runs) == 0:
        raise Exception(f"No runs found for run_name: {run_name}")
    elif len(runs) > 1:
        if not return_first:
            raise Exception(f"Multiple runs found for run_name: {run_name}")
    
    run = runs[0]
    return run.summary
