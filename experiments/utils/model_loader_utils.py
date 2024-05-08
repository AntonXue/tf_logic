import sys
from pathlib import Path
import torch
import wandb
from safetensors import safe_open
import torch
import torch.nn as nn
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


def load_checkpoint_from_wandb(
    experiment_out_dir: str,
    experiment_id: str
):
    if os.path.isdir(experiment_out_dir):
        # First try fetching checkpoint from local logs
        checkpoints = os.listdir(experiment_out_dir)
        checkpoints = sorted([
            int(name.split("checkpoint-")[-1])
            for name in checkpoints if name.startswith("checkpoint")
        ])

        if len(checkpoints) > 0:
            # pick the latest checkpoint
            latest_checkpoint = f"checkpoint-{checkpoints[-1]}"
            return str(Path(experiment_out_dir, latest_checkpoint))
    
    # download the latest checkpoint
    # This may not always give the latest epoch
    checkpoint_id = "checkpoint-" + experiment_id + ":latest"
    artifact_dir = Path(DUMP_DIR, "artifacts", checkpoint_id)
    checkpoint = download_artifact(
        artifact_id = checkpoint_id,
        artifact_dir = artifact_dir,
        raise_exception_if_not_found = False
    )
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
    learning_rate: Optional[float] = None,
    train_len: int = 32768,
    eval_len: int = 4096,
    wandb_project: str = "transformer_friends/transformer_friends",
    syn_exp_name: str = "next_state",
    seed: int = 101,
    train_batch_size: int = 64,
    return_first: bool = True,
    include_seed_in_run_name: bool = True,
    include_train_batch_size_in_run_name: bool = True,  # To support recent name changes
    exp_type: str = None, # Set this to "attack" if you want to load attack results
    attack_params: dict = None # Set this to the attack params if you want to load attack results
):
    print(f"LOAD STATS FROM WANDB CALLED, {syn_exp_name}")
    if syn_exp_name == "next_state":
        run_name = f"SynNS_{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}" + \
            f"__nv{num_vars}_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ns{num_states_range[0]}-{num_states_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_sp{state_prob_range[0]:.2f}-{state_prob_range[1]:.2f}" + \
            f"_ntr{train_len}_ntt{eval_len}"
        
    elif syn_exp_name == "autoreg_ksteps":
        run_name = f"SynAR_{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}" + \
            f"__nv{num_vars}_ns{num_steps}" + \
            f"_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_cl{chain_len_range[0]}-{chain_len_range[1]}" + \
            f"_ntr{train_len}_ntt{eval_len}" 
    else:
        raise Exception(f"Unknown Task: {syn_exp_name}")

    if include_train_batch_size_in_run_name:
        run_name += f"_bsz{train_batch_size}"

    if learning_rate is not None:
        run_name += f"_lr{learning_rate:.5f}"

    if include_seed_in_run_name:
        run_name += f"_seed{seed}"

    if exp_type == "attack" and syn_exp_name == "autoreg_ksteps":
        run_name = f"SynARAttack_{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}" + \
            f"_nv{num_vars}_ns{num_steps}" + \
            f"_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_seed{seed}" + \
            f"_atk-{attack_params['base_attack_model_name']}_nat{attack_params['num_attack_tokens']}" + \
            f"_atkntr{attack_params['attack_train_len']}_atkntt{attack_params['attack_eval_len']}"
        
    api = wandb.Api()

    runs = api.runs(wandb_project, filters={"config.run_name": run_name})

    if len(runs) == 0:
        raise Exception(f"No runs found for run_name: {run_name}")
    elif len(runs) > 1:
        if not return_first:
            raise Exception(f"Multiple runs found for run_name: {run_name}")
    run = runs[0]
    return run.summary


def load_model_and_dataset_from_big_grid(
    embed_dim: int,
    num_vars: int,
    model_name: str = "gpt2",
    num_layers: int = 1,
    num_heads: int = 1,
    dataset_len: Optional[int] = None,
    num_rules: int = 32,
    exph: float = 3.0,
    train_len: int = 262144,
    eval_len: int = 65536,
    num_train_steps: int = 8192,
    learning_rate: float = 1e-4,
    batch_size: int = 512,
    seed: int = 591,
    quiet: bool = False,
    wandb_project: str = "transformer_friends/transformer_friends",
    overwrite: bool = True,
    max_test_seq_len: int = 1024,
):
    seqcls_model = MyGPT2SeqClsModel(MyGPT2Config(
        input_dim = 2 * num_vars,
        num_vars = num_vars,
        embed_dim = embed_dim,
        num_heads = 1,
        num_layers = 1,
        use_positional_embedding = False,
        max_seq_len = max_test_seq_len,
    ))

    model = AutoregKStepsTaskModel(
        seqcls_model = seqcls_model,
        num_steps = 3,
        train_supervision_mode = "all"
    )

    model_str = f"{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}"
    dataset_str = f"DMD_nv{num_vars}_nr{num_rules}_exph{exph:.3f}"
    train_str = f"ntr{train_len}_ntt{eval_len}_bsz{batch_size}" + \
        f"_steps{num_train_steps}_lr{learning_rate:.5f}"

    artifact_id = f"model-SynSAR_{model_str}__{dataset_str}__{train_str}_seed{seed}:v0"

    artifact_dir = Path(DUMP_DIR, "artifacts", artifact_id)
    artifact_dir = download_artifact(
        artifact_id = artifact_id, 
        artifact_dir = artifact_dir,
        wandb_project = wandb_project,
        quiet = quiet,
        overwrite = overwrite
    )

    artifact_file = Path(artifact_dir, "model.safetensors")
    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    # Delete the positional embedding key
    del tensors["seqcls_model.gpt2s.transformer.wpe.weight"]

    model.load_state_dict(tensors, strict=False)

    # Modify the wpe to be of the right sequence length
    model.seqcls_model.gpt2s.transformer.wpe = nn.Embedding(max_test_seq_len, embed_dim)
    model.seqcls_model.gpt2s.transformer.wpe.requires_grad_(False)
    model.seqcls_model.gpt2s.transformer.wpe.weight.fill_(0)
    model.eval()

    dataset_len = train_len if dataset_len is None else dataset_len
    dataset = AutoregDiamondTokensDataset(
        num_vars = num_vars,
        num_rules = num_rules,
        exp_hots = exph,
        dataset_len = dataset_len
    )

    return model, dataset


def load_big_grid_stats_from_wandb(
    embed_dim: int,
    num_vars: int,
    model_name: str = "gpt2",
    num_layers: int = 1,
    num_heads: int = 1,
    num_rules: int = 32,
    exph: float = 3.0,
    train_len: int = 262144,
    eval_len: int = 65536,
    num_train_steps: int = 8192,
    learning_rate: float = 5e-4,
    batch_size: int = 512,
    seed: int = 591,
    quiet: bool = False,
    wandb_project: str = "transformer_friends/transformer_friends",
    overwrite: bool = True,
    max_test_seq_len: int = 1024,
    return_first: bool = True
):
    model_str = f"{model_name}_d{embed_dim}_L{num_layers}_H{num_heads}"
    dataset_str = f"DMD_nv{num_vars}_nr{num_rules}_exph{exph:.3f}"
    train_str = f"ntr{train_len}_ntt{eval_len}_bsz{batch_size}" + \
        f"_steps{num_train_steps}_lr{learning_rate:.5f}"

    run_name = f"SynSAR_{model_str}__{dataset_str}__{train_str}_seed{seed}"

    api = wandb.Api()
    runs = api.runs(wandb_project, filters={"config.run_name": run_name})

    if len(runs) == 0:
        raise Exception(f"No runs found for run_name {run_name}")
    elif len(runs) > 1:
        if not return_first:
            raise Exception(f"Multiple runs found for run_name {run_name}")
    run = runs[0]
    return run.summary


