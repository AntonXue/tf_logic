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

WANDB_PROJECT = "transformer_friends/transformer_friends"


def download_artifact(artifact_id, quiet=False, overwrite=False):
    artifact_dir = Path(NOTEBOOKS_DIR, "artifacts", artifact_id)
    if not artifact_dir.is_dir() or overwrite:
        if not quiet:
            print(f"Querying id: {artifact_id}")
        api = wandb.Api()
        artifact = api.artifact(str(Path(WANDB_PROJECT, artifact_id)), type="model")
        if not quiet:
            print(f"Downloading: {artifact}")
        artifact_dir = artifact.download()
    artifact_file = Path(artifact_dir, "model.safetensors")
    return artifact_file


def load_arks_model_and_dataset(
    embed_dim: int,
    num_vars: int,
    num_steps: int,
    model_name: str = "gpt2",
    num_rules_range: tuple[int, int] = (32, 64),
    ante_prob_range: tuple[float, float] = (0.25, 0.25),
    conseq_prob_range: tuple[float, float] = (0.25, 0.25),
    chain_len_range: tuple[int, int] = (4, 7),
    num_trains: int = 131072,
    num_evals: int = 65536,
    learning_rate: float = 5e-4,
    batch_size: int = 64,
    seed: int = 401,
    tag: str = "v0",
    quiet: bool = False,
):
    seqcls_model = MyGPT2SeqClsModel(MyGPT2Config(
        input_dim = 2 * num_vars,
        num_vars = num_vars,
        embed_dim = embed_dim,
        num_heads = 1,
        num_layers = 1,
        use_positional_embedding = False
    ))

    model = AutoregKStepsTaskModel(
        seqcls_model = seqcls_model,
        num_steps = num_steps,
        train_supervision_mode = "all"
    )

    artifact_id = f"model-SynAR_{model_name}_d{embed_dim}_L1_H1" + \
            f"__nv{num_vars}_ns{num_steps}" + \
            f"_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_cl{chain_len_range[0]}-{chain_len_range[1]}" + \
            f"_ntr{num_trains}_ntt{num_evals}" + \
            f"_bsz{batch_size}" + \
            f"_lr{learning_rate:.5f}" + \
            f"_seed{seed}:{tag}"

    artifact_file = download_artifact(artifact_id)

    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model.load_state_dict(tensors)
    model.eval()
    dataset = AutoregKStepsTokensDataset(
        num_vars = num_vars,
        num_rules_range = num_rules_range,
        ante_prob_range = ante_prob_range,
        conseq_prob_range = conseq_prob_range,
        chain_len_range = chain_len_range,
        num_prevs_range = (1, chain_len_range[0]),
        num_steps = num_steps,
        dataset_len = num_evals
    )

    return model, dataset



def load_mytfs_model_and_dataset(
    # embed_dim: int,
    num_vars: int,
    attention_style: str,
    do_layer_norm: bool,
    num_rules_range: tuple[int, int] = (32, 64),
    ante_prob_range: tuple[float, float] = (0.2, 0.3),
    conseq_prob_range: tuple[float, float] = (0.2, 0.3),
    chain_len_range: tuple[int, int] = (4, 7),
    num_trains: int = 131072,
    num_evals: int = 65536,
    batch_size: int = 512,
    seed: int = 501,
    tag: str = "v0",
    quiet: bool = False,
):

    embed_dim = 2 * num_vars + 1

    model = MyTfSuccTaskModel(
        num_vars = num_vars,
        embed_dim = embed_dim,
        num_layers = 1,
        num_heads = 1,
        do_layer_norm = do_layer_norm,
        attention_style = attention_style,
        use_nn_linear_bias = False,
    )

    artifact_id = f"model-SynMyTfS_d{embed_dim}_LNone_HNone" + \
            f"_{attention_style}" + \
            ("_LN1" if do_layer_norm else "_LN0") + \
            f"__nv{num_vars}" + \
            f"_nr{num_rules_range[0]}-{num_rules_range[1]}" + \
            f"_ap{ante_prob_range[0]:.2f}-{ante_prob_range[1]:.2f}" + \
            f"_bp{conseq_prob_range[0]:.2f}-{conseq_prob_range[1]:.2f}" + \
            f"_cl{chain_len_range[0]}-{chain_len_range[1]}" + \
            f"_ntr{num_trains}_ntt{num_evals}_bsz{batch_size}_seed{seed}" + f":{tag}"

    artifact_file = download_artifact(artifact_id, overwrite=True)
    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model.load_state_dict(tensors)
    model.eval()
    dataset = MyTfSuccTokensDataset(
        num_vars = num_vars,
        num_rules_range = num_rules_range,
        ante_prob_range = ante_prob_range,
        conseq_prob_range = conseq_prob_range,
        chain_len_range = chain_len_range,
        num_prevs_range = (1, chain_len_range[0]),
        dataset_len = num_evals
    )
    
    return model, dataset


def load_small_simple_dataset(num_vars, dataset_len=1000):
    return SmallTfSuccTokensDataset(num_vars, dataset_len=dataset_len)


def load_sms_gpt2(n, d, lr, seed, dataset="Dsimple", ap=(0.2, 0.8), bp=(0.2, 0.8), lf="bce", als=1.0):
    apstr = f"ap{ap[0]:.2f}-{ap[1]:.2f}"
    bpstr = f"bp{bp[0]:.2f}-{bp[1]:.2f}"
    artifact_id = f"model-SMS_gpt2_n{n}_d{d}_{lf}_als{als:.3f}" + \
                f"_{dataset}_n{n}_{apstr}_{bpstr}_lr{lr:.5f}_seed{seed}:v0"
    artifact_file = download_artifact(artifact_id, overwrite=True, quiet=False)
    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model = SmallGPT2(n, d, loss_fn=lf, attn_loss_scale = als)
    model.load_state_dict(tensors)
    model.eval()
    return model


def load_tfa(n, d, loss_fn, seed, dataset="Dsimple"):
    artifact_id = f"model-SMS_tfa_n{n}_d{d}_{loss_fn}_{dataset}_n{n}_ap0.5_bp0.5_seed{seed}:v0"
    artifact_file = download_artifact(artifact_id)
    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model = SmallTfA(n, d, loss_fn)
    model.load_state_dict(tensors)
    model.eval()
    return model


def load_tfb(n, loss_fn, seed, dataset="Dsimple"):
    artifact_id = f"model-SMS_tfb_n{n}_{loss_fn}_{dataset}_n{n}_ap0.5_bp0.5_seed{seed}:v0"
    artifact_file = download_artifact(artifact_id)
    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model = SmallTfB(n, loss_fn)
    model.load_state_dict(tensors)
    model.eval()
    return model


def load_tfe(n, attn_fn, iv, seed, lr=5e-4, dataset="Dsimple"):
    ivstr = "IVR" if iv is None else f"IV{init_value}"
    artifact_id = f"model-SMS_tfe_n{n}_{attn_fn}_{ivstr}" + \
                f"_{dataset}_n{n}_ap0.5_bp0.5_lr{lr:.5f}_seed{seed}:v0"
    artifact_file = download_artifact(artifact_id)
    with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    model = SmallTfE(n, attn_fn, init_value=iv)
    model.load_state_dict(tensors)
    model.eval()
    return model