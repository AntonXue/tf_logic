import torch
from models import AutoregTheoryModel
from my_datasets import AutoregDataset


def load_model_and_dataset(
    num_props: int,
    dataset_len: int,
    models_dir: str,
    do_layer_norm: bool = False,
):
    ln_str = "1" if do_layer_norm else "0"
    saveto = f"theory_n{num_props}_ln{ln_str}_ar3_bsz512_ns8192_lr0.0005.pt"
    save_dict = torch.load(models_dir + "/" + saveto, map_location="cpu")
    model = AutoregTheoryModel(num_props, num_steps=3, do_layer_norm=do_layer_norm)
    model.load_state_dict(save_dict["model_state_dict"])
    model.eval()
    dataset = AutoregDataset(num_props, dataset_len)
    return model, dataset

