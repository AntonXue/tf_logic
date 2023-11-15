import os
from pathlib import Path
import wandb

""" Some directories """
TFL_ROOT = str(Path(__file__).parent.parent.resolve())
DUMP_DIR = str(Path(TFL_ROOT, "_dump"))
Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)
WANDB_PROJECT = "transformer_friends"


""" WandB setup """
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_DIR"] = DUMP_DIR  # _dump/wandb

