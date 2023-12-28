import os
import sys
from pathlib import Path
import wandb

""" Some directories """
PROJ_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, PROJ_ROOT)

DUMP_DIR = str(Path(PROJ_ROOT, "_dump"))
Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)
WANDB_PROJECT = "transformer_friends"

""" WandB setup """
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_DIR"] = DUMP_DIR  # _dump/wandb
os.environ["WANDB_RESUME"]="allow"

