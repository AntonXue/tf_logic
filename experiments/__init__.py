import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve))

from my_datasets import *
from models import *


from .synthetic import *


