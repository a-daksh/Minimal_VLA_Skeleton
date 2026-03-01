import itertools
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vla_model import VLAModel


def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def infinite(loader: DataLoader):
    return itertools.cycle(loader)
