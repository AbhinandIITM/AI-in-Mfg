import os
import numpy as np
import torch
import random

def set_seed(seed):
    """
    Set seed globally for reproducibility across all libraries.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder='big')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
