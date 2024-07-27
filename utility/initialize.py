import random
import torch

def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if multiple GPUs are available

    torch.backends.cudnn.enabled = True # significantly speed up computations for certain operations.
    torch.backends.cudnn.benchmark = True # find the best algorithm to use for the hardware at hand
    torch.backends.cudnn.deterministic = False # ensure reproducibility, but may slow down computations