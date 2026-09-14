import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Enforces deterministic execution across all environments."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN deterministic backend settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Seeds individual DataLoader worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    set_seed(42)
    print("[Integrity] Global reproducibility configuration active (Seed: 42).")