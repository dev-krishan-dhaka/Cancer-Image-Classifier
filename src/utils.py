
import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
