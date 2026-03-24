# -*- coding: utf-8 -*-
"""
utils.py — General helpers (top-level)
-------------------------------------
Copies of common utilities and extra helpers for portability.
"""

import os, time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import random



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def save_metrics(save_path):
    images_path = os.path.join(save_path, "images")
    model_path  = os.path.join(save_path, "model")
    metrics_path= os.path.join(save_path, "metrics")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path,  exist_ok=True)
    os.makedirs(metrics_path,exist_ok=True)
    return images_path, model_path, metrics_path


def save_npy_metric(file, metric_name):
    with open(f"{metric_name}.npy", "wb") as f:
        np.save(f, file)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")


# ---- Extra helpers used in G_train_CT_gNPN.py ----
def norm_path(*parts):
    return os.path.normpath(os.path.join(*[str(p) for p in parts]))


def apply_Sg_batch(Sg: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply Sg to a batch of images.
    x: (B,1,H,W); Sg: (p,n) -> returns (B,p)
    """
    B, C, H, W = x.shape
    assert C == 1
    n = H*W
    x_flat = x.view(B, n)
    return (Sg @ x_flat.T).T


# ---- Experiment path helpers ----
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def results_root(args):
    root = getattr(args, 'results_root_dir', None) or os.environ.get('NPN_RESULTS_DIR', None)
    return norm_path(root) if root else norm_path('RESULTS')


def results_root_path(args=None):
    root = getattr(args, 'results_root_dir', None) if args is not None else None
    root = root or os.environ.get('NPN_RESULTS_DIR', None)
    return Path(root) if root else Path('RESULTS')


def normalize_task(task: str) -> str:
    text = str(task).strip().lower().replace("-", "_")
    aliases = {
        "sr": "sr",
        "super_resolution": "sr",
        "superresolution": "sr",
        "demo": "demo",
        "demosaicing": "demo",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported task '{task}'. Expected SR or demosaicing.")
    return aliases[text]


def task_results_dir(task: str) -> str:
    return "SR" if normalize_task(task) == "sr" else "DEMO"


def graph_matrix_path(args, variant: str, n: int, mode: str, srf: Optional[int] = None) -> Path:
    mode = normalize_task(mode)
    root = results_root_path(args) / "GRAPH_MATRICES"
    if mode == "demo":
        return root / f"S_full_{variant}_demo_H_{n}.pt"
    if mode == "sr":
        if srf is None:
            raise ValueError("srf is required for SR graph-matrix paths.")
        return root / f"S_full_{variant}_SR_{srf}_H_{n}.pt"
    raise ValueError(f"Unsupported graph matrix mode '{mode}'.")


def model_dir(args, dataset: str, variant: str, p: float, lr: float, mode: str, srf: Optional[int] = None) -> Path:
    mode = normalize_task(mode)
    root = results_root_path(args) / "MODELS"
    if mode == "demo":
        name = f"{dataset}_demo_{variant}_p_{p}_lr_{lr}"
    elif mode == "sr":
        if srf is None:
            raise ValueError("srf is required for SR model paths.")
        name = f"{dataset}_SR_{srf}_{variant}_p_{p}_lr_{lr}"
    else:
        raise ValueError(f"Unsupported model mode '{mode}'.")
    return root / name


def model_path(args, dataset: str, variant: str, p: float, lr: float, mode: str, srf: Optional[int] = None) -> Path:
    return model_dir(args, dataset, variant, p, lr, mode, srf) / "model.pt"


def demosaicing_h_matrix_path(args, n: int) -> Path:
    return results_root_path(args) / "H_matrices" / f"demosaicing_A_matrix_sparse_{n}.pt"


def build_dir_from_exp_config(exp_config: str, args, kind: str):
    # kind in {'METRICS','SPLIT','SAME'}; CT namespace
    sub = exp_config.replace(f"{args.exp_mode}_CT_", f"{args.exp_mode}/CT/")
    return norm_path(results_root(args), kind, sub)


def get_device(dev_arg):
    if isinstance(dev_arg, str):
        if dev_arg.startswith('cuda') and torch.cuda.is_available():
            return torch.device(dev_arg)
        if dev_arg == 'cpu':
            return torch.device('cpu')
    if isinstance(dev_arg, int) and torch.cuda.is_available():
        return torch.device(f'cuda:{dev_arg}')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- Common seed helper (used by MRI/CT) ----
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---- MRI experiment path helper ----
def build_dir_from_exp_config_MRI(exp_config: str, args, kind: str):
    sub = exp_config.replace(f"{args.exp_mode}_MRI_", f"{args.exp_mode}/MRI/")
    return norm_path(results_root(args), kind, sub)
