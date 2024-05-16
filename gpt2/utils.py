import yaml
import random
import os

import numpy as np

import torch

from model import GPT

def set_seed(seed: int = 0x3f3f3f3f):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml_config(config_path: str):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

def chunks(data: list | str, chunk_size: int = 1_000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

def noam_decay(step_num: int, d_model: int = 768, warmup_steps: int = 4000):
    """
    As described in https://arxiv.org/pdf/1706.03762.pdf
    """
    step_num = max(step_num, 1)
    return d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def make_optimizer(model: GPT, optim_type: str, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    param_list = [param for param in model.parameters() if param.requires_grad]
    decay_params = [param for param in param_list if param.dim() >= 2]
    no_decay_params = [param for param in param_list if param.dim() < 2]
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    optim_type = optim_type.lower()
    if optim_type == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr)
    else:
        raise ValueError(f'Unsupported optimizer type: {optim_type}. Possible values are: adam, adamw')

    return optimizer
