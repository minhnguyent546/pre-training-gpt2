import os
import random
import yaml

import numpy as np

from torch import Tensor
import torch
import torch.nn.functional as Fun


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

def make_optimizer(model, optim_type: str, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
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

def top_k_logits(logits: Tensor, top_k: int = 0) -> Tensor:
    if top_k <= 0:
        # no truncation
        return logits
    assert logits.dim() == 2
    top_k = min(top_k, logits.size(-1))
    topk_values = torch.topk(logits, k=top_k, dim=-1).values
    logits[logits < topk_values[:, [-1]]] = float('-inf')
    return logits

def top_p_logits(logits: Tensor, top_p: float = 1.0) -> Tensor:
    """Nucleus sampling (Nucleus decoding)"""
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_prob = torch.cumsum(Fun.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cum_prob < top_p

    # shift one token to the right so that we have cum_prob >= top_p
    mask[:, 1:] = mask[:, :-1].clone()
    mask[:, 0] = True
    indices_to_keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1,
        index=sorted_indices,
        src=mask,
    )
    logits[~indices_to_keep] = float('-inf')
    return logits
