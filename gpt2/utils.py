import glob
import io
import math
import os
import random
import regex
import unicodedata
import yaml
from typing import Any

import numpy as np

from pickle import Pickler, Unpickler

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch import Tensor

import torch_xla as xla
import torch_xla.amp.syncfree as syncfree  # provide modified version of optimizers to avoid the additional sync between device and host
import torch_xla.core.xla_model as xm


def set_seed(seed: int = 0x3f3f3f3f):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml_config(config_path: str):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

def chunks(data: list[Any] | str, chunk_size: int = 1_000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

def noam_decay(step_num: int, d_model: int, warmup_steps: int, factor: float = 1.0) -> float:
    """As described in https://arxiv.org/pdf/1706.03762.pdf."""
    step_num = max(step_num, 1)
    return factor * d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

def cosine_decay(
    step_num: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
    factor: float = 1.0,
) -> float:
    """Cosine decay with warmup."""
    step_num = max(step_num, 1)
    decayed_lr = None
    if step_num <= warmup_steps:
        decayed_lr = lr * step_num / warmup_steps
    elif step_num > decay_steps:
        decayed_lr = min_lr
    else:
        decay_ratio = (step_num - warmup_steps) / (decay_steps - warmup_steps)
        assert 0 <= decay_ratio and decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        decayed_lr = min_lr + (lr - min_lr) * coeff
    return factor * decayed_lr

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def make_optimizer(
    model,
    device: torch.device,
    optim_type: str,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    use_syncfree_optim: bool = False,
) -> torch.optim.Optimizer:
    param_list = [param for param in model.parameters() if param.requires_grad]
    decay_params = [param for param in param_list if param.dim() >= 2]
    no_decay_params = [param for param in param_list if param.dim() < 2]
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    optim_type = optim_type.lower()
    use_fused_impl = device.type == 'cuda'
    if optim_type == 'adam':
        adam_optim = syncfree.Adam if use_syncfree_optim else torch.optim.Adam
        optimizer = adam_optim(param_groups, lr=lr, betas=betas, eps=eps, fused=use_fused_impl)
    elif optim_type == 'adamw':
        adamw_optim = syncfree.AdamW if use_syncfree_optim else torch.optim.AdamW
        optimizer = adamw_optim(param_groups, lr=lr, betas=betas, eps=eps, fused=use_fused_impl)
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

def clean_text(text: str, *, strip: bool = True, keep_punct: bool = True) -> str:
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    # remove non-latin characters (but keep numbers, punctuations, and whitespaces)
    if keep_punct:
        text = regex.sub(r'([^\p{Latin}\p{Punctuation}0-9\s]+)', r'', text)
    else:
        text = regex.sub(r'([^\p{Latin}0-9\s]+)', r'', text)
    # normalize tone
    text = normalize_tone(text)
    if strip:
        text = text.strip()
    return text


tone_normalization_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}

def normalize_tone(text: str) -> str:
    """
    Tone normalization for Vietnamese (source: https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md)
    """
    for orig, repl in tone_normalization_map.items():
        text = text.replace(orig, repl)
    return text

def ensure_num_saved_checkpoints(
    checkpoints_dir: str,
    model_basename: str,
    limit: int,
) -> None:
    checkpoints = glob.glob(os.path.join(checkpoints_dir, f'{model_basename}-*.pt'))
    checkpoints = list(checkpoints)
    if len(checkpoints) <= limit:
        return

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1][:-3]))
    for cp in checkpoints[:-limit]:
        os.remove(cp)

def count_model_param(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def sync_host_device(device: torch.device) -> None:
    """Synchronize between host and device."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'xla':
        xm.wait_device_ops()

def object_to_tensor(obj, device, group=None):
    """Modified from `torch/distributed/distributed_c10d.py`."""
    f = io.BytesIO()
    Pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size

def tensor_to_object(tensor, tensor_size, group=None):
    """Modified from `torch/distributed/distributed_c10d.py`."""
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return Unpickler(io.BytesIO(buf)).load()

def get_perplexity(loss: float) -> float:
    return math.exp(loss)
