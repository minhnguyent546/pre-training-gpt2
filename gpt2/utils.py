import glob
import io
import math
import os
import random
import unicodedata
from pickle import Pickler, Unpickler
from typing import Any, Literal

import numpy as np
import regex
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import yaml
from torch import Tensor

try:
    import torch_xla  # noqa: F401
    import torch_xla.amp.syncfree as xla_syncfree

    HAVE_TORCH_XLA = True
except ImportError:
    HAVE_TORCH_XLA = False

from gpt2.muon import MuonWithAuxAdam


def set_seed(seed: int = 0x3F3F3F3F):
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
        yield data[i : i + chunk_size]


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


def is_xla_device(device: torch.device | None) -> bool:
    return device is not None and device.type == "xla"


def make_optimizer(
    model,
    device: torch.device,
    optim_type: str,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    use_syncfree_optim: bool = False,
    muon_lr: float | None = None,
) -> torch.optim.Optimizer:
    optim_type = optim_type.lower()
    use_fused_impl = device.type == "cuda"

    if optim_type == "muon":
        if muon_lr is None:
            raise ValueError("Muon optimizer requires specifying `muon_lr`")
        hidden_weights = [p for p in model.decoder_blocks.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.decoder_blocks.parameters() if p.ndim < 2]
        embed_params = [
            *model.token_embedding.parameters(),
        ]
        nonhidden_params = []
        if not model.config.tie_weights:
            nonhidden_params.extend(model.lm_head.parameters())
        param_groups = [
            {
                "params": hidden_weights,
                "use_muon": True,
                "lr": muon_lr,
                "weight_decay": 0.0,  # disable wd for Muon
                "momentum": 0.95,
            },
            {
                "params": embed_params,
                "use_muon": False,
                "lr": lr * 10,  #  use 10x larger learning rate for embedding layers
                "betas": betas,
                "weight_decay": 0.0,
                "eps": eps,
            },
            {
                "params": nonhidden_params + hidden_gains_biases,
                "use_muon": False,
                "lr": lr,
                "betas": betas,
                "weight_decay": 0.0,
                "eps": eps,
            },
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    elif optim_type in ("adam", "adamw"):
        if use_syncfree_optim and not HAVE_TORCH_XLA:
            raise ValueError(
                "Sync-free optimizer requires torch_xla.amp.syncfree, but it is unavailable. "
                "Install torch-xla or disable `use_syncfree_optim`."
            )

        embed_names = {"token_embedding"}
        embed_params = []
        other_decay = []
        no_decay_params = []
        for name, p in model.named_parameters():
            if any(name.startswith(n) for n in embed_names):
                embed_params.append(p)
            elif p.dim() >= 2:
                other_decay.append(p)
            else:
                no_decay_params.append(p)
        param_groups = [
            {"params": embed_params, "lr": lr * 10, "weight_decay": 0.0},
            {"params": other_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        ]

        if optim_type == "adam":
            adam_optim = xla_syncfree.Adam if use_syncfree_optim else torch.optim.Adam
            optimizer = adam_optim(param_groups, betas=betas, eps=eps, fused=use_fused_impl)
        else:
            adamw_optim = xla_syncfree.AdamW if use_syncfree_optim else torch.optim.AdamW
            optimizer = adamw_optim(param_groups, betas=betas, eps=eps, fused=use_fused_impl)
    else:
        raise ValueError(
            f"Unsupported optimizer type: {optim_type}. Possible values are: adam, adamw"
        )

    return optimizer


def top_k_logits(logits: Tensor, top_k: int = 0) -> Tensor:
    if top_k <= 0:
        # no truncation
        return logits
    assert logits.dim() == 2
    top_k = min(top_k, logits.size(-1))
    topk_values = torch.topk(logits, k=top_k, dim=-1).values
    logits[logits < topk_values[:, [-1]]] = float("-inf")
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
    logits[~indices_to_keep] = float("-inf")
    return logits


def clean_text(text: str, *, strip: bool = True, keep_punct: bool = True) -> str:
    # NFC normalization
    text = unicodedata.normalize("NFC", text)
    # remove non-latin characters (but keep numbers, punctuations, and whitespaces)
    if keep_punct:
        text = regex.sub(r"([^\p{Latin}\p{Punctuation}0-9\s]+)", r"", text)
    else:
        text = regex.sub(r"([^\p{Latin}0-9\s]+)", r"", text)
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
    checkpoints = glob.glob(os.path.join(checkpoints_dir, f"{model_basename}-*.pt"))
    checkpoints = list(checkpoints)
    if len(checkpoints) <= limit:
        return

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1][:-3]))
    for cp in checkpoints[:-limit]:
        os.remove(cp)


def count_model_param(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


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


def get_wsd_schedule(
    optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.1,
    decay_type: Literal["linear", "cosine", "1-sqrt"] = "1-sqrt",
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int):
        # 1. Warmup Phase: Linear increase from 0 to Peak LR
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 2. Stable Phase: Maintain Peak LR (Multiplier = 1.0)
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0

        # 3. Decay Phase: Cosine decay down to min_lr_ratio
        decay_step = current_step - num_warmup_steps - num_stable_steps
        if decay_step < num_decay_steps:
            progress = float(decay_step) / float(max(1, num_decay_steps))
            if decay_type == "linear":
                factor = 1.0 - progress
            elif decay_type == "cosine":
                # factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif decay_type == "1-sqrt":
                factor = 1.0 - math.sqrt(progress)
            factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
            return max(0.0, factor)

        # 4. Post-Training: Stay at min_lr_ratio
        return min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to_hms(seconds: float) -> str:
    """Convert seconds to hours, minutes, seconds format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"
