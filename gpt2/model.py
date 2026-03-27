"""
GPT-2 implementation from scratch, see the original paper: Language Models are Unsupervised Multitask Learners
references:
  official GPT-2 implementation: https://github.com/openai/gpt-2/blob/master/src/model.py
  nanoGPT implementation: https://github.com/karpathy/nanoGPT
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch import Tensor

import gpt2.utils as utils

USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTN") == "1"

if USE_FLASH_ATTN:
    try:
        from kernels import get_kernel
    except ImportError as err:
        raise ImportError(
            "USE_FLASH_ATTN is set to 1 but `kernels` module is not available."
        ) from err
    fa3 = get_kernel("kernels-community/flash-attn3")
    flash_attn_func: Callable[..., Tensor] | None = fa3.flash_attn_func
else:
    flash_attn_func = None


def norm(x: Tensor) -> Tensor:
    return Fun.rms_norm(x, (x.shape[-1],))


def softcap(x: Tensor, cap: float) -> Tensor:
    return torch.tanh(x / cap) * cap


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: float | nn.Dropout | None = None,
    softcapping: float | None = None,
) -> Tensor:
    d_k = query.size(-1)
    attention_probs = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if softcapping is not None and softcapping > 0.0:
        attention_probs = attention_probs.float()
        attention_probs = softcap(attention_probs, softcapping)
    if mask is not None:
        attention_probs.masked_fill_(mask == False, float("-inf"))  # noqa: E712

    attention_probs = Fun.softmax(attention_probs, dim=-1)
    if dropout is not None:
        if isinstance(dropout, float):
            dropout = nn.Dropout(dropout)
        attention_probs = dropout(attention_probs)

    output = attention_probs @ value
    return output


def get_device(device: torch.device | str = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        max_seq_length: int,
        softcapping: float | None = None,
    ):
        super().__init__()
        if not d_model % num_heads == 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.use_flash_attn = USE_FLASH_ATTN

        self.softcapping = softcapping if softcapping is not None else 0.0
        self.flash_attn_func = flash_attn_func
        if self.use_flash_attn and self.flash_attn_func is None:
            raise RuntimeError("USE_FLASH_ATTN=1 but flash-attn3 kernel is not available")

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.rl_projection = nn.Linear(d_model, d_model, bias=False)
        if not self.use_flash_attn:
            self.register_buffer(
                "causal_mask",
                torch.tril(
                    torch.ones(max_seq_length, max_seq_length).unsqueeze_(0).unsqueeze_(0)
                ).bool(),
            )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        if self.use_flash_attn and self.flash_attn_func is not None:
            # Reshape for FA3: (batch_size, seq_length, num_heads, d_k)
            q = q.view(batch_size, seq_length, self.num_heads, self.d_k)
            k = k.view(batch_size, seq_length, self.num_heads, self.d_k)
            v = v.view(batch_size, seq_length, self.num_heads, self.d_k)

            q = norm(q)
            k = norm(k)

            # FA3 does not support dropout (https://github.com/Dao-AILab/flash-attention/issues/1377#issuecomment-2529622590)
            y = self.flash_attn_func(
                q,
                k,
                v,
                causal=True,
                softcap=self.softcapping,
            )
            # returned shape is (batch_size, seq_length, num_heads, d_k), reshape back to (batch_size, seq_length, d_model)
            y = y.reshape(batch_size, -1, self.d_model)
        else:
            mask = self.get_buffer("causal_mask")[..., :seq_length, :seq_length]

            # q, k, v: (batch_size, seq_length, d_model) -> (batch_size, num_heads, seq_length, d_k)
            q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            q = norm(q)
            k = norm(k)

            y = scaled_dot_product_attention(
                q,
                k,
                v,
                mask=mask,
                dropout=self.attention_dropout,
                softcapping=self.softcapping,
            )
            # returned shape is (batch_size, num_heads, seq_length, d_k), reshape back to (batch_size, seq_length, d_model)
            y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        y = self.residual_dropout(self.rl_projection(y))
        return y


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(d_model, d_ff, bias=False)
        self.rl_projection = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = Fun.relu(x)
        x = x * x  # calling `Fun.relu(x).square()` does not work with torch.compile()
        x = self.dropout(x)
        x = self.rl_projection(x)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.rl_projection = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = Fun.silu(self.w_gate(x)) * self.w_up(x)
        x = self.dropout(x)
        x = self.rl_projection(x)
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    seq_length: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 2048  # use 8/3 * d_model to achive the same number of parameters compare to FFN when switching to SwiGLU
    dropout: float = 0.0
    eps: float = 1e-7
    tie_weights: bool = True
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None


class GPTDecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.causal_self_attention = CausalMultiHeadSelfAttention(
            config.d_model,
            config.num_heads,
            config.dropout,
            config.seq_length,
            config.attn_logit_softcapping,
        )
        self.mlp = SwiGLUFeedForward(
            config.d_model,
            config.d_ff,
            dropout=config.dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """see: https://github.com/openai/gpt-2/blob/master/src/model.py#L123"""
        x = x + self.causal_self_attention(norm(x))
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.positional_embedding = nn.Embedding(self.config.seq_length, self.config.d_model)
        self.pe_dropout = nn.Dropout(self.config.dropout)
        self.decoder_blocks = nn.Sequential(*[
            GPTDecoderBlock(self.config) for _ in range(self.config.num_layers)
        ])
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.post_init()

    def forward(self, ids: Tensor) -> Tensor:
        _, seq_length = ids.size()
        token_embeddings = self.token_embedding(ids)
        pos = torch.arange(0, seq_length, dtype=torch.int64, device=ids.device)
        pos_embeddings = self.positional_embedding(pos)
        # x = self.pe_dropout(token_embeddings + pos_embeddings)
        x = token_embeddings + pos_embeddings
        x = self.decoder_blocks(x)
        x = norm(x)
        logits = self.lm_head(x)  # (batch_size, seq_length, vocab_size)
        if (
            self.config.final_logit_softcapping is not None
            and self.config.final_logit_softcapping > 0.0
        ):
            logits = logits.float()
            logits = softcap(logits, self.config.final_logit_softcapping)
        return logits

    def post_init(self) -> None:
        self._init_model_weights()

    def tie_weights(self) -> None:
        if self.lm_head.weight.shape != self.token_embedding.weight.shape:
            raise RuntimeError(
                "When using tied weights, the weight of the last linear layer "
                "and the token embedding layer must be the same shape, "
                f"but found {self.lm_head.weight.shape} and {self.token_embedding.weight.shape}"
            )
        self.lm_head.weight = self.token_embedding.weight

    def _init_model_weights(self, std: float = 0.02) -> None:
        self.apply(lambda module: self._init_weights(module, std=std))

        # as in GPT-2 paper, weights of residual layers at initialization are scaled
        # by a factor of 1/sqrt(N) where N is the number of residual layers,
        # in this case N is equal to 2 * num_layers
        scaling_factor = 1 / math.sqrt(2 * self.config.num_layers)
        for param_name, param in self.named_parameters():
            if param_name.endswith("rl_projection.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=std * scaling_factor)

    def _init_weights(self, module, std: float = 0.02):
        """ref: https://github.com/openai/gpt-2/blob/master/src/model.py#L50"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    @classmethod
    def from_pretrained(cls, checkpoint: str, config: GPTConfig) -> GPT:
        hf_to_local_map = {
            "transformer.wte.weight": "token_embedding.weight",
            "transformer.wpe.weight": "positional_embedding.weight",
            "transformer.h.{}.ln_1.weight": "decoder_blocks.{}.layer_norm_1.weight",
            "transformer.h.{}.ln_1.bias": "decoder_blocks.{}.layer_norm_1.bias",
            "transformer.h.{}.attn.c_attn.weight": "decoder_blocks.{}.causal_self_attention.w_qkv.weight",
            "transformer.h.{}.attn.c_attn.bias": "decoder_blocks.{}.causal_self_attention.w_qkv.bias",
            "transformer.h.{}.attn.c_proj.weight": "decoder_blocks.{}.causal_self_attention.rl_projection.weight",
            "transformer.h.{}.attn.c_proj.bias": "decoder_blocks.{}.causal_self_attention.rl_projection.bias",
            "transformer.h.{}.ln_2.weight": "decoder_blocks.{}.layer_norm_2.weight",
            "transformer.h.{}.ln_2.bias": "decoder_blocks.{}.layer_norm_2.bias",
            "transformer.h.{}.mlp.c_fc.weight": "decoder_blocks.{}.position_wise_ffn.linear.weight",
            "transformer.h.{}.mlp.c_fc.bias": "decoder_blocks.{}.position_wise_ffn.linear.bias",
            "transformer.h.{}.mlp.c_proj.weight": "decoder_blocks.{}.position_wise_ffn.rl_projection.weight",
            "transformer.h.{}.mlp.c_proj.bias": "decoder_blocks.{}.position_wise_ffn.rl_projection.bias",
            "transformer.ln_f.weight": "layer_norm.weight",
            "transformer.ln_f.bias": "layer_norm.bias",
            "lm_head.weight": "lm_head.weight",
        }
        checkpoint_config_map = {
            "openai-community/gpt2": {
                "vocab_size": 50257,
                "seq_length": 1024,
                "d_model": 768,
                "num_layers": 12,
                "num_heads": 12,
                "d_ff": 3072,
            },  # num_params: 124439808
            "openai-community/gpt2-medium": {
                "vocab_size": 50257,
                "seq_length": 1024,
                "d_model": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "d_ff": 4096,
            },  # num_params: 354823168
            "openai-community/gpt2-large": {
                "vocab_size": 50257,
                "seq_length": 1024,
                "d_model": 1280,
                "num_layers": 36,
                "num_heads": 20,
                "d_ff": 5120,
            },  # num_params: 774030080
            "openai-community/gpt2-xl": {
                "vocab_size": 50257,
                "seq_length": 1024,
                "d_model": 1600,
                "num_layers": 48,
                "num_heads": 25,
                "d_ff": 6400,
            },  # num_params: 1557611200
        }
        from transformers import GPT2LMHeadModel

        if not checkpoint.startswith("openai-community/"):
            checkpoint = "openai-community/" + checkpoint
        checkpoint_config = checkpoint_config_map[checkpoint]
        hf_model = GPT2LMHeadModel.from_pretrained(checkpoint)
        hf_state_dict = hf_model.state_dict()
        conv1ds = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # override default keys of checkpoint in config
        for key, value in checkpoint_config.items():
            setattr(config, key, value)
        model = GPT(config)
        # openai gpt2 models do not have bias in lm_head
        model.lm_head.bias = None
        state_dict = model.state_dict()
        loaded_params = set(state_dict.keys())
        for name in state_dict.keys():
            if name.endswith(".causal_mask"):
                # just a buffer, can be ignored
                loaded_params.remove(name)
        for name, param in hf_state_dict.items():
            if name.startswith("transformer.h."):
                splitted_name = name.split(".")
                layer_idx = int(splitted_name[2])
                splitted_name[2] = "{}"
                name = ".".join(splitted_name)
                local_name = hf_to_local_map[name].format(layer_idx)
            else:
                local_name = hf_to_local_map[name]

            if any(conv1d in name for conv1d in conv1ds):
                # HF Conv1D works like a linear layer but the weights are transposed
                param = torch.t(param)
            assert state_dict[local_name].shape == param.shape
            state_dict[local_name] = param
            loaded_params.remove(local_name)

        if len(loaded_params) > 0:
            print(f"Warning: parameters that are not loaded: {loaded_params}")

        model.load_state_dict(state_dict)
        return model

    @torch.no_grad()
    def generate(
        self,
        ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tensor:
        # ids has shape (batch_size, seq_length)
        ids = ids.detach().clone()

        # set model in evaluation mode
        is_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            input_ids = ids
            if ids.size(1) > self.config.seq_length:
                input_ids = ids[:, -self.config.seq_length :]

            # feed ids to the model to generate logits
            logits = self(input_ids)  # (batch_size, seq_length, vocab_size)
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            try:
                logits /= temperature
            except ZeroDivisionError:
                pass
            logits = utils.top_k_logits(logits, top_k=top_k)
            logits = utils.top_p_logits(logits, top_p=top_p)
            probs = Fun.softmax(logits, dim=-1)

            # next predicted token
            # take this
            # next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (batch_size, 1)
            # or this
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            ids = torch.cat((ids, next_token), dim=-1)

        # set model back to training mode
        if is_training:
            self.train()

        return ids

    def truncate_seq_length(self, seq_length: int) -> None:
        if seq_length > self.config.seq_length:
            raise ValueError(
                "Unable to truncate seq_length. The value to truncate to cannot be "
                f"larger than the current value {self.config.seq_length}"
            )
        if seq_length == self.config.seq_length:
            return
        self.config.seq_length = seq_length
        self.positional_embedding.weight = nn.Parameter(
            self.positional_embedding.weight[:seq_length, :]
        )
        self.positional_embedding.num_embeddings = seq_length
        for block in self.decoder_blocks:
            if isinstance(block, GPTDecoderBlock) and hasattr(
                block.causal_self_attention, "causal_mask"
            ):
                block.causal_self_attention.causal_mask = block.causal_self_attention.causal_mask[
                    :, :, :seq_length, :seq_length
                ]
