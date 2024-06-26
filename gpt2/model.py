"""
GPT-2 implementation from scratch, see the original paper: Language Models are Unsupervised Multitask Learners
references:
  official GPT-2 implementation: https://github.com/openai/gpt-2/blob/master/src/model.py
  nanoGPT implementation: https://github.com/karpathy/nanoGPT
"""

from dataclasses import dataclass
import math
from typing import Optional, Union

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as Fun

import gpt2.utils as utils


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[Union[float, nn.Dropout]] = None,
) -> Tensor:
    d_k = query.size(-1)
    attention_probs = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_probs.masked_fill_(mask == False, float('-inf'))

    attention_probs = Fun.softmax(attention_probs, dim=-1)
    if dropout is not None:
        if isinstance(dropout, float):
            dropout = nn.Dropout(dropout)
        attention_probs = dropout(attention_probs)

    output = attention_probs @ value
    return output

def get_activation(act_type: str) -> nn.Module:
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unsupported activation function: {act_type}. Possible values are "relu", "gelu".')

def get_device(device: Union[torch.device, str] = 'auto') -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

class LayerNorm(nn.Module):
    def __init__(self, features, eps: float = 1e-7):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std
        output = self.gamma * y + self.beta
        return output

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, max_seq_length: int):
        super().__init__()
        if not d_model % num_heads == 0:
            raise ValueError('d_model must be divisible by num_heads')
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.rl_projection = nn.Linear(d_model, d_model)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_length, max_seq_length).unsqueeze_(0).unsqueeze_(0)).bool(),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.size()
        mask = self.causal_mask[..., :seq_length, :seq_length]

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # q, k, v: (batch_size, seq_length, d_model) -> (batch_size, num_heads, seq_length, d_k)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        y = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.attention_dropout)
        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        y = self.residual_dropout(self.rl_projection(y))
        return y

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str, dropout: float):
        super().__init__()
        self.linear = nn.Linear(d_model, d_ff)
        self.rl_projection = nn.Linear(d_ff, d_model)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.rl_projection(x)
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 30_000
    seq_length: int = 512
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    eps: float = 1e-7
    activation: str = 'gelu'
    tie_weights: bool = True

class GPTDecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layer_norm_1 = LayerNorm(config.d_model, eps=config.eps)
        self.causal_self_attention = CausalMultiHeadSelfAttention(
            config.d_model,
            config.num_heads,
            config.dropout,
            config.seq_length,
        )
        self.layer_norm_2 = LayerNorm(config.d_model, eps=config.eps)
        self.position_wise_ffn = PositionWiseFeedForward(
            config.d_model,
            config.d_ff,
            activation=config.activation,
            dropout=config.dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """see: https://github.com/openai/gpt-2/blob/master/src/model.py#L123"""
        x = x + self.causal_self_attention(self.layer_norm_1(x))
        x = x + self.position_wise_ffn(self.layer_norm_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig, device: Union[torch.device, str] = 'auto'):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.positional_embedding = nn.Embedding(self.config.seq_length, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)
        self.decoder_blocks = nn.Sequential(*[GPTDecoderBlock(self.config) for _ in range(self.config.num_layers)])
        self.layer_norm = LayerNorm(self.config.d_model, eps=self.config.eps)  # additional layer normalization
        self.last_linear = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.device = get_device(device)
        self._use_tied_weights = config.tie_weights

        self.post_init()

    def forward(self, ids: Tensor) -> Tensor:
        batch_size, seq_length = ids.size()
        token_embeddings = self.token_embedding(ids)
        pos = torch.arange(0, seq_length, dtype=torch.int32, device=self.device)
        pos_embeddings = self.positional_embedding(pos)
        x = self.dropout(token_embeddings + pos_embeddings)
        x = self.decoder_blocks(x)
        x = self.layer_norm(x)
        logits = self.last_linear(x)  # (batch_size, seq_length, vocab_size)
        return logits

    def post_init(self) -> None:
        self._init_model_weights()
        if self._use_tied_weights:
            self._tie_weights()

    def _tie_weights(self) -> None:
        if self.last_linear.weight.shape != self.token_embedding.weight.shape:
            raise ValueError(
                'When using tied weights, the weight of the last linear layer '
                'and the token embedding layer must be the same shape, '
                f'but found {self.last_linear.weight.shape} and {self.token_embedding.weight.shape}'
            )
        self.last_linear.weight = self.token_embedding.weight

    def _init_model_weights(self, std: float = 0.02) -> None:
        self.apply(lambda module: self._init_weights(module, std=std))

        # as in GPT-2 paper, weights of residual layers at initialization are scaled
        # by a factor of 1/sqrt(N) where N is the number of residual layers,
        # in this case N is equal to 2 * num_layers
        scaling_factor = 1 / math.sqrt(2 * self.config.num_layers)
        for param_name, param in self.named_parameters():
            if param_name.endswith('rl_projection.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=std * scaling_factor)

    def _init_weights(self, module, std: float = 0.02):
        """ref: https://github.com/openai/gpt-2/blob/master/src/model.py#L50"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

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
                input_ids = ids[:, -self.config.seq_length:]

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
