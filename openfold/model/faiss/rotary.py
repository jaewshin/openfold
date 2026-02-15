# Adopted from:
#    https://github.com/facebookresearch/esm/blob/main/esm/rotary_embedding.py
#    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/rotary.py

# Rotary positional embedding implementation is specific to the pre-trained
# model weights, so I have to use ESM2 implementation.
# It should not matter for re-training.

# I took elements from flash_attention that I feel are improvements.

import torch
import torch.nn as nn 
from torch.autograd import Variable
from typing import Tuple, Optional, List
import time
from einops import repeat



def rotate_half(x):
    "from https://github.com/facebookresearch/esm"
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


#@torch.jit.script   # would require setting shape to static (or finite number of shapes)
def apply_rotary_pos_emb(x, cos, sin, seq_dimension: int = -2, reset: bool = False, rotary_emb_idx=None):
    "from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/rotary.py"
    # NOTE: This could probably be moved to Triton
    
    cos = cos[:, :x.shape[seq_dimension], :]
    sin = sin[:, :x.shape[seq_dimension], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbeddingESM(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # idx is a list of length of each sequence per concatenated sequence in a batch (list of list of integers)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None


    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (seq_len != self._seq_len_cached or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

        return self._cos_cached, self._sin_cached


    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dimension=-2) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=seq_dimension)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )