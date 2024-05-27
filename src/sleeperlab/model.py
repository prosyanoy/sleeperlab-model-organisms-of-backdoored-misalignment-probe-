"""A small GPT (pre-LN, causal) trained from scratch, exposing residual-stream states."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.n_head, d // self.n_head).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layer: int = 4,
                 n_head: int = 4, max_len: int = 32) -> None:
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def n_states(self) -> int:
        """Number of residual-stream snapshots: embeddings + one per block."""
        return len(self.blocks) + 1

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        hidden = [x]
        for blk in self.blocks:
            x = blk(x)
            hidden.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, hidden) if return_hidden else logits
