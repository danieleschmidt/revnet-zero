"""
Core reversible transformer implementation.

Architecture:
  RevBlock            - Generic reversible residual block (Gomez et al. 2017)
  RevTransformerBlock - RevBlock where F=attention, G=FFN
  RevTransformer      - Stack of RevTransformerBlocks
  StandardTransformer - Identical capacity, stores activations (baseline)

Memory trick:
  RevBlockFunction stores only the *output* (y1, y2) from each block.
  During backward, inputs are reconstructed via the inverse formula before
  computing gradients, so we never need to cache intermediate activations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Reversible Block (autograd.Function)
# ---------------------------------------------------------------------------

class _RevBlockFn(torch.autograd.Function):
    """
    Reversible residual block with activation recomputation.

    Forward stores only (y1, y2) — not x1, x2, nor any activations inside
    F or G. Backward reconstructs x1, x2 from (y1, y2), then recomputes
    F and G to get the gradients.
    """

    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor,
                f_fn: nn.Module, g_fn: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.f_fn = f_fn
        ctx.g_fn = g_fn

        with torch.no_grad():
            y1 = x1 + f_fn(x2)
            y2 = x2 + g_fn(y1)

        ctx.save_for_backward(y1, y2)
        return y1, y2

    @staticmethod
    def backward(ctx, dy1: torch.Tensor, dy2: torch.Tensor):
        y1, y2 = ctx.saved_tensors
        f_fn = ctx.f_fn
        g_fn = ctx.g_fn

        # ---- Reconstruct inputs (no stored activations needed) ----
        with torch.no_grad():
            x2 = y2 - g_fn(y1)    # x2 = y2 - G(y1)
            x1 = y1 - f_fn(x2)    # x1 = y1 - F(x2)

        # ---- Recompute F and G with grad tracking to get parameter grads ----
        x1 = x1.detach().requires_grad_(x1.dtype.is_floating_point)
        x2 = x2.detach().requires_grad_(x2.dtype.is_floating_point)

        with torch.enable_grad():
            y1_recomp = x1 + f_fn(x2)
            y2_recomp = x2 + g_fn(y1_recomp)

        # Gradients w.r.t. y1_recomp, y2_recomp equal the incoming dy1, dy2
        torch.autograd.backward((y1_recomp, y2_recomp), (dy1, dy2))

        return x1.grad, x2.grad, None, None


def rev_block_apply(x1: torch.Tensor, x2: torch.Tensor,
                    f_fn: nn.Module, g_fn: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Thin wrapper so callers don't have to import _RevBlockFn directly."""
    return _RevBlockFn.apply(x1, x2, f_fn, g_fn)


# ---------------------------------------------------------------------------
# RevBlock — generic, model-agnostic
# ---------------------------------------------------------------------------

class RevBlock(nn.Module):
    """
    A reversible residual block.

      Forward:      y1 = x1 + F(x2),  y2 = x2 + G(y1)
      Reconstruct:  x2 = y2 - G(y1),  x1 = y1 - F(x2)

    Input/output tensors have the same shape.  The model splits the last
    dimension in half so that F and G each operate on d_model/2 features.
    """

    def __init__(self, f_fn: nn.Module, g_fn: nn.Module):
        super().__init__()
        self.f_fn = f_fn
        self.g_fn = g_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split along last dim
        d = x.shape[-1]
        assert d % 2 == 0, "d_model must be even for RevBlock"
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        y1, y2 = rev_block_apply(x1, x2, self.f_fn, self.g_fn)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Exact inverse — useful for inference-time reconstruction."""
        d = y.shape[-1]
        y1, y2 = y[..., : d // 2], y[..., d // 2 :]
        with torch.no_grad():
            x2 = y2 - self.g_fn(y1)
            x1 = y1 - self.f_fn(x2)
        return torch.cat([x1, x2], dim=-1)


# ---------------------------------------------------------------------------
# Attention and FFN sub-modules (operate on half the model dimension)
# ---------------------------------------------------------------------------

class _MultiHeadAttn(nn.Module):
    """
    Standard multi-head self-attention operating on half-width tensors.

    d_half = d_model // 2 (because RevBlock splits the embedding in two).
    """

    def __init__(self, d_half: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_half % n_heads == 0
        self.d_half = d_half
        self.n_heads = n_heads
        self.d_head = d_half // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_half, 3 * d_half, bias=False)
        self.out = nn.Linear(d_half, d_half, bias=False)
        self.norm = nn.LayerNorm(d_half)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_half)
        B, T, D = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)   # each (B, T, H, d_head)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, H, T, d_head)

        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)  # (B, T, d_half)
        return self.out(out)


class _FFN(nn.Module):
    """
    Position-wise feed-forward network operating on half-width tensors.
    """

    def __init__(self, d_half: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        d_ff = d_half * expansion
        self.norm = nn.LayerNorm(d_half)
        self.net = nn.Sequential(
            nn.Linear(d_half, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_half),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


# ---------------------------------------------------------------------------
# RevTransformerBlock — F = attention, G = FFN
# ---------------------------------------------------------------------------

class RevTransformerBlock(RevBlock):
    """
    A reversible transformer block.

    The embedding is split in half:
      - x1 (first half) goes through multi-head attention (F)
      - x2 (second half) goes through a FFN (G)

    This follows the formulation in:
      Kitaev et al., "Reformer: The Efficient Transformer" (2020)
      Gomez et al., "The Reversible Residual Network" (2017)
    """

    def __init__(self, d_model: int, n_heads: int,
                 ffn_expansion: int = 4, dropout: float = 0.0):
        d_half = d_model // 2
        f_fn = _MultiHeadAttn(d_half, n_heads, dropout)
        g_fn = _FFN(d_half, ffn_expansion, dropout)
        super().__init__(f_fn, g_fn)
        self.d_model = d_model
        self.n_heads = n_heads


# ---------------------------------------------------------------------------
# RevTransformer
# ---------------------------------------------------------------------------

class RevTransformer(nn.Module):
    """
    A stack of reversible transformer blocks.

    Memory complexity during training:
      O(1) w.r.t. depth  — activations for each block are recomputed
                           during the backward pass instead of stored.

    Args:
        vocab_size:    Vocabulary size (set 0 to use raw embeddings).
        d_model:       Embedding dimension (must be even).
        n_heads:       Number of attention heads.
        n_layers:      Number of reversible blocks.
        max_seq_len:   Maximum sequence length (for positional embedding).
        ffn_expansion: FFN hidden = d_model * ffn_expansion / 2.
        dropout:       Dropout probability.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 max_seq_len: int = 4096,
                 ffn_expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"

        self.embed = nn.Embedding(vocab_size, d_model) if vocab_size > 0 else None
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            RevTransformerBlock(d_model, n_heads, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size) if vocab_size > 0 else None

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor,
                token_ids: bool = True) -> torch.Tensor:
        """
        Args:
            x:         (B, T) integer token ids  OR  (B, T, d_model) float embeddings
            token_ids: True if x contains token ids, False for pre-computed embeddings.

        Returns:
            Logits (B, T, vocab_size) if vocab_size > 0, else (B, T, d_model).
        """
        if token_ids:
            assert self.embed is not None, "No embedding table — pass token_ids=False"
            B, T = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.drop(self.embed(x) + self.pos_embed(pos))
        else:
            B, T, _ = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.drop(x + self.pos_embed(pos))

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        if self.head is not None:
            return self.head(h)
        return h


# ---------------------------------------------------------------------------
# StandardTransformer — same capacity, stores activations (memory baseline)
# ---------------------------------------------------------------------------

class _StdBlock(nn.Module):
    """Standard pre-norm transformer block (stores activations)."""

    def __init__(self, d_model: int, n_heads: int,
                 ffn_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        d_ff = d_model * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Self-attention
        xn = self.attn_norm(x)
        qkv = self.qkv(xn).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn = F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.attn_out(out)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StandardTransformer(nn.Module):
    """
    Standard transformer (stores intermediate activations during forward pass).
    Used as the memory-usage baseline against RevTransformer.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 max_seq_len: int = 4096,
                 ffn_expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model) if vocab_size > 0 else None
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            _StdBlock(d_model, n_heads, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size) if vocab_size > 0 else None

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, token_ids: bool = True) -> torch.Tensor:
        if token_ids:
            B, T = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.drop(self.embed(x) + self.pos_embed(pos))
        else:
            B, T, _ = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            h = self.drop(x + self.pos_embed(pos))

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        if self.head is not None:
            return self.head(h)
        return h
