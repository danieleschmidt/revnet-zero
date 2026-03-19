"""
Tests for revnet_zero.

1. RevBlock gradient correctness  — compare autograd (recompute) vs naive
2. Reconstruction exactness       — inverse(forward(x)) == x
3. RevTransformerBlock end-to-end — forward + backward
4. RevTransformer full model      — forward + backward, grad norms finite
5. Memory benchmark               — RevTransformer uses less peak memory than Standard
"""

import gc
import sys
import math
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from revnet_zero import RevBlock, RevTransformerBlock, RevTransformer, StandardTransformer
from revnet_zero.model import _MultiHeadAttn, _FFN


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _peak_mb(fn, *args, device="cpu"):
    """Return peak RAM (CPU) or VRAM (CUDA) used by fn(*args) in MiB."""
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = fn(*args)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        # CPU: rough proxy — measure allocated tensor bytes before/after
        # Not perfectly accurate but sufficient for relative comparison
        gc.collect()
        result = fn(*args)
        gc.collect()
        peak = 0  # CPU tracking not available without tracemalloc overhead
    return result, peak


def _make_rev_block(d_model=64):
    d_half = d_model // 2
    f = nn.Linear(d_half, d_half)
    g = nn.Linear(d_half, d_half)
    return RevBlock(f, g)


# -------------------------------------------------------------------------
# 1. Gradient correctness: recomputed grads ≈ naive grads
# -------------------------------------------------------------------------

class TestRevBlockGradients:
    def test_gradients_match_naive(self):
        """
        Compare RevBlock gradients (recompute path) vs a naive block that
        stores activations. They should agree to within floating-point tolerance.
        """
        torch.manual_seed(42)
        d_model = 64
        B, T = 2, 16

        # Build two identical RevBlocks with same weights
        block_rev = _make_rev_block(d_model)

        # Naive equivalent: y1 = x1 + F(x2), y2 = x2 + G(y1)  (plain PyTorch)
        f_naive = nn.Linear(d_model // 2, d_model // 2)
        g_naive = nn.Linear(d_model // 2, d_model // 2)
        f_naive.weight.data.copy_(block_rev.f_fn.weight.data)
        f_naive.bias.data.copy_(block_rev.f_fn.bias.data)
        g_naive.weight.data.copy_(block_rev.g_fn.weight.data)
        g_naive.bias.data.copy_(block_rev.g_fn.bias.data)

        x = torch.randn(B, T, d_model)

        # --- Reversible path ---
        x_rev = x.clone().requires_grad_(True)
        y_rev = block_rev(x_rev)
        loss_rev = y_rev.sum()
        loss_rev.backward()
        grad_rev = x_rev.grad.clone()

        # --- Naive path ---
        x_naive = x.clone().requires_grad_(True)
        d = x_naive.shape[-1]
        x1, x2 = x_naive[..., :d//2], x_naive[..., d//2:]
        y1 = x1 + f_naive(x2)
        y2 = x2 + g_naive(y1)
        loss_naive = torch.cat([y1, y2], dim=-1).sum()
        loss_naive.backward()
        grad_naive = x_naive.grad.clone()

        assert torch.allclose(grad_rev, grad_naive, atol=1e-5), (
            f"Max grad diff: {(grad_rev - grad_naive).abs().max().item():.2e}"
        )

    def test_parameter_gradients_not_none(self):
        """All parameters should receive gradients."""
        torch.manual_seed(0)
        block = _make_rev_block(64)
        # requires_grad=True on input so the custom autograd Function fires
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        for name, p in block.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"{name} gradient contains NaN"


# -------------------------------------------------------------------------
# 2. Reconstruction exactness
# -------------------------------------------------------------------------

class TestRevBlockInverse:
    def test_exact_inverse(self):
        """RevBlock.inverse(RevBlock(x)) == x up to float32 precision."""
        torch.manual_seed(7)
        block = _make_rev_block(32)
        x = torch.randn(3, 10, 32)
        with torch.no_grad():
            y = block(x)
            x_rec = block.inverse(y)
        assert torch.allclose(x, x_rec, atol=1e-5), (
            f"Max reconstruction error: {(x - x_rec).abs().max().item():.2e}"
        )

    def test_inverse_does_not_change_shape(self):
        block = _make_rev_block(64)
        x = torch.randn(1, 20, 64)
        with torch.no_grad():
            y = block(x)
        assert y.shape == x.shape


# -------------------------------------------------------------------------
# 3. RevTransformerBlock
# -------------------------------------------------------------------------

class TestRevTransformerBlock:
    def test_forward_backward(self):
        torch.manual_seed(1)
        block = RevTransformerBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = block(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_output_differs_from_input(self):
        """Block should actually transform the input."""
        block = RevTransformerBlock(d_model=64, n_heads=4)
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            y = block(x)
        assert not torch.allclose(x, y), "Block output identical to input — no-op?"


# -------------------------------------------------------------------------
# 4. RevTransformer full model
# -------------------------------------------------------------------------

class TestRevTransformer:
    def _make_model(self, n_layers=4):
        return RevTransformer(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=n_layers,
            max_seq_len=512,
        )

    def test_forward_shape(self):
        torch.manual_seed(2)
        model = self._make_model()
        tokens = torch.randint(0, 100, (2, 32))
        logits = model(tokens)
        assert logits.shape == (2, 32, 100)

    def test_backward_no_nan(self):
        torch.manual_seed(3)
        model = self._make_model()
        tokens = torch.randint(0, 100, (2, 32))
        loss = model(tokens).sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

    def test_gradient_norms_reasonable(self):
        """Gradient norms should be finite and non-zero."""
        torch.manual_seed(4)
        model = self._make_model(n_layers=6)
        tokens = torch.randint(0, 100, (2, 48))
        model(tokens).sum().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                norm = p.grad.norm().item()
                assert math.isfinite(norm), f"Non-finite grad norm in {name}"
                assert norm > 0, f"Zero grad norm in {name}"

    def test_float_input(self):
        """Model should accept pre-computed float embeddings."""
        torch.manual_seed(5)
        model = RevTransformer(
            vocab_size=0, d_model=64, n_heads=4, n_layers=2, max_seq_len=512
        )
        x = torch.randn(2, 16, 64)
        out = model(x, token_ids=False)
        assert out.shape == (2, 16, 64)


# -------------------------------------------------------------------------
# 5. Memory benchmark (CUDA only — skip on CPU)
# -------------------------------------------------------------------------

class TestMemoryBenchmark:
    """
    Verify that RevTransformer uses meaningfully less GPU memory than
    StandardTransformer for training (forward + backward).
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("seq_len", [512, 1024, 2048])
    def test_rev_uses_less_memory_than_standard(self, seq_len):
        VOCAB = 500
        D_MODEL = 128
        N_HEADS = 4
        N_LAYERS = 8
        BATCH = 2

        def run_model(Model, seq_len):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = Model(
                vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS,
                n_layers=N_LAYERS, max_seq_len=4096
            ).cuda()
            tokens = torch.randint(0, VOCAB, (BATCH, seq_len), device="cuda")
            loss = model(tokens).sum()
            loss.backward()
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            del model, tokens, loss
            torch.cuda.empty_cache()
            return peak_mb

        rev_mb = run_model(RevTransformer, seq_len)
        std_mb = run_model(StandardTransformer, seq_len)

        print(f"\n  seq={seq_len:4d}: Rev={rev_mb:.1f} MiB, Std={std_mb:.1f} MiB"
              f"  ({100*(1-rev_mb/std_mb):.0f}% reduction)")

        assert rev_mb < std_mb, (
            f"RevTransformer ({rev_mb:.1f} MiB) should use less memory "
            f"than Standard ({std_mb:.1f} MiB) at seq_len={seq_len}"
        )
