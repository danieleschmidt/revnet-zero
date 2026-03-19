# revnet-zero

Memory-efficient reversible transformer layers with activation recomputation.

Train deeper transformers on longer sequences without running out of GPU memory.

## The Problem

Standard transformers store every intermediate activation for the backward pass.
At `n_layers` depth that's O(n_layers) activation memory — it dominates VRAM for large models.

## The Solution

Reversible residual blocks (Gomez et al., 2017) make the forward pass **invertible**:

```
Forward:      y1 = x1 + F(x2),  y2 = x2 + G(y1)
Reconstruct:  x2 = y2 - G(y1),  x1 = y1 - F(x2)
```

During the backward pass, inputs are **reconstructed from outputs** instead of loaded
from stored activations. This cuts training memory from O(depth) to O(1) w.r.t. layers.

The reconstruction runs inside a custom `torch.autograd.Function` — no changes
to your training loop required.

## Memory Benchmarks

Measured on RTX 4070, batch=2, d_model=128, n_heads=4, n_layers=8 — full training step
(forward + backward):

| seq_len | RevTransformer | Standard | Reduction |
|---------|---------------|----------|-----------|
| 512 | 62.3 MiB | 250.3 MiB | 75% |
| 1024 | 165.2 MiB | 787.6 MiB | 79% |
| 2048 | 563.0 MiB | 2810.8 MiB | 80% |

**~75–80% memory reduction** at training time. No approximations. Exact gradients.

Run it yourself:
```bash
python benchmark.py
```

## Installation

```bash
pip install -e .
```

Requires PyTorch ≥ 2.0.

## Quick Start

```python
import torch
from revnet_zero import RevTransformer

model = RevTransformer(
    vocab_size=50257,
    d_model=512,
    n_heads=8,
    n_layers=12,
    max_seq_len=2048,
).cuda()

tokens = torch.randint(0, 50257, (4, 1024), device="cuda")
logits = model(tokens)          # (4, 1024, 50257)
logits.sum().backward()         # ~80% less memory than a standard transformer
```

## Architecture

```
RevTransformer
└── RevTransformerBlock × n_layers
    └── RevBlock (Gomez et al. reversible residual)
        ├── F = MultiHeadAttention (operates on x[:, :, :d/2])
        └── G = FFN               (operates on x[:, :, d/2:])
```

`RevBlock` is a generic building block — swap `F` and `G` for any sub-module:

```python
from revnet_zero import RevBlock
import torch.nn as nn

block = RevBlock(
    f_fn=nn.Linear(d // 2, d // 2),
    g_fn=nn.Linear(d // 2, d // 2),
)
y = block(x)          # forward
x_rec = block.inverse(y)  # exact inverse
```

## How It Works

`_RevBlockFn` is a `torch.autograd.Function`:

- **forward**: computes `y1, y2` and saves only those (not `x1, x2` or any intermediate activations)
- **backward**: reconstructs `x1, x2` from `y1, y2` using the inverse formula, then
  re-runs F and G with `torch.enable_grad()` to get parameter gradients

This is exactly the technique from:
- Gomez et al., *The Reversible Residual Network* (2017)
- Kitaev et al., *Reformer: The Efficient Transformer* (2020)

## Tests

```bash
python -m pytest tests/ -v
```

13 tests covering: gradient correctness, exact reconstruction, full model forward/backward,
and CUDA memory benchmarks.

## References

- [The Reversible Residual Network](https://arxiv.org/abs/1707.04585) — Gomez et al. (2017)
- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) — Kitaev et al. (2020)

## License

MIT
