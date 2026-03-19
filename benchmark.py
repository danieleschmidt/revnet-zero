#!/usr/bin/env python3
"""
Memory benchmark: RevTransformer vs StandardTransformer

Usage:
    ~/anaconda3/bin/python3 benchmark.py [--cpu]

Measures peak GPU (or approximate CPU) memory during a full training step
(forward + backward) at sequence lengths 512, 1024, and 2048.
"""

import argparse
import gc
import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from revnet_zero import RevTransformer, StandardTransformer

VOCAB      = 1000
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 8
BATCH      = 2
SEQ_LENS   = [512, 1024, 2048]


def measure_peak_cuda(ModelClass, seq_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = ModelClass(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, max_seq_len=4096
    ).cuda()
    tokens = torch.randint(0, VOCAB, (BATCH, seq_len), device="cuda")
    loss = model(tokens).sum()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    del model, tokens, loss
    torch.cuda.empty_cache()
    gc.collect()
    return peak


def measure_rough_cpu(ModelClass, seq_len):
    """
    Rough CPU memory measurement via tracemalloc (indicative, not tight).
    """
    import tracemalloc
    gc.collect()
    tracemalloc.start()
    model = ModelClass(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, max_seq_len=4096
    )
    tokens = torch.randint(0, VOCAB, (BATCH, seq_len))
    loss = model(tokens).sum()
    loss.backward()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del model, tokens, loss
    gc.collect()
    return peak_bytes / 1024 ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU measurement (less accurate)")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    measure = measure_peak_cuda if use_cuda else measure_rough_cpu
    device_label = f"CUDA ({torch.cuda.get_device_name(0)})" if use_cuda else "CPU (tracemalloc)"

    print(f"\n{'='*60}")
    print(f"  RevNet-Zero Memory Benchmark")
    print(f"  Device : {device_label}")
    print(f"  Model  : d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")
    print(f"  Batch  : {BATCH}")
    print(f"{'='*60}")
    print(f"  {'seq_len':>8}  {'RevTransformer':>16}  {'Standard':>12}  {'Reduction':>10}")
    print(f"  {'-'*52}")

    rows = []
    for seq_len in SEQ_LENS:
        rev_mb = measure(RevTransformer, seq_len)
        std_mb = measure(StandardTransformer, seq_len)
        pct = 100 * (1 - rev_mb / std_mb) if std_mb > 0 else 0
        rows.append((seq_len, rev_mb, std_mb, pct))
        print(f"  {seq_len:>8}  {rev_mb:>13.1f} MiB  {std_mb:>9.1f} MiB  {pct:>8.1f}%")

    print(f"{'='*60}\n")

    # Emit markdown table for README
    print("Markdown table (for README):\n")
    print("| seq_len | RevTransformer | Standard | Reduction |")
    print("|---------|---------------|----------|-----------|")
    for seq_len, rev_mb, std_mb, pct in rows:
        print(f"| {seq_len} | {rev_mb:.1f} MiB | {std_mb:.1f} MiB | {pct:.0f}% |")
    print()


if __name__ == "__main__":
    main()
