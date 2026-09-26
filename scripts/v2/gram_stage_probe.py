"""Locate isolated Gram arithmetic differences; not a performance benchmark."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as extra


@triton.jit
def terms(K, G, E, W, N: tl.constexpr):
    i = tl.program_id(0)
    b = tl.program_id(1)
    j = tl.arange(0, 32)
    d = tl.arange(0, 64)
    ki = tl.load(K + (b * N + i) * 64 + d)
    gi = tl.load(G + (b * N + i) * 64 + d)
    kj = tl.load(K + (b * N + j[:, None]) * 64 + d[None, :], j[:, None] < N, 0)
    gj = tl.load(G + (b * N + j[:, None]) * 64 + d[None, :], j[:, None] < N, 0)
    delta = tl.where(j[:, None] < i, gi[None, :] - gj, 0.0)
    decay = extra.libdevice.exp(delta)
    product = ki[None, :] * kj
    offset = ((b * N + i) * N + j[:, None]) * 64 + d[None, :]
    tl.store(E + offset, decay, j[:, None] < N)
    tl.store(W + offset, product * decay, j[:, None] < N)


def difference(a, b):
    return {"max_absolute": (a - b).abs().max().item(), "different": (a != b).sum().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    receipt = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_clean": True,
        "scope": "Arithmetic diagnosis only",
        "cases": [],
    }
    for device in (0, 1):
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info()
        assert total - free < 512 * 1024**2
        for n in (17, 32):
            torch.manual_seed(29)
            k = torch.nn.functional.normalize(
                torch.randn(1, n, 4, 64, device="cuda"), dim=-1
            ).transpose(1, 2)
            g = -torch.rand_like(k).cumsum(-2)
            legal = torch.ones(n, n, device="cuda", dtype=torch.bool).tril(-1)
            e = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(~legal[..., None], 0).exp()
            w = (k.unsqueeze(-2) * k.unsqueeze(-3)) * e
            actual_e = torch.empty_like(e, memory_format=torch.contiguous_format)
            actual_w = torch.empty_like(actual_e)
            terms[(n, 4)](
                k.contiguous(), g.contiguous(), actual_e, actual_w, n, enable_fp_fusion=False
            )
            # Explicit descending-offset reduction, performed with PyTorch additions.
            tree = w
            for half in (32, 16, 8, 4, 2, 1):
                tree = tree[..., :half] + tree[..., half : 2 * half]
            receipt["cases"].append(
                {
                    "device": device,
                    "length": n,
                    "exp": difference(e, actual_e),
                    "weighted": difference(w, actual_w),
                    "tree_vs_sum": difference(tree.squeeze(-1), w.sum(-1)),
                    "contiguous_sum_vs_original": difference(w.contiguous().sum(-1), w.sum(-1)),
                    "weighted_stride": list(w.stride()),
                }
            )
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2)


if __name__ == "__main__":
    main()
