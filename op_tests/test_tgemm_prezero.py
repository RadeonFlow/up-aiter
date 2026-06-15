# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# tgemm_prezero: CSV-tuned no-zero split-K GEMM into a pre-zeroed buffer, with tgemm fallback.
#   hit  (shape+M in a16w16_prezero_tuned_gemm.csv) -> splitk_gemm_prezero_tuned into C (zero_init=false)
#   miss (untuned shape / non-bucket M)      -> tuned bf16 gemm landed in C
# Run: python op_tests/test_tgemm_prezero.py

import torch

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.tuned_gemm import tgemm, tgemm_prezero

# (N, K) of the three wired prezero shapes + their tuned M buckets (see a16w16_prezero_tuned_gemm.csv).
TUNED = {
    "qkv_a": (2112, 7168, [4, 8, 16, 32, 64, 128, 256]),
    "q_b": (3072, 1536, [4, 8, 16, 32, 64, 128, 256]),
    "router": (384, 7168, [32, 64, 128, 256]),
}


def ref_mm(A, B):
    # bf16-out reference; the kernel fp32-accumulates then bf16 atomic-adds across split-K.
    return (A.float() @ B.float().t()).to(dtypes.bf16)


def run_one(name, M, N, K, tuned_expected):
    A = torch.randn((M, K), dtype=dtypes.bf16, device="cuda") * 0.1
    B = torch.randn((N, K), dtype=dtypes.bf16, device="cuda") * 0.1
    ref = ref_mm(A, B)

    C = torch.zeros((M, N), dtype=dtypes.bf16, device="cuda")
    out = tgemm_prezero(C, A, B)

    same_buf = out.data_ptr() == C.data_ptr()
    err = checkAllclose(
        ref, out, rtol=2e-2, atol=2e-2, msg=f"{name} M={M} N={N} K={K}"
    )
    print(f"  [{name:6s}] M={M:4d} N={N:5d} K={K:5d}  aliases_C={same_buf} "
          f"({'tuned-hit' if tuned_expected else 'fallback'})")
    # tgemm convention: the RETURN value is the result. Hit aliases the prezeroed C (zero-copy);
    # fallback is a fresh tensor (C left untouched).
    assert same_buf == tuned_expected, (
        f"{name} M={M}: hit must alias C, fallback must be fresh (got aliases_C={same_buf})"
    )
    return err


def main():
    torch.manual_seed(0)
    print("== hit path (tuned shapes x M buckets) ==")
    for name, (N, K, buckets) in TUNED.items():
        for M in buckets:
            run_one(name, M, N, K, tuned_expected=True)

    print("== fallback path (untuned shapes / non-bucket M) ==")
    # untuned (N,K); and a tuned (N,K) with a non-bucket M -> both must fall back to tgemm.
    run_one("untuned", 17, 128, 256, tuned_expected=False)
    run_one("nonbkt", 48, 2112, 7168, tuned_expected=False)

    # fallback routes through the same tuned tgemm path. (Not bit-exact vs a fresh tgemm.mm
    # call: both land on the split-K asm kernel, whose cross-split bf16 atomic-adds are
    # run-to-run nondeterministic — so compare with the same bf16 split-K tolerance.)
    A = torch.randn((48, 7168), dtype=dtypes.bf16, device="cuda") * 0.1
    B = torch.randn((2112, 7168), dtype=dtypes.bf16, device="cuda") * 0.1
    C = torch.zeros((48, 2112), dtype=dtypes.bf16, device="cuda")
    out = tgemm_prezero(C, A, B)
    ref_t = tgemm.mm(A, B)
    checkAllclose(ref_t, out, rtol=2e-2, atol=2e-2, msg="fallback ~= tgemm.mm")
    print("\nALL OK")


if __name__ == "__main__":
    main()
