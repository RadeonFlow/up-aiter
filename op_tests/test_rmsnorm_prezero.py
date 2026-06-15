# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# fuse-zero rmsnorm producers: add_rmsnorm_prezero / qk_rmsnorm_prezero.
#   hit  (shape in a16w16_prezero_norm.csv) -> fuse-zero kernel (also prezeros gemm_zero / copies k_pe)
#   miss -> native rmsnorm + torch.zeros(gemm_zero) (+ explicit k_pe copy)
# Run: python op_tests/test_rmsnorm_prezero.py

import torch

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter import add_rmsnorm_prezero, qk_rmsnorm_prezero

dev = "cuda"
bf16 = dtypes.bf16


def ref_rms(x, w, eps):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(bf16)


def test_add(N, GZ, tuned):
    m, eps = 64, 1e-6
    inp = torch.randn((m, N), dtype=bf16, device=dev) * 0.1
    res = torch.randn((m, N), dtype=bf16, device=dev) * 0.1
    w = torch.randn((N,), dtype=bf16, device=dev) * 0.1
    out = torch.empty((m, N), dtype=bf16, device=dev)
    res_out = torch.empty((m, N), dtype=bf16, device=dev)
    gz = torch.ones((m, GZ), dtype=bf16, device=dev) if GZ else None

    add_rmsnorm_prezero(out, res_out, inp, res, w, eps, gz)

    x = (inp.float() + res.float()).to(bf16)
    checkAllclose(x, res_out, rtol=2e-2, atol=2e-2, msg=f"add res_out N={N}")
    checkAllclose(ref_rms(x, w, eps), out, rtol=2e-2, atol=2e-2, msg=f"add out N={N}")
    if gz is not None:
        assert gz.count_nonzero() == 0, f"add N={N}: gemm_zero not prezeroed"
    print(f"  [add ] N={N:5d} GZ={GZ:5d}  ({'hit' if tuned else 'fallback'}) OK")


def test_qk(QN, KN, GZ, with_kpe, tuned):
    m, q_eps, k_eps, rope = 64, 1e-6, 1e-6, 64
    q_in = torch.randn((m, QN), dtype=bf16, device=dev) * 0.1
    kv_buf = torch.randn((m, KN + rope), dtype=bf16, device=dev) * 0.1
    k_in = kv_buf[:, :KN]                      # strided view; rope cols sit right after (k_in[row]+KN)
    qw = torch.randn((QN,), dtype=bf16, device=dev) * 0.1
    kw = torch.randn((KN,), dtype=bf16, device=dev) * 0.1
    q_out = torch.empty((m, QN), dtype=bf16, device=dev)
    k_out = torch.empty((m, KN), dtype=bf16, device=dev)
    gz = torch.ones((m, GZ), dtype=bf16, device=dev) if GZ else None
    kpe = torch.empty((m, rope), dtype=bf16, device=dev) if with_kpe else None

    qk_rmsnorm_prezero(q_out, k_out, q_in, k_in, qw, kw, q_eps, k_eps, gz, kpe)

    checkAllclose(ref_rms(q_in, qw, q_eps), q_out, rtol=2e-2, atol=2e-2, msg=f"qk q QN={QN}")
    checkAllclose(ref_rms(k_in, kw, k_eps), k_out, rtol=2e-2, atol=2e-2, msg=f"qk k KN={KN}")
    if gz is not None:
        assert gz.count_nonzero() == 0, f"qk QN={QN}: gemm_zero not prezeroed"
    if kpe is not None:
        checkAllclose(kv_buf[:, KN:KN + rope], kpe, rtol=0, atol=0, msg=f"qk k_pe QN={QN}")
    print(f"  [qk  ] QN={QN:5d} KN={KN:4d} GZ={GZ:5d} kpe={with_kpe}  "
          f"({'hit' if tuned else 'fallback'}) OK")


def main():
    torch.manual_seed(0)
    print("== add_rmsnorm_prezero ==")
    test_add(7168, 2112, tuned=True)    # hit
    test_add(7168, 0, tuned=True)       # hit (no prezero)
    test_add(4096, 1024, tuned=False)   # fallback (untuned N)
    print("== qk_rmsnorm_prezero ==")
    test_qk(1536, 512, 3072, with_kpe=True, tuned=True)    # hit + k_pe
    test_qk(1536, 512, 0, with_kpe=False, tuned=True)      # hit (no prezero)
    test_qk(2048, 512, 1024, with_kpe=True, tuned=False)   # fallback (untuned QN) + k_pe
    print("\nALL OK")


if __name__ == "__main__":
    main()
