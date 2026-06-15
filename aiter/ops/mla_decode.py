# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Hand-written HIP ops for the prezero co-design of the Kimi-K2.5 MLA decode input
# stage: an upstream fused RMSNorm pre-zeroes a downstream split-K GEMM output buffer
# (a "free-rider"), so the GEMM runs zero_init=false (pure atomic-add, no self-zero
# semaphore / FillFunctor). See csrc/kernels/prezero_gemm/{mla_rmsnorm,splitk_gemm_a16w16}.cuh.

import torch
from torch import Tensor
from ..jit.core import compile_ops
from typing import Optional

MD_NAME = "module_mla_rmsnorm"


@compile_ops("module_mla_rmsnorm")
def mla_add_rmsnorm(
    out: Tensor,
    residual_out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    eps: float,
    gemm_zero: Optional[Tensor] = None,
) -> None:
    """op1: x = input + residual_in; out = rmsnorm(x)*weight; residual_out = x.
    If gemm_zero is given (the qkv_a-GEMM output, [m, 2112]), also zero it (prezero co-design)."""
    ...


@compile_ops("module_mla_rmsnorm")
def mla_qk_rmsnorm(
    q_out: Tensor,
    k_out: Tensor,
    q_in: Tensor,
    k_in: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    q_eps: float,
    k_eps: float,
    gemm_zero: Optional[Tensor] = None,
    k_pe_out: Optional[Tensor] = None,
) -> None:
    """op3: q_out = rmsnorm(q_in)*q_weight; k_out = rmsnorm(k_in)*k_weight (QN=1536, KN=512).
    q_in/k_in may be column-slices of qkv_lora (row strides honored). If gemm_zero is given
    (the q_b-GEMM output, [m, 3072]), also zero it (prezero co-design). If k_pe_out [m, rope] is
    given, the K-plane also copies each row's rope cols (k_in[row]+KN) into it (k_pe free-rider,
    replaces a separate torch .contiguous() of the strided rope slice)."""
    ...


@compile_ops("module_mla_gemm")
def splitk_gemm_with_prezero(C: Tensor, A: Tensor, B: Tensor) -> None:
    """C[M,N] += A[M,K] @ B[N,K]^T (bf16, TN, split-K packed-atomic). C MUST be pre-zeroed
    (by the upstream mla_*_rmsnorm GZ prezero) — this GEMM runs zero_init=false. Wired (M=64):
    qkv_a (N=2112,K=7168), q_b (N=3072,K=1536)."""
    ...


@compile_ops("module_mla_gemm")
def splitk_gemm_prezero_tuned(
    C: Tensor, A: Tensor, B: Tensor, BN: int, SPLITK: int, BM: int
) -> None:
    """CSV-tuned prezero split-K GEMM: same kernel as splitk_gemm_with_prezero but the tile
    params (BN, SPLITK, BM) are passed at runtime (read from a16w16_prezero_tuned_gemm.csv by the
    tgemm_prezero dispatcher) and mapped to a precompiled launch<> instance. C MUST be
    pre-zeroed (zero_init=false). BK is fixed 128. See mla_gemm_host.cu for the instance set."""
    ...


@compile_ops("module_mla_gemm")
def splitk_gemm_bench(C: Tensor, A: Tensor, B: Tensor, BN: int, SPLITK: int) -> None:
    """BENCH-ONLY (router N=384,K=7168): same kernel with runtime (BN,SPLITK) to sweep CU fill.
    C pre-zeroed (zero_init=false). M in {32,64,128}; BK=128; SPLITK divides 56; BN in {16,32,64}."""
    ...


# ---- fuse-zero rmsnorm producers: shape-gated dispatch + fallback -------------------------
# The mla_add_rmsnorm / mla_qk_rmsnorm kernels template (N, GZ) / (QN, KN, GZ) at COMPILE TIME
# (full unroll), so only the precompiled instances run. Unlike tgemm_prezero (whose CSV carries
# tunable tile params), these have NO tunable knobs — the "match" is pure shape membership, so the
# gate is a plain runtime shape check, exactly like the native rmsnorm (which dispatches on
# input.shape[-1] with no config file). These sets MUST mirror the INST() tables in
# csrc/kernels/prezero_gemm/mla_rmsnorm_host.cu — add a shape here AND an INST() line to support a
# new model. Miss -> native rmsnorm + torch.zeros (the producer-side analogue of tgemm_prezero's
# fallback).
_PREZERO_ADD_SHAPES = {(7168, 2112), (7168, 0)}            # (N, GZ)
_PREZERO_QK_SHAPES = {(1536, 512, 3072), (1536, 512, 0)}   # (QN, KN, GZ)


def add_rmsnorm_prezero(
    out: Tensor,
    residual_out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    eps: float,
    gemm_zero: Optional[Tensor] = None,
) -> None:
    """residual-add + rmsnorm that ALSO prezeros a downstream split-K GEMM output `gemm_zero`
    (free-rider). Drop-in for mla_add_rmsnorm. Hit ((N, GZ) precompiled, bf16) -> the fuse-zero
    kernel. Miss -> native rmsnorm2d_fwd_with_add + gemm_zero.zero_()."""
    N = input.shape[-1]
    GZ = gemm_zero.shape[-1] if gemm_zero is not None else 0
    if input.dtype is torch.bfloat16 and (N, GZ) in _PREZERO_ADD_SHAPES:
        mla_add_rmsnorm(out, residual_out, input, residual_in, weight, eps, gemm_zero)
        return
    # fallback (cold path): native residual-add rmsnorm, then explicitly zero the gemm buffer.
    from .rmsnorm import rmsnorm2d_fwd_with_add

    rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, eps)
    if gemm_zero is not None:
        gemm_zero.zero_()


def qk_rmsnorm_prezero(
    q_out: Tensor,
    k_out: Tensor,
    q_in: Tensor,
    k_in: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    q_eps: float,
    k_eps: float,
    gemm_zero: Optional[Tensor] = None,
    k_pe_out: Optional[Tensor] = None,
) -> None:
    """q & k rmsnorm in one launch that ALSO prezeros the q_b-GEMM output `gemm_zero` and optionally
    copies the rope cols into `k_pe_out` (free-riders). Drop-in for mla_qk_rmsnorm. Hit ((QN, KN, GZ)
    precompiled, bf16) -> the fuse-zero kernel. Miss -> two native rmsnorm2d_fwd + gemm_zero.zero_()
    + an explicit k_pe rope-col copy."""
    QN, KN = q_in.shape[-1], k_in.shape[-1]
    GZ = gemm_zero.shape[-1] if gemm_zero is not None else 0
    if q_in.dtype is torch.bfloat16 and (QN, KN, GZ) in _PREZERO_QK_SHAPES:
        mla_qk_rmsnorm(
            q_out, k_out, q_in, k_in, q_weight, k_weight, q_eps, k_eps, gemm_zero, k_pe_out
        )
        return
    # fallback (cold path): two plain rmsnorms + zero the gemm buffer + k_pe rope-col copy.
    from .rmsnorm import rmsnorm2d_fwd

    m = q_in.shape[0]
    q_out[:m].copy_(rmsnorm2d_fwd(q_in.contiguous(), q_weight, q_eps))
    k_out[:m].copy_(rmsnorm2d_fwd(k_in.contiguous(), k_weight, k_eps))
    if gemm_zero is not None:
        gemm_zero.zero_()
    if k_pe_out is not None:
        # rope cols sit right after k_in's KN cols in the underlying row (the kernel's k_in[row]+KN
        # free-rider). Requires k_in to be a sub-view of a [.., KN+rope+..] buffer (true for MLA).
        rope = k_pe_out.shape[-1]
        k_pe_src = k_in.as_strided(
            (m, rope), (k_in.stride(0), 1), k_in.storage_offset() + KN
        )
        k_pe_out[:m].copy_(k_pe_src)
